"""Serial, resumable simulator collection with fail-closed development screening.

Each attempt has its own immutable directory, source hashes, seed, logs and
verdict. Accepted manifests never contain failed or interrupted attempts.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path(
    os.environ.get("G1_OFFICIAL_ROOT", str(ROOT.parent / "Binjiang_Competition"))
).resolve()
WAYPOINTS = OFFICIAL / "src/examples/dataCollection/unitree_g1/my_waypoint_button/marked"
TASKS = {
    "press": {
        "waypoint": WAYPOINTS / "my_waypoint_press_01.yaml",
        "request": ROOT / "configs/goal_stop_text.json",
        "prompt": "按压停止按钮",
        "target": "cabinet02.stop",
    },
    "rotate": {
        "waypoint": ROOT / "configs/waypoint_rotate_clearance.yaml",
        "request": ROOT / "configs/goal_rotate_demo.json",
        "prompt": "旋转旋钮",
        "target": "cabinet02.knob",
    },
    "toggle": {
        "waypoint": WAYPOINTS / "my_waypoint_toggle_01.yaml",
        "request": ROOT / "configs/goal_toggle_demo.json",
        "prompt": "拨动拨杆式按钮",
        "target": "cabinet03.toggle",
    },
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    tmp.replace(path)


@contextmanager
def batch_lock(root):
    # An abandoned lock deliberately requires inspection, rather than risking two controllers.
    lock = Path(root) / ".running.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    try:
        yield
    finally:
        lock.unlink()


def variant(template, seed):
    rng = random.Random(seed)
    data = copy.deepcopy(template)
    scale = rng.uniform(0.92, 1.08)
    offsets = [rng.uniform(-0.008, 0.008), rng.uniform(-0.008, 0.008), rng.uniform(-0.006, 0.006)]
    for segment in data["segments"]:
        segment["steps"] = max(100, round(segment["steps"] * scale))
    # Only the initial free-space approach pose varies; contact waypoints stay fixed.
    data["segments"][0]["r_target_b"] = [
        x + d for x, d in zip(data["segments"][0]["r_target_b"], offsets)
    ]
    return data, {
        "seed": seed,
        "duration_scale": scale,
        "first_waypoint_offset_m": offsets,
        "scene_position_randomized": False,
        "lighting_randomized": False,
    }


def plan_batch(root, per_task, seed):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    snapshots = root / "inputs"
    snapshots.mkdir()
    for name, path in {
        "catalog": ROOT / "configs/task_catalog.json",
        "criteria": ROOT / "configs/task_success_dev.json",
        "scene": ROOT / "configs/official_g1_buttons.yaml",
    }.items():
        shutil.copy2(path, snapshots / (name + path.suffix))
    tasks = {}
    for name, task in TASKS.items():
        shutil.copy2(task["request"], snapshots / f"{name}_goal.json")
        shutil.copy2(task["waypoint"], snapshots / f"{name}.yaml")
        tasks[name] = {
            "prompt": task["prompt"],
            "target": task["target"],
            "source_waypoint": str(task["waypoint"]),
            "source_sha256": digest(task["waypoint"]),
        }
    plan = {
        "schema_version": 1,
        "per_task": per_task,
        "seed": seed,
        "fps": 24,
        "tasks": tasks,
        "max_attempts_per_task": per_task + max(5, per_task // 4),
        "validation_per_task": max(1, per_task // 5) if per_task > 1 else 0,
        "limits": {"minimum_free_gib": 5, "process_timeout_s": 240, "max_consecutive_failures": 3},
        "quality": {
            "minimum_unique_frame_ratio": 0.5,
            "max_position_step_m": 0.10,
            "max_orientation_step_rad": 1.2,
        },
        "input_hashes": {p.name: digest(p) for p in snapshots.iterdir()},
        "official_commit": subprocess.check_output(
            ["git", "-C", str(OFFICIAL), "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    write(root / "plan.json", plan)
    return plan


def screen(run, goal, catalog, criteria, limits, expected_prompt=None):
    """Inspect real videos, parquet and joint traces; rejects unknown/missing evidence."""
    import numpy as np
    import pyarrow.parquet as pq
    from task_success import SuccessEvaluator
    from validate_official_dataset import validate

    run = Path(run)
    reasons = []
    try:
        validate(run / "dataset_verified", run / "validation")
        validation = read(run / "validation/validation.json")
        if not validation["all_checks_passed"]:
            reasons.append("dataset_checks_failed")
        if expected_prompt is not None and {t["task"] for t in validation["tasks"]} != {
            expected_prompt
        }:
            reasons.append("task_label_mismatch")
        for key, v in validation["videos"].items():
            if (
                v["unique_decoded_images"] / max(v["decoded_frames"], 1)
                < limits["minimum_unique_frame_ratio"]
            ):
                reasons.append("stale_video:" + key)
        summary = read(run / "audit/physics_summary.json")
        if summary["physics_finite"] is not True or summary["steps"] <= 0:
            reasons.append("invalid_physics")
        evaluator = SuccessEvaluator(goal, catalog, criteria)
        trace = [
            json.loads(s)
            for s in (run / "audit/physics_trace.jsonl").read_text(encoding="utf-8").splitlines()
            if s.strip()
        ]
        for row in trace:
            evaluator.update(row)
        evaluation = evaluator.finalize()
        if evaluation["proxy_success"] is not True:
            reasons.append("task:" + evaluation["status"])
        if not trace or abs(len(trace) - validation["frames"]) > 2:
            reasons.append("audit_video_count_mismatch")
        if trace and abs(trace[-1]["simulation_time_s"] - summary["last_simulation_time_s"]) > 0.1:
            reasons.append("incomplete_audit_tail")
        paths = list((run / "dataset_verified/data").rglob("*.parquet"))
        state = np.array(pq.read_table(paths[0])["observation.state"].to_pylist())
        max_position = max(
            float(np.max(np.linalg.norm(np.diff(state[:, a : a + 3], axis=0), axis=1)))
            for a in [0, 7]
        )
        max_angle = 0.0
        for a in [3, 10]:
            q = state[:, a : a + 4]
            q = q / np.linalg.norm(q, axis=1, keepdims=True)
            max_angle = max(
                max_angle,
                float(np.max(2 * np.arccos(np.clip(np.abs(np.sum(q[1:] * q[:-1], axis=1)), 0, 1)))),
            )
        if max_position > limits["max_position_step_m"]:
            reasons.append("position_discontinuity")
        if max_angle > limits["max_orientation_step_rad"]:
            reasons.append("orientation_discontinuity")
        evidence = {
            "goal": goal,
            "evaluation": evaluation,
            "trace": str((run / "audit/physics_trace.jsonl").resolve()),
            "dataset": str((run / "dataset_verified").resolve()),
            "audit_sha256": digest(run / "audit/physics_trace.jsonl"),
        }
        write(run / "goal_evaluation.json", evidence)
        metrics = {
            "frames": validation["frames"],
            "max_position_step_m": max_position,
            "max_orientation_step_rad": max_angle,
        }
    except Exception as exc:
        reasons.append(type(exc).__name__ + ": " + str(exc))
        metrics = {}
    verdict = {
        "accepted": not reasons,
        "reasons": reasons,
        "metrics": metrics,
        "criterion": "development_proxy_not_official",
        "checked_at_unix": time.time(),
    }
    write(run / "quality.json", verdict)
    return verdict


def summarize(root, plan, status="running", message=None):
    records = [read(p) for p in sorted((root / "attempts").glob("*/result.json"))]
    counts = {
        name: sum(r["task"] == name and r["accepted"] for r in records) for name in plan["tasks"]
    }
    manifests = {
        split: {"schema_version": 1, "datasets": []} for split in ["all", "train", "validation"]
    }
    seen = collections.Counter()
    for record in records:
        if not record["accepted"]:
            continue
        name = record["task"]
        task = plan["tasks"][name]
        split = (
            "train" if seen[name] < plan["per_task"] - plan["validation_per_task"] else "validation"
        )
        seen[name] += 1
        item = {
            "name": name,
            "path": record["directory"] + "/dataset_verified",
            "expected_prompt": task["prompt"],
            "target_id": task["target"],
            "evaluation": record["directory"] + "/goal_evaluation.json",
        }
        manifests["all"]["datasets"].append(item)
        manifests[split]["datasets"].append(item)
    for split, value in manifests.items():
        write(root / f"manifest_{split}.json", value)
    summary = {
        "status": status,
        "message": message,
        "requested_per_task": plan["per_task"],
        "accepted_per_task": counts,
        "attempts": len(records),
        "accepted": sum(counts.values()),
        "rejected": sum(not r["accepted"] for r in records),
        "frames": sum(r.get("metrics", {}).get("frames", 0) for r in records if r["accepted"]),
        "train_episodes": len(manifests["train"]["datasets"]),
        "validation_episodes": len(manifests["validation"]["datasets"]),
        "official_success_verified": False,
        "split_scope": "held-out trajectory variants in the same scene, not unseen-scene generalization",
        "records": records,
    }
    write(root / "summary.json", summary)
    return summary


def run_batch(root, plan, max_new=None):
    from task_goal import resolve_goal

    inputs = root / "inputs"
    for name, sha in plan["input_hashes"].items():
        if digest(inputs / name) != sha:
            raise ValueError("Frozen batch inputs changed: " + name)
    catalog = read(inputs / "catalog.json")
    criteria = read(inputs / "criteria.json")
    env = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONIOENCODING": "utf-8"}
    failures = 0
    new = 0
    for attempt in range(plan["max_attempts_per_task"]):
        for task_index, (name, task) in enumerate(plan["tasks"].items()):
            summary = summarize(root, plan)
            if summary["accepted_per_task"][name] >= plan["per_task"]:
                continue
            run = root / "attempts" / f"{attempt:03d}_{name}"
            if (run / "result.json").exists():
                continue
            if max_new is not None and new >= max_new:
                return summarize(
                    root,
                    plan,
                    "checkpoint",
                    "Invocation attempt limit reached; resume with --resume",
                )
            if (root / "STOP").exists():
                return summarize(root, plan, "stopped", "STOP file detected")
            if shutil.disk_usage(root).free < plan["limits"]["minimum_free_gib"] * 2**30:
                return summarize(root, plan, "blocked", "Insufficient disk space")
            try:
                with socket.create_connection(("localhost", 50051), timeout=3):
                    pass
            except OSError:
                return summarize(
                    root, plan, "blocked", "OrcaLab runtime localhost:50051 unavailable"
                )
            if run.exists():
                # A prior interrupted attempt is preserved and excluded, never overwritten.
                record = {
                    "task": name,
                    "directory": run.relative_to(root).as_posix(),
                    "accepted": False,
                    "reasons": ["interrupted_attempt"],
                }
                write(run / "result.json", record)
                continue
            run.mkdir(parents=True)
            template = yaml.safe_load((inputs / f"{name}.yaml").read_text(encoding="utf-8"))
            waypoint, variation = variant(template, plan["seed"] + attempt * 3 + task_index)
            (run / "waypoint.yaml").write_text(
                yaml.safe_dump(waypoint, allow_unicode=True, sort_keys=False), encoding="utf-8"
            )
            write(run / "variation.json", variation)
            cmd = [
                sys.executable,
                str(ROOT / "scripts/run_official_demo.py"),
                "--official-root",
                str(OFFICIAL),
                "--audit-output",
                str(run / "audit"),
                "--",
                "--task_config",
                str(inputs / "scene.yaml"),
                "--agent_name",
                "g1_pick",
                "--waypoint_files",
                str(run / "waypoint.yaml"),
                "--lerobot_out",
                str(run / "dataset_verified"),
                "--repo_id",
                f"local/batch_{name}_{attempt}",
                "--fps",
                "24",
                "--num_episodes",
                "1",
                "--clock",
                "sim",
                "--cameras",
                "head,wrist_r",
                "--cam_resolution",
                "960x1280",
                "--joint_strip",
                "on",
                "--strip_col",
                "off",
                "--time_step",
                ".001",
                "--frame_skip",
                "5",
                "--track_log_every",
                "500",
            ]
            write(run / "command.json", cmd)
            print(f"START {attempt:03d} {name} seed={variation['seed']}", flush=True)
            try:
                with (run / "capture.log").open("w", encoding="utf-8") as log:
                    completed = subprocess.run(
                        cmd,
                        cwd=WAYPOINTS.parents[1],
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=plan["limits"]["process_timeout_s"],
                    )
                if completed.returncode:
                    raise RuntimeError("Capture exit code " + str(completed.returncode))
                goal = resolve_goal(read(inputs / f"{name}_goal.json"), catalog)
                verdict = screen(run, goal, catalog, criteria, plan["quality"], task["prompt"])
            except Exception as exc:
                verdict = {
                    "accepted": False,
                    "reasons": [type(exc).__name__ + ": " + str(exc)],
                    "metrics": {},
                }
                write(run / "quality.json", verdict)
            record = {
                **verdict,
                "task": name,
                "directory": run.relative_to(root).as_posix(),
                "variation": variation,
            }
            write(run / "result.json", record)
            failures = 0 if verdict["accepted"] else failures + 1
            new += 1
            summary = summarize(root, plan)
            print(
                json.dumps(
                    {
                        k: summary[k]
                        for k in ["accepted_per_task", "attempts", "rejected", "frames"]
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if failures >= plan["limits"]["max_consecutive_failures"]:
                return summarize(
                    root,
                    plan,
                    "blocked",
                    "Three consecutive failed attempts; inspect logs before resuming",
                )
    summary = summarize(root, plan)
    done = all(v >= plan["per_task"] for v in summary["accepted_per_task"].values())
    return summarize(
        root,
        plan,
        "complete" if done else "incomplete",
        "Attempt budget exhausted" if not done else None,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--per-task", type=int, default=20)
    p.add_argument("--seed", type=int, default=20261005)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--max-new", type=int)
    args = p.parse_args()
    if args.per_task < 1 or (args.max_new is not None and args.max_new < 1):
        p.error("Counts must be positive")
    root = args.output.resolve()
    plan = read(root / "plan.json") if args.resume else plan_batch(root, args.per_task, args.seed)
    with batch_lock(root):
        result = (
            summarize(root, plan, "planned")
            if args.plan_only
            else run_batch(root, plan, args.max_new)
        )
    print(
        json.dumps({k: v for k, v in result.items() if k != "records"}, ensure_ascii=False),
        flush=True,
    )
    return 0 if result["status"] in ["complete", "planned", "checkpoint"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
