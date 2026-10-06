"""Run the official G1 entry point with read-only physics audit, preserving its controls."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "Binjiang_Competition",
    )
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--entry", choices=["scripted", "replay"], default="scripted")
    parser.add_argument("--goal-request", type=Path)
    parser.add_argument("--goal-catalog", type=Path)
    parser.add_argument("--goal-criteria", type=Path)
    args, forwarded = parser.parse_known_args()
    evaluator = None
    goal = None
    goal_args = [args.goal_request, args.goal_catalog, args.goal_criteria]
    if any(goal_args):
        if not all(goal_args):
            parser.error("--goal-request, --goal-catalog and --goal-criteria are required together")
        from task_goal import resolve_goal
        from task_success import SuccessEvaluator

        def read_json(p):
            return json.loads(p.read_text(encoding="utf-8"))

        catalog = read_json(args.goal_catalog)
        goal = resolve_goal(read_json(args.goal_request), catalog)
        evaluator = SuccessEvaluator(goal, catalog, read_json(args.goal_criteria))
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    root = args.official_root.resolve()
    folder = root / "src/examples/dataCollection/unitree_g1"
    name = (
        "g1_pick_osc_collection_scripted_lerobot.py"
        if args.entry == "scripted"
        else "g1_pick_osc_replay_lerobot.py"
    )
    for p in [root / "src", folder, folder.parent / "common"]:
        sys.path.insert(0, str(p))
    import mujoco
    from envs.dataCollection.dataCollection_env import DataCollectionEnv

    args.audit_output.mkdir(parents=True, exist_ok=False)
    step_original = DataCollectionEnv.step
    count = 0
    joints = {}
    right_positions = []
    first_sim_time = last_sim_time = next_sample = None
    audit_file = (args.audit_output / "physics_trace.jsonl").open("w", encoding="utf-8")
    finite = True
    started = time.time_ns()

    def audited_step(env, action):
        nonlocal count, first_sim_time, last_sim_time, next_sample, finite
        result = step_original(env, action)
        model, data = env.gym._mjModel, env.gym._mjData
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            finite = False
            raise RuntimeError("Non-finite physics state during official demonstration")
        count += 1
        last_sim_time = float(data.time)
        if first_sim_time is None:
            first_sim_time = last_sim_time
            next_sample = last_sim_time
            for i in range(model.njnt):
                n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
                if not n.startswith("g1_pick_") and int(model.jnt_type[i]) in (2, 3):
                    v = float(data.qpos[model.jnt_qposadr[i]])
                    joints[n] = {
                        "address": int(model.jnt_qposadr[i]),
                        "initial": v,
                        "min": v,
                        "max": v,
                        "final": v,
                        "unit": "m" if int(model.jnt_type[i]) == 2 else "rad",
                    }
        for item in joints.values():
            v = float(data.qpos[item["address"]])
            item["min"], item["max"], item["final"] = min(item["min"], v), max(item["max"], v), v
        obs = result[0]
        pos = np.asarray(obs.get("/action/end/position", []))
        r_pos = pos[1].tolist() if pos.shape == (2, 3) else None
        if r_pos:
            right_positions.append(r_pos)
        if last_sim_time >= next_sample:
            audit_sample = {
                "step": count,
                "simulation_time_s": last_sim_time,
                "right_end_position_b": r_pos,
                "scene_joint_position": {n: j["final"] for n, j in joints.items()},
                "contact_count": int(data.ncon),
            }
            audit_file.write(json.dumps(audit_sample, allow_nan=False) + "\n")
            if evaluator is not None:
                evaluator.update(audit_sample)
            next_sample += 1 / 24
        return result

    DataCollectionEnv.step = audited_step
    sys.argv = [str(folder / name)] + forwarded
    try:
        # Official __main__ uses os._exit(0), which bypasses audit finalizers and
        # masks failures. Import its namespace and call its main instead.
        namespace = runpy.run_path(str(folder / name), run_name="official_demo_entry")
        namespace["main"]()
    finally:
        DataCollectionEnv.step = step_original
        audit_file.close()
        for item in joints.values():
            item["max_abs_change"] = max(
                abs(item["min"] - item["initial"]), abs(item["max"] - item["initial"])
            )
        r = np.asarray(right_positions)
        summary = {
            "steps": count,
            "physics_finite": finite,
            "first_simulation_time_s": first_sim_time,
            "last_simulation_time_s": last_sim_time,
            "scene_joints": joints,
            "right_end_path_length_m": float(np.linalg.norm(np.diff(r, axis=0), axis=1).sum())
            if len(r) > 1
            else 0,
            "started_wall_ns": started,
            "finished_wall_ns": time.time_ns(),
            "official_script": name,
            "arguments": forwarded,
            "note": "Physics audit does not represent an official referee success verdict.",
        }
        (args.audit_output / "physics_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if evaluator is not None:
            (args.audit_output / "task_evaluation.json").write_text(
                json.dumps(
                    {"goal": goal, "evaluation": evaluator.finalize()}, ensure_ascii=False, indent=2
                ),
                encoding="utf-8",
            )
        print(
            f"Physics audit: {count} steps; finite={finite}; {args.audit_output.resolve()}",
            flush=True,
        )
    if count == 0:
        raise SystemExit("Official entry did not execute any physics steps; check logs.")


if __name__ == "__main__":
    import os
    import traceback

    code = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        code = 1
    finally:
        # Match the official entry's shutdown policy, after persisting our audit.
        # Its legacy camera daemon threads otherwise race interpreter shutdown.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
