"""Paired, read-only Pi05 validation in separate GPU processes.

No optimizer, backward pass or robot connection. Loss uses the official 32D
flow objective; sampled-action metrics use only the 18 real robot dimensions.
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from types import MethodType

import numpy as np
from g1_openpi_eval import (
    ACTION_NAMES,
    action_metrics,
    equal_mean,
    paired_comparison,
    stable_window_id,
    summarize_episodes,
    window_starts,
)

OPENPI_COMMIT = "981483dca0fd9acba698fea00aa6e52d56a66c58"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_plan(args):
    """Validate contracts without decoding videos or importing JAX."""
    commit = subprocess.check_output(
        ["git", "-C", str(args.openpi_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != OPENPI_COMMIT:
        raise ValueError("Expected the exact pinned OpenPI commit")
    config = read(args.model_config)
    expected_model = {
        "pi05": True,
        "dtype": "bfloat16",
        "paligemma_variant": "gemma_2b_lora",
        "action_expert_variant": "gemma_300m_lora",
        "action_dim": 32,
        "action_horizon": 24,
        "max_token_len": 200,
    }
    if config["model"] != expected_model or config["training"]["freeze_vision"] is not True:
        raise ValueError("Expected the tested frozen-vision dual-LoRA Pi05 model")
    if config["openpi_commit"] != commit or config["data"]["fps"] != 24:
        raise ValueError("Model/data version mismatch")
    if config["data"]["action_representation"] != "absolute_base_pose_xyzw_gripper":
        raise ValueError("Unsupported action representation")
    if digest(args.dataset) != config["data"]["validation_manifest_sha256"]:
        raise ValueError("Validation manifest differs from the trained artifact")
    if digest(args.train_manifest) != config["data"]["train_manifest_sha256"]:
        raise ValueError("Training manifest differs from the trained artifact")
    norm = args.checkpoint / "assets/g1_buttons/norm_stats.json"
    if digest(norm) != config["data"]["norm_stats_sha256"]:
        raise ValueError("Checkpoint normalization differs from training")
    if not (args.checkpoint / "_CHECKPOINT_METADATA").exists():
        raise ValueError("Expected a committed complete checkpoint")
    validation = read(args.dataset)["datasets"]
    train_paths = {
        (args.train_manifest.parent / row["path"]).resolve()
        for row in read(args.train_manifest)["datasets"]
    }
    seen, episodes, files = set(), [], {str(norm): digest(norm)}
    for source in validation:
        path = (args.dataset.parent / source["path"]).resolve()
        if path in seen or path in train_paths:
            raise ValueError("Duplicate episode or training/validation leakage")
        seen.add(path)
        info = read(path / "meta/info.json")
        if info["fps"] != 24 or info["codebase_version"] != "v2.1":
            raise ValueError("Expected 24 FPS LeRobot v2.1")
        for feature in ["observation.state", "action"]:
            if info["features"][feature]["names"] != [ACTION_NAMES]:
                raise ValueError("Recorded 18D field order differs from the metric contract")
        evidence = read(args.dataset.parent / source["evaluation"])
        if (
            evidence["evaluation"]["proxy_success"] is not True
            or evidence["goal"]["target_id"] != source["target_id"]
        ):
            raise ValueError("Validation source does not pass its developmental quality gate")
        tasks = [read_line for read_line in read_jsonl(path / "meta/tasks.jsonl")]
        if {task["task"] for task in tasks} != {source["expected_prompt"]}:
            raise ValueError("Prompt mismatch")
        for ep in read_jsonl(path / "meta/episodes.jsonl"):
            eid, length = ep["episode_index"], ep["length"]
            key = f"{source['path']}#episode={eid}"
            episodes.append(
                {
                    "key": key,
                    "path": str(path),
                    "episode_id": eid,
                    "task": source["name"],
                    "prompt": source["expected_prompt"],
                    "target_id": source["target_id"],
                    "frames": length,
                    "loss_starts": window_starts(length, 24, args.loss_stride),
                    "action_starts": window_starts(length, 24, args.action_stride),
                }
            )
            kwargs = {"episode_index": eid, "episode_chunk": eid // info["chunks_size"]}
            for name in [
                info["data_path"].format(**kwargs),
                *[
                    info["video_path"].format(video_key=camera, **kwargs)
                    for camera in ["observation.images.cam_head", "observation.images.cam_wrist_r"]
                ],
            ]:
                files[str(path / name)] = digest(path / name)
    if args.smoke:
        episodes = episodes[:1]
        for ep in episodes:
            ep["loss_starts"] = ep["loss_starts"][:2]
            ep["action_starts"] = ep["action_starts"][:2]
    plan = {
        "schema_version": 1,
        "openpi_commit": commit,
        "model": config,
        "checkpoint": str(args.checkpoint),
        "seeds": args.seeds,
        "loss_stride": args.loss_stride,
        "action_stride": args.action_stride,
        "sampling_steps": args.sampling_steps,
        "loss_seed_batch": args.loss_seed_batch,
        "smoke": args.smoke,
        "episodes": episodes,
        "validation_manifest_sha256": digest(args.dataset),
        "training_manifest_sha256": digest(args.train_manifest),
        "norm_stats_sha256": digest(norm),
        "source_data_sha256": files,
        "evaluation_source_sha256": {
            name: digest(Path(__file__).parent / name)
            for name in [
                "evaluate_g1_openpi.py",
                "g1_openpi_eval.py",
                "g1_openpi_data.py",
                "g1_openpi_memory.py",
            ]
        },
        "aggregation": "Unique windows -> per-episode mean -> per-task mean -> equal task macro",
        "scope": "Offline validation on recorded observations; no robot execution.",
    }
    plan["sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    return plan


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def restore_model(plan, variant):
    """Preserve actual mixed training dtypes; allocate no random base weights."""
    import flax.traverse_util as traverse
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from g1_openpi_memory import MemoryEfficientCheckpointWeightLoader
    from openpi.models.pi0_config import Pi0Config
    from openpi.shared import nnx_utils

    config = Pi0Config(**plan["model"]["model"])
    shape_state = nnx.state(nnx.eval_shape(config.create, jax.random.key(0)))
    freeze = nnx.Any(config.get_freeze_filter(), nnx_utils.PathRegex(".*PaliGemma.*img.*"))
    frozen = set(traverse.flatten_dict(shape_state.filter(freeze).to_pure_dict()))
    reference = traverse.unflatten_dict(
        {
            key: jax.ShapeDtypeStruct(value.shape, jnp.bfloat16 if key in frozen else value.dtype)
            for key, value in traverse.flatten_dict(shape_state.to_pure_dict()).items()
        }
    )
    location = (
        plan["model"]["base_checkpoint"]
        if variant == "baseline"
        else str(Path(plan["checkpoint"]) / "params")
    )
    loaded = MemoryEfficientCheckpointWeightLoader(location).load(reference)
    flat = traverse.flatten_dict(loaded)
    del loaded
    lora_digest = hashlib.sha256()
    if variant == "baseline":

        @jax.jit
        def initial_lora(rng):
            model = config.create(rng)
            return nnx.state(model, nnx_utils.PathRegex(".*lora.*")).to_pure_dict()

        # Match upstream scripts/train.py exactly (two consecutive splits).
        _, init_rng = jax.random.split(jax.random.key(plan["model"]["training"]["seed"]))
        _, model_rng = jax.random.split(init_rng)
        initialized = traverse.flatten_dict(initial_lora(model_rng))
        for key in sorted(initialized):
            lora_digest.update("/".join(key).encode())
            lora_digest.update(np.asarray(initialized[key]).tobytes())
        for key, value in list(flat.items()):
            if isinstance(value, jax.ShapeDtypeStruct):
                flat[key] = initialized.pop(key)
        if initialized:
            raise ValueError("Unexpected baseline LoRA parameters")
    elif any(isinstance(v, jax.ShapeDtypeStruct) for v in flat.values()):
        raise ValueError("Trained checkpoint has missing LoRA parameters")
    device = {}
    for key in list(flat):
        value = flat.pop(key)
        if isinstance(value, jax.ShapeDtypeStruct):
            raise ValueError(f"Missing checkpoint parameter: {key}")
        device[key] = jax.device_put(value)
        jax.block_until_ready(device[key])
    model = config.load(traverse.unflatten_dict(device))
    model.eval()
    logging.info("Restored %s on GPU with original mixed parameter dtypes", variant)
    return model, config, lora_digest.hexdigest() if variant == "baseline" else None


def worker(args):
    started = time.monotonic()
    sys.path[:0] = [
        str(args.openpi_root / "src"),
        str(args.openpi_root / "packages/openpi-client/src"),
    ]
    import jax
    import jax.numpy as jnp
    from g1_openpi_data import G1Dataset, G1Inputs, G1Outputs
    from openpi import transforms as tr
    from openpi.models import model as om
    from openpi.shared import nnx_utils, normalize
    from openpi.training.config import ModelTransformFactory

    if sys.platform != "linux" or not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("Real pretrained evaluation requires Linux + JAX GPU")
    plan = read(args.output / "plan.json")
    for name, expected in plan["evaluation_source_sha256"].items():
        if digest(Path(__file__).parent / name) != expected:
            raise ValueError("Evaluation code changed between paired workers")
    directory = args.output / args.variant
    directory.mkdir(exist_ok=False)
    model, config, initial_digest = restore_model(plan, args.variant)
    stats = normalize.load(args.checkpoint / "assets/g1_buttons")
    pipeline = tr.compose(
        [
            G1Inputs(),
            tr.Normalize(stats, use_quantiles=True),
            *ModelTransformFactory()(config).inputs,
        ]
    )
    unnormalize = tr.Unnormalize({"actions": stats["actions"]}, use_quantiles=True)

    def seed_losses(model, keys, observation, actions):
        # Vmap only independent seed evaluations; each official call keeps B=1.
        return jax.vmap(lambda key: model.compute_loss(key, observation, actions, train=False))(
            keys
        )

    if plan["loss_seed_batch"] == 1:
        official_loss = nnx_utils.module_jit(model.compute_loss, static_argnames=("train",))

        def loss_fn(keys, observation, actions):
            return official_loss(keys[0], observation, actions, train=False)[None]
    else:
        loss_fn = nnx_utils.module_jit(MethodType(seed_losses, model))
    sample_fn = nnx_utils.module_jit(model.sample_actions, static_argnames=("num_steps",))

    def batch(raw):
        data = pipeline(raw)
        target = jnp.asarray(data.pop("actions"))[None]
        inputs = jax.tree.map(lambda v: jnp.asarray(v)[None], data)
        return om.Observation.from_dict(inputs), target

    def rng(seed, episode, frame, namespace):
        return jax.random.fold_in(jax.random.key(seed), stable_window_id(episode, frame, namespace))

    loss_rows, action_rows, episode_results = [], [], []
    predicted, targets, starts, seeds, episode_keys = [], [], [], [], []
    cold, warm = {}, False
    ds, current_path = None, None
    for entry in plan["episodes"]:
        if entry["path"] != current_path:
            ds = G1Dataset(entry["path"], 24, image_cache=args.image_cache)
            current_path = entry["path"]
        ep_number = next(i for i, ep in enumerate(ds.episodes) if ep["id"] == entry["episode_id"])
        lookup = {t: index for index, (e, t) in enumerate(ds.indices) if e == ep_number}
        logging.info(
            "%s %s: %d loss windows, %d action windows x %d seeds",
            args.variant,
            entry["key"],
            len(entry["loss_starts"]),
            len(entry["action_starts"]),
            len(plan["seeds"]),
        )
        if not warm:
            ob, target = batch(ds[lookup[0]])
            keys = jnp.stack(
                [
                    rng(seed, entry["key"], 0, "loss")
                    for seed in plan["seeds"][: plan["loss_seed_batch"]]
                ]
            )
            tick = time.monotonic()
            jax.block_until_ready(loss_fn(keys, ob, target))
            cold["loss_compile_and_first_call_ms"] = (time.monotonic() - tick) * 1000
            key = rng(plan["seeds"][0], entry["key"], 0, "action")
            noise = jax.random.normal(key, (1, 24, 32))
            tick = time.monotonic()
            jax.block_until_ready(sample_fn(key, ob, noise=noise, num_steps=plan["sampling_steps"]))
            cold["action_compile_and_first_call_ms"] = (time.monotonic() - tick) * 1000
            warm = True
        ep_losses, ep_actions, holds = [], [], []
        for count, frame in enumerate(entry["loss_starts"]):
            ob, target = batch(ds[lookup[frame]])
            for offset in range(0, len(plan["seeds"]), plan["loss_seed_batch"]):
                group = plan["seeds"][offset : offset + plan["loss_seed_batch"]]
                keys = jnp.stack([rng(seed, entry["key"], frame, "loss") for seed in group])
                tick = time.monotonic()
                values = np.asarray(loss_fn(keys, ob, target))
                elapsed = (time.monotonic() - tick) * 1000 / len(group)
                if values.shape != (len(group), 1, 24) or not np.isfinite(values).all():
                    raise ValueError("Invalid validation loss")
                for seed, sample in zip(group, values, strict=True):
                    value = float(sample.mean())
                    ep_losses.append({"flow_loss_32": value})
                    loss_rows.append(
                        {
                            "episode": entry["key"],
                            "task": entry["task"],
                            "frame": frame,
                            "seed": seed,
                            "flow_loss_32": value,
                            "amortized_forward_ms": elapsed,
                        }
                    )
            if count % 80 == 0:
                logging.info(
                    "%s loss %s %d/%d",
                    args.variant,
                    entry["task"],
                    count + 1,
                    len(entry["loss_starts"]),
                )
        for frame in entry["action_starts"]:
            raw = ds[lookup[frame]]
            truth = raw["action"].copy()
            hold = np.broadcast_to(raw["observation.state"], truth.shape)
            holds.append(action_metrics(hold, truth))
            ob, target = batch(raw)
            for seed in plan["seeds"]:
                key = rng(seed, entry["key"], frame, "action")
                noise = jax.random.normal(key, (1, 24, 32))
                tick = time.monotonic()
                normalized = np.asarray(
                    sample_fn(key, ob, noise=noise, num_steps=plan["sampling_steps"])
                )[0]
                elapsed = (time.monotonic() - tick) * 1000
                prediction = G1Outputs()(unnormalize({"actions": normalized}))["actions"]
                metrics = action_metrics(prediction, truth)
                metrics["normalized_action_mse_18"] = float(
                    np.mean(np.square(normalized[:, :18] - np.asarray(target)[0, :, :18]))
                )
                ep_actions.append(metrics)
                action_rows.append(
                    {
                        "episode": entry["key"],
                        "task": entry["task"],
                        "frame": frame,
                        "seed": seed,
                        "sampling_ms": elapsed,
                        **metrics,
                    }
                )
                predicted.append(prediction)
                targets.append(truth)
                starts.append(frame)
                seeds.append(seed)
                episode_keys.append(entry["key"])
        result = {
            "key": entry["key"],
            "task": entry["task"],
            "target_id": entry["target_id"],
            "frames": entry["frames"],
            "loss_windows": len(entry["loss_starts"]),
            "action_windows": len(entry["action_starts"]),
            "metrics": {**equal_mean(ep_losses), **equal_mean(ep_actions)},
            "hold_current_state": equal_mean(holds),
        }
        episode_results.append(result)
        write(directory / "episodes.partial.json", episode_results)
        logging.info(
            "Completed %s %s: loss %.6f, position %.3f mm",
            args.variant,
            entry["key"],
            result["metrics"]["flow_loss_32"],
            result["metrics"]["position_mm_horizon"],
        )
    for filename, rows in [("loss_windows.csv", loss_rows), ("action_windows.csv", action_rows)]:
        with (directory / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    np.savez_compressed(
        directory / "predictions.npz",
        prediction=np.asarray(predicted),
        target=np.asarray(targets),
        frame=np.asarray(starts),
        seed=np.asarray(seeds),
        episode=np.asarray(episode_keys),
    )
    result = {
        "schema_version": 1,
        "status": "complete",
        "variant": args.variant,
        "plan_sha256": plan["sha256"],
        "episodes": episode_results,
        "summary": summarize_episodes(episode_results),
        "seeds": plan["seeds"],
        "unique_loss_windows": sum(e["loss_windows"] for e in episode_results),
        "unique_action_windows": sum(e["action_windows"] for e in episode_results),
        "loss_forward_calls": len(loss_rows),
        "action_sampling_calls": len(action_rows),
        "baseline_initial_lora_sha256": initial_digest,
        "warmup": cold,
        "mean_loss_forward_ms": float(np.mean([row["amortized_forward_ms"] for row in loss_rows])),
        "mean_action_sampling_ms": float(np.mean([row["sampling_ms"] for row in action_rows])),
        "elapsed_seconds": time.monotonic() - started,
        "devices": [str(d) for d in jax.devices()],
        "optimizer_created": False,
        "scope": plan["scope"],
        "official_success_rate": None,
        "flow_loss_definition": "Official compute_loss(train=False); mean of 32 dimensions including 14 padding dimensions, 24 timesteps and fixed seeds.",
        "action_metric_definition": "18D physical units after training-stat unnormalization; no clipping; quaternion sign invariant. First/first6/full24 reported.",
    }
    write(directory / "result.json", result)
    logging.info("Completed %s evaluation in %.1f seconds", args.variant, result["elapsed_seconds"])


def plots(output, comparison, plan):
    """Static exportable charts; optional plotting dependency, not an eval prerequisite."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {"created": False, "reason": "matplotlib unavailable; CSV/NPZ metrics are complete"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    tasks = list(comparison["tasks"])
    for ax, name, title in zip(
        axes.flat,
        ["flow_loss_32", "position_mm_horizon", "orientation_deg_horizon", "gripper_mae_horizon"],
        [
            "Validation flow loss (32D)",
            "Position error (mm)",
            "Orientation error (degrees)",
            "Gripper MAE",
        ],
        strict=True,
    ):
        x = np.arange(len(tasks))
        for delta, variant, color in [
            (-0.18, "baseline", "#657c94"),
            (0.18, "finetuned", "#1b998b"),
        ]:
            ax.bar(
                x + delta,
                [comparison["tasks"][task][name][variant] for task in tasks],
                0.36,
                label=variant,
                color=color,
            )
        ax.set_xticks(x, tasks)
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        ax.legend()
    fig.savefig(output / "task_metrics.png", dpi=180)
    plt.close(fig)
    files = ["task_metrics.png"]
    with (
        np.load(output / "baseline/predictions.npz", allow_pickle=False) as before,
        np.load(output / "finetuned/predictions.npz", allow_pickle=False) as after,
    ):
        for task in tasks:
            episode = next(ep for ep in plan["episodes"] if ep["task"] == task)
            indices = np.flatnonzero(
                (before["episode"] == episode["key"]) & (before["seed"] == plan["seeds"][0])
            )
            index = int(indices[len(indices) // 2])
            if (
                after["episode"][index] != before["episode"][index]
                or after["frame"][index] != before["frame"][index]
            ):
                raise ValueError("Prediction archive order differs")
            fig, axes = plt.subplots(2, 4, figsize=(13, 6), constrained_layout=True)
            seconds = np.arange(24) / 24
            for arm, columns in enumerate([[0, 1, 2], [7, 8, 9]]):
                for ax, dim in zip(axes[arm, :3], columns, strict=True):
                    for variant, array, style in [
                        ("demonstration", before["target"], "k-"),
                        ("baseline", before["prediction"], "--"),
                        ("finetuned", after["prediction"], "-"),
                    ]:
                        ax.plot(seconds, array[index, :, dim], style, label=variant)
                    ax.set_title(ACTION_NAMES[dim] + " (m)")
                    ax.set_xlabel("future seconds")
                ax = axes[arm, 3]
                for dim in [14 + arm * 2, 15 + arm * 2]:
                    for variant, array, style in [
                        ("truth", before["target"], "-"),
                        ("baseline", before["prediction"], ":"),
                        ("finetuned", after["prediction"], "--"),
                    ]:
                        ax.plot(
                            seconds,
                            array[index, :, dim],
                            style,
                            label=variant + " " + ACTION_NAMES[dim],
                        )
                ax.set_title("gripper controls")
                ax.set_xlabel("future seconds")
            axes[0, 0].legend(fontsize=8)
            axes[0, 3].legend(fontsize=6)
            fig.suptitle(f"{task}: first episode, middle sampled window, seed {plan['seeds'][0]}")
            filename = f"trajectory_{task}.png"
            fig.savefig(output / filename, dpi=160)
            plt.close(fig)
            files.append(filename)
    return {
        "created": True,
        "files": files,
        "selection": "First held-out episode per task, middle sampled anchor, first seed; not selected by error.",
    }


def run(args):
    plan = validate_plan(args)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "plan.json", plan)
    variants = ["baseline", "finetuned"] if args.variant == "compare" else [args.variant]
    for variant in variants:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--variant",
            variant,
            "--openpi-root",
            str(args.openpi_root),
            "--dataset",
            str(args.dataset),
            "--checkpoint",
            str(args.checkpoint),
            "--model-config",
            str(args.model_config),
            "--output",
            str(args.output),
            "--image-cache",
            str(args.image_cache),
        ]
        environment = {
            **os.environ,
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "OMP_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
        }
        with (args.output / f"{variant}.log").open("w", encoding="utf-8") as log:
            logging.info(
                "Starting %s in a separate GPU process; log: %s",
                variant,
                args.output / f"{variant}.log",
            )
            process = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            write(
                args.output / "failure.json", {"variant": variant, "exit_code": process.returncode}
            )
            raise RuntimeError(f"{variant} failed with exit {process.returncode}; inspect its log")
    if args.variant == "compare":
        before, after = [read(args.output / name / "result.json") for name in variants]
        comparison = paired_comparison(before, after)
        comparison.update(
            {
                "status": "complete",
                "plan_sha256": plan["sha256"],
                "smoke": plan["smoke"],
                "scope": plan["scope"],
            }
        )
        write(args.output / "comparison.json", comparison)
        write(args.output / "plots.json", plots(args.output, comparison, plan))
        lines = [
            "# OpenPI 独立验证结果",
            "",
            "三任务按示范、任务等权汇总；负变化表示误差降低。",
            "",
            "| 任务 | flow loss：训练前 → 训练后 | 位置误差 mm：训练前 → 训练后 | 姿态误差 °：训练前 → 训练后 | 夹爪 MAE：训练前 → 训练后 |",
            "|---|---:|---:|---:|---:|",
        ]
        for task, metrics in {**comparison["tasks"], "macro": comparison["macro"]}.items():
            values = [
                metrics[name]
                for name in [
                    "flow_loss_32",
                    "position_mm_horizon",
                    "orientation_deg_horizon",
                    "gripper_mae_horizon",
                ]
            ]
            lines.append(
                f"| {task} | "
                + " | ".join(f"{v['baseline']:.6f} → {v['finetuned']:.6f}" for v in values)
                + " |"
            )
        lines += [
            "",
            f"唯一 loss 窗口：{before['unique_loss_windows']}；唯一动作窗口：{before['unique_action_windows']}；每窗口使用 {len(plan['seeds'])} 个固定种子。",
            "",
            "loss 包含模型的 32 维；动作误差只计算真实 18 维，含双臂位姿和四个夹爪通道。",
            "",
            "这次是记录观测上的离线评估，尚未执行仿真闭环或官方成功率测试。",
        ]
        (args.output / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info("Evaluation complete: %s", args.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--openpi-root", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True, help="Held-out validation manifest")
    p.add_argument(
        "--train-manifest", type=Path, help="Leakage check; defaults to sibling manifest_train.json"
    )
    p.add_argument(
        "--checkpoint", type=Path, required=True, help="Complete step directory, e.g. 99"
    )
    p.add_argument("--model-config", type=Path, help="Defaults to experiment/model_config.json")
    p.add_argument(
        "--output", type=Path, required=True, help="New directory; never overwrites a run"
    )
    p.add_argument("--image-cache", type=Path, required=True)
    p.add_argument("--variant", choices=["compare", "baseline", "finetuned"], default="compare")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--loss-stride", type=int, default=1)
    p.add_argument("--action-stride", type=int, default=24)
    p.add_argument("--sampling-steps", type=int, default=10)
    p.add_argument(
        "--loss-seed-batch",
        type=int,
        default=1,
        help="Vectorize this many fixed noise seeds per loss window; same paired plan",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Only first episode and two windows; not full evaluation",
    )
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    args.openpi_root, args.dataset = args.openpi_root.resolve(), args.dataset.resolve()
    args.checkpoint, args.output = args.checkpoint.resolve(), args.output.resolve()
    args.image_cache = args.image_cache.resolve()
    args.train_manifest = (
        args.train_manifest or args.dataset.with_name("manifest_train.json")
    ).resolve()
    args.model_config = (
        args.model_config or args.checkpoint.parent / "model_config.json"
    ).resolve()
    if min(args.loss_stride, args.action_stride, args.sampling_steps, args.loss_seed_batch) < 1:
        p.error("strides and sampling-steps must be positive")
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):
        p.error("seeds must be unique uint32 integers")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    worker(args) if args.worker else run(args)


if __name__ == "__main__":
    main()
