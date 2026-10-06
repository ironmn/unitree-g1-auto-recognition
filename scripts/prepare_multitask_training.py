"""Prepare training-only OpenPI statistics; never starts a model or simulator."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sources(manifest):
    spec = read(manifest)
    if spec.get("schema_version") != 1 or not spec.get("datasets"):
        raise ValueError("Invalid manifest")
    result = []
    for source in spec["datasets"]:
        path = (manifest.parent / source["path"]).resolve()
        evidence = read(manifest.parent / source["evaluation"])
        if (
            evidence["evaluation"]["proxy_success"] is not True
            or evidence["goal"]["target_id"] != source["target_id"]
        ):
            raise ValueError("Failed task evidence")
        tasks = [
            json.loads(row)
            for row in (path / "meta/tasks.jsonl").read_text(encoding="utf-8").splitlines()
            if row.strip()
        ]
        if {t["task"] for t in tasks} != {source["expected_prompt"]}:
            raise ValueError("Task label mismatch")
        result.append((path, source))
    if len({p for p, s in result}) != len(result):
        raise ValueError("Duplicate episode source")
    return result


def training_arrays(train_manifest, validation_manifest):
    train = sources(train_manifest)
    validation = sources(validation_manifest)
    if {p for p, s in train} & {p for p, s in validation}:
        raise ValueError("Training/validation episode leakage")
    arrays = {"state": [], "actions": []}
    details = []
    for root, source in train:
        info = read(root / "meta/info.json")
        paths = list((root / "data").rglob("*.parquet"))
        if info["fps"] != 24 or info["total_episodes"] != 1 or len(paths) != 1:
            raise ValueError("Expected one 24 FPS episode per source")
        table = pq.read_table(paths[0]).to_pydict()
        state = np.asarray(table["observation.state"], np.float32)
        action = np.asarray(table["action"], np.float32)
        n = len(state)
        if state.shape != (n, 18) or action.shape != state.shape or n < 24:
            raise ValueError("Invalid state/action shape")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError("Non-finite data")
        if n != info["total_frames"] or not np.array_equal(table["frame_index"], np.arange(n)):
            raise ValueError("Frame count mismatch")
        if not np.allclose(table["timestamp"], np.arange(n) / 24, atol=1e-5):
            raise ValueError("Timestamp mismatch")
        if not np.allclose(action[:-1], state[1:], atol=1e-6):
            raise ValueError("Unexpected action representation")
        arrays["state"].append(state)
        arrays["actions"].append(action)
        details.append(
            {
                "path": str(root),
                "task": source["expected_prompt"],
                "frames": n,
                "parquet_sha256": hashlib.sha256(paths[0].read_bytes()).hexdigest(),
            }
        )
    return {k: np.concatenate(v) for k, v in arrays.items()}, details, len(validation)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument("--validation-manifest", type=Path, required=True)
    p.add_argument("--openpi-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output directory")
    sys.path.insert(0, str(args.openpi_root / "src"))
    from openpi.shared import normalize

    arrays, details, val_count = training_arrays(
        args.train_manifest.resolve(), args.validation_manifest.resolve()
    )
    stats = {}
    for key, values in arrays.items():
        low, high = np.quantile(values, [0.01, 0.99], axis=0)
        mid = (low + high) / 2
        half = np.maximum((high - low) / 2, 1e-3)
        stats[key] = normalize.NormStats(
            mean=values.mean(0), std=np.maximum(values.std(0), 1e-3), q01=mid - half, q99=mid + half
        )
    args.output.mkdir(parents=True)
    asset = args.output / "assets/g1_buttons"
    normalize.save(asset, stats)
    restored = normalize.load(asset)
    for key in stats:
        for field in ["mean", "std", "q01", "q99"]:
            if not np.allclose(getattr(stats[key], field), getattr(restored[key], field)):
                raise ValueError("Statistics roundtrip failed")
    report = {
        "status": "prepared_not_trained",
        "train_episodes": len(details),
        "validation_episodes": val_count,
        "training_frames": len(arrays["state"]),
        "validation_used_for_statistics": False,
        "train_manifest": str(args.train_manifest.resolve()),
        "validation_manifest": str(args.validation_manifest.resolve()),
        "manifest_sha256": {
            k: hashlib.sha256(v.read_bytes()).hexdigest()
            for k, v in [("train", args.train_manifest), ("validation", args.validation_manifest)]
        },
        "statistics_roundtrip_passed": True,
        "sources": details,
        "statistics_method": "Exact training-frame quantiles; q01/q99 half range and std floor 1e-3; same as existing G1Dataset.norm_stats",
    }
    (args.output / "preparation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps({k: v for k, v in report.items() if k != "sources"}, ensure_ascii=True, indent=2)
    )


if __name__ == "__main__":
    main()
