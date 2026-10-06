"""Verify batch split/evidence integrity and sample real OpenPI input transforms."""

import argparse
import collections
import json
from pathlib import Path

import numpy as np
from batch_collect import digest, read, write
from g1_openpi_data import G1Dataset, G1Inputs


def verify(root):
    root = Path(root).resolve()
    plan = read(root / "plan.json")
    summary = read(root / "summary.json")
    assert summary["status"] == "complete", "Batch is not complete"
    split_paths = {}
    counts = {}
    samples = []
    frames = 0
    variants = set()
    for split in ["train", "validation"]:
        sources = read(root / f"manifest_{split}.json")["datasets"]
        paths = set()
        counts[split] = collections.Counter()
        sampled = set()
        for source in sources:
            dataset = (root / source["path"]).resolve()
            run = dataset.parent
            assert dataset not in paths, "Duplicate dataset"
            paths.add(dataset)
            quality = read(run / "quality.json")
            record = read(run / "result.json")
            assert quality["accepted"] is True and record["accepted"] is True
            report = read(run / "validation/validation.json")
            assert report["all_checks_passed"] is True
            assert {t["task"] for t in report["tasks"]} == {source["expected_prompt"]}
            evidence = read(root / source["evaluation"])
            assert evidence["evaluation"]["proxy_success"] is True
            assert evidence["goal"]["target_id"] == source["target_id"]
            assert evidence["audit_sha256"] == digest(run / "audit/physics_trace.jsonl")
            variation = read(run / "variation.json")
            assert variation["seed"] not in variants, "Duplicate variation seed"
            variants.add(variation["seed"])
            info = read(dataset / "meta/info.json")
            assert info["fps"] == 24 and info["total_frames"] == report["frames"]
            frames += info["total_frames"]
            counts[split][source["name"]] += 1
            if source["name"] not in sampled:
                ds = G1Dataset(dataset)
                example = G1Inputs()(ds[len(ds) // 2])
                assert example["state"].shape == (18,)
                assert example["actions"].shape == (24, 18)
                assert all(im.shape == (224, 224, 3) for im in example["image"].values())
                assert np.isfinite(example["actions"]).all()
                samples.append(
                    {
                        "split": split,
                        "task": source["name"],
                        "dataset": source["path"],
                        "full_action_windows": len(ds),
                        "prompt": example["prompt"],
                        "action_shape": list(example["actions"].shape),
                    }
                )
                sampled.add(source["name"])
                del ds, example
        split_paths[split] = paths
    assert not split_paths["train"] & split_paths["validation"], "Episode leakage"
    all_paths = {(root / s["path"]).resolve() for s in read(root / "manifest_all.json")["datasets"]}
    assert all_paths == split_paths["train"] | split_paths["validation"]
    for name in plan["tasks"]:
        assert counts["train"][name] == plan["per_task"] - plan["validation_per_task"]
        assert counts["validation"][name] == plan["validation_per_task"]
    assert frames == summary["frames"]
    for name, sha in plan["input_hashes"].items():
        assert digest(root / "inputs" / name) == sha
    result = {
        "passed": True,
        "episodes": len(all_paths),
        "frames": frames,
        "split_counts": counts,
        "adapter_samples": samples,
        "scope": "Every accepted report and split checked; one real episode per task per split decoded through OpenPI input adapter. No model training.",
        "official_success_verified": False,
    }
    write(root / "batch_verification.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    print(json.dumps(verify(p.parse_args().root), ensure_ascii=False, indent=2))
