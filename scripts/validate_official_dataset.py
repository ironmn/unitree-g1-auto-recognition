"""Validate a single official LeRobot G1 episode and decode both complete videos."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw


def validate(root, output, fps=24):
    output.mkdir(parents=True, exist_ok=True)
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    tasks = [
        json.loads(s) for s in (root / "meta/tasks.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    episodes = [
        json.loads(s)
        for s in (root / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    tables = list((root / "data").rglob("*.parquet"))
    checks = {}

    def check(name, passed):
        checks[name] = bool(passed)

    check("metadata_fps", info["fps"] == fps)
    check(
        "single_complete_episode",
        info["total_episodes"] == 1 and len(episodes) == 1 and len(tables) == 1,
    )
    if len(tables) != 1:
        raise ValueError("Expected exactly one episode parquet")
    table = pq.read_table(tables[0])
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    timestamp = np.asarray(table["timestamp"].to_pylist())
    frames = np.asarray(table["frame_index"].to_pylist())
    n = len(states)
    check("nonempty_18d_states_actions", n > 2 and states.shape == actions.shape == (n, 18))
    check("metadata_frame_count", info["total_frames"] == n and episodes[0]["length"] == n)
    check("finite_states_actions", np.isfinite(states).all() and np.isfinite(actions).all())
    check("frame_indices", np.array_equal(frames, np.arange(n)))
    check("timestamps_24fps", np.allclose(timestamp, np.arange(n) / fps, atol=1e-5))
    check("action_equals_next_absolute_state", np.allclose(actions[:-1], states[1:], atol=1e-6))
    check(
        "unit_quaternions",
        all(
            np.allclose(np.linalg.norm(states[:, a:b], axis=1), 1, atol=1e-3)
            for a, b in [(3, 7), (10, 14)]
        ),
    )
    check("gripper_normalized", np.all((states[:, 14:18] >= 0) & (states[:, 14:18] <= 1)))
    check("right_arm_moves", np.linalg.norm(np.ptp(states[:, 7:10], axis=0)) > 0.05)
    check("right_gripper_changes", np.max(np.ptp(states[:, 16:18], axis=0)) > 0.2)
    video_results = {}
    video_keys = [k for k, f in info["features"].items() if f["dtype"] == "video"]
    check(
        "two_camera_features",
        set(video_keys) == {"observation.images.cam_head", "observation.images.cam_wrist_r"},
    )
    for key in video_keys:
        paths = list((root / "videos").rglob(f"{key}/episode_000000.mp4"))
        if len(paths) != 1:
            raise ValueError(f"Expected one video for {key}: {paths}")
        chosen = set(np.linspace(0, n - 1, 6, dtype=int).tolist())
        selected = []
        hashes = set()
        count = 0
        shapes = set()
        with av.open(str(paths[0])) as container:
            stream = container.streams.video[0]
            rate = float(stream.average_rate)
            codec = stream.codec_context.name
            for frame in container.decode(video=0):
                rgb = frame.to_ndarray(format="rgb24")
                shapes.add(tuple(rgb.shape))
                hashes.add(hashlib.sha256(rgb.tobytes()).digest())
                if count in chosen:
                    selected.append((count, Image.fromarray(rgb)))
                count += 1
        check(key + "_decoded_frame_count", count == n)
        check(key + "_24fps", rate == fps)
        check(key + "_1280x960", shapes == {(960, 1280, 3)})
        video_results[key] = {
            "decoded_frames": count,
            "fps": rate,
            "codec": codec,
            "unique_decoded_images": len(hashes),
            "shape_hwc": [list(s) for s in shapes],
            "file": str(paths[0].resolve()),
        }
        sheet = Image.new("RGB", (1280, 700), (20, 20, 20))
        draw = ImageDraw.Draw(sheet)
        for slot, (index, image) in enumerate(selected):
            x, y = slot % 3 * 426, slot // 3 * 350
            image.thumbnail((426, 320))
            sheet.paste(image, (x, y))
            draw.text(
                (x + 8, y + 326),
                f"{key.split('.')[-1]} frame {index} / {index / fps:.2f}s",
                fill="white",
            )
        sheet.save(output / (key.split(".")[-1] + "_contact_sheet.png"))
    report = {
        "dataset": str(root.resolve()),
        "fps": fps,
        "frames": n,
        "duration_s": n / fps,
        "tasks": tasks,
        "state_names": info["features"]["observation.state"]["names"],
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "videos": video_results,
        "right_end_position_span_m": np.ptp(states[:, 7:10], axis=0).tolist(),
        "right_gripper_span": np.ptp(states[:, 16:18], axis=0).tolist(),
        "limitations": [
            "Checks dataset consistency and motion, not official referee success.",
            "Official latest-camera sampling does not guarantee exact physics-frame alignment.",
        ],
    }
    (output / "validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "all_checks_passed": report["all_checks_passed"],
                "frames": n,
                "fps": fps,
                "failed": [k for k, v in checks.items() if not v],
            },
            ensure_ascii=False,
        )
    )
    return report["all_checks_passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if validate(args.dataset, args.output) else 1)
