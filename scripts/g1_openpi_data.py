"""G1 LeRobot v2.1 -> OpenPI. Absolute base-frame poses, xyzw quaternions.

No extra temporal shift: action[t] is already the next sampled state.
Only full action windows within each episode are used for this first baseline.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import uuid
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from PIL import __version__ as pillow_version

CAMERAS = {
    "base_0_rgb": "observation.images.cam_head",
    "right_wrist_0_rgb": "observation.images.cam_wrist_r",
}


def resize_pad(rgb, size=224):
    im = Image.fromarray(rgb)
    scale = size / max(im.size)
    im = im.resize((round(im.width * scale), round(im.height * scale)), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size))
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return np.asarray(canvas)


def read_video_images(path, n, image_cache=None):
    """Decode identical RGB frames; optional read-only, content-addressed mmap."""
    path = Path(path)
    if image_cache is None:
        with av.open(str(path)) as container:
            frames = [resize_pad(f.to_ndarray(format="rgb24")) for f in container.decode(video=0)]
        if len(frames) != n:
            raise ValueError("Video/table mismatch")
        return np.stack(frames)
    with path.open("rb") as stream:
        video_digest = hashlib.file_digest(stream, "sha256").digest()
    version = f"g1_rgb224_v1:{n}:{av.__version__}:{pillow_version}".encode()
    name = hashlib.sha256(video_digest + version).hexdigest() + ".npy"
    directory = Path(image_cache).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    if not target.exists():
        temporary = directory / (name + "." + uuid.uuid4().hex + ".partial")
        frames = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=np.uint8, shape=(n, 224, 224, 3)
        )
        count = 0
        try:
            with av.open(str(path)) as container:
                for frame in container.decode(video=0):
                    if count >= n:
                        raise ValueError("Video/table mismatch")
                    frames[count] = resize_pad(frame.to_ndarray(format="rgb24"))
                    count += 1
            if count != n:
                raise ValueError("Video/table mismatch")
            frames.flush()
        finally:
            del frames
        os.replace(temporary, target)
    frames = np.load(target, mmap_mode="r", allow_pickle=False)
    if frames.shape != (n, 224, 224, 3) or frames.dtype != np.uint8:
        raise ValueError("Invalid image cache shape or dtype")
    return frames


@dataclasses.dataclass(frozen=True)
class G1Inputs:
    def __call__(self, data):
        images = {}
        for name, key in CAMERAS.items():
            rgb = np.asarray(data[key])
            if rgb.shape[0] == 3:
                rgb = np.moveaxis(rgb, 0, -1)
            if np.issubdtype(rgb.dtype, np.floating):
                if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
                    raise ValueError("Float images must be in [0, 1]")
                rgb = np.round(rgb * 255).astype(np.uint8)
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise ValueError("Expected RGB uint8 HWC or float CHW")
            images[name] = resize_pad(rgb)
        images["left_wrist_0_rgb"] = np.zeros_like(images["base_0_rgb"])
        state = np.asarray(data["observation.state"], dtype=np.float32)
        if state.shape != (18,) or not np.isfinite(state).all():
            raise ValueError("Expected finite 18D state")
        result = {
            "image": images,
            "image_mask": {
                "base_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
            },
            "state": state,
            "prompt": str(data["prompt"]),
        }
        if "action" in data:
            actions = np.asarray(data["action"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[-1] != 18 or not np.isfinite(actions).all():
                raise ValueError("Expected finite H x 18 actions")
            result["actions"] = actions
        return result


@dataclasses.dataclass(frozen=True)
class G1Outputs:
    def __call__(self, data):
        # Raw absolute predictions for offline evaluation, NOT a motor safety controller.
        actions = np.asarray(data["actions"])[..., :18]
        if not np.isfinite(actions).all():
            raise ValueError("Non-finite model action")
        return {"actions": actions}


class G1Dataset:
    """Local, offline decoder; deliberately avoids hub access and codec ambiguity."""

    def __init__(self, root, horizon=24, *, image_cache=None):
        self.root, self.horizon = Path(root), horizon
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        self.info = json.loads((self.root / "meta/info.json").read_text(encoding="utf-8"))
        if self.info["fps"] != 24 or self.info["codebase_version"] != "v2.1":
            raise ValueError("This baseline expects 24 FPS LeRobot v2.1")
        self.tasks = {x["task_index"]: x["task"] for x in self._jsonl("tasks.jsonl")}
        self.episodes, self.indices = [], []
        for meta in self._jsonl("episodes.jsonl"):
            eid = meta["episode_index"]
            kwargs = {"episode_index": eid, "episode_chunk": eid // self.info["chunks_size"]}
            table = pq.read_table(self.root / self.info["data_path"].format(**kwargs)).to_pydict()
            states = np.asarray(table["observation.state"], np.float32)
            actions = np.asarray(table["action"], np.float32)
            n = len(states)
            if states.shape != (n, 18) or actions.shape != states.shape:
                raise ValueError("Invalid state/action shape")
            if not np.isfinite(states).all() or not np.isfinite(actions).all():
                raise ValueError("Non-finite source")
            if n != meta["length"] or not np.array_equal(table["frame_index"], np.arange(n)):
                raise ValueError("Episode length/frame mismatch")
            if not np.allclose(table["timestamp"], np.arange(n) / 24, atol=1e-5):
                raise ValueError("Timestamp mismatch")
            if not np.allclose(actions[:-1], states[1:], atol=1e-6):
                raise ValueError("Expected already-shifted next-state targets")
            images = {}
            for key in CAMERAS.values():
                path = self.root / self.info["video_path"].format(video_key=key, **kwargs)
                images[key] = read_video_images(path, n, image_cache)
            self.episodes.append(
                {
                    "id": eid,
                    "states": states,
                    "actions": actions,
                    "tasks": table["task_index"],
                    "images": images,
                }
            )
            self.indices.extend((len(self.episodes) - 1, t) for t in range(n - horizon + 1))
        if not self.indices:
            raise ValueError("No full action windows")
        if sum(len(e["states"]) for e in self.episodes) != self.info["total_frames"]:
            raise ValueError("Metadata frame count mismatch")

    def _jsonl(self, name):
        return [
            json.loads(s)
            for s in (self.root / "meta" / name).read_text(encoding="utf-8").splitlines()
            if s.strip()
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        ep, t = self.indices[index]
        e = self.episodes[ep]
        return {
            "observation.state": e["states"][t].copy(),
            "action": e["actions"][t : t + self.horizon].copy(),
            "prompt": self.tasks[e["tasks"][t]],
            **{k: v[t] for k, v in e["images"].items()},
        }

    def norm_stats(self):
        from openpi.shared.normalize import NormStats

        result = {}
        for out_key, ep_key in [("state", "states"), ("actions", "actions")]:
            values = np.concatenate([e[ep_key] for e in self.episodes])
            low, high = np.quantile(values, [0.01, 0.99], axis=0)
            # Constant/near-constant dimensions need a finite scale even with one demo.
            mid = (low + high) / 2
            half = np.maximum((high - low) / 2, 1e-3)
            result[out_key] = NormStats(
                mean=values.mean(0),
                std=np.maximum(values.std(0), 1e-3),
                q01=mid - half,
                q99=mid + half,
            )
        return result


class G1MultiTaskDataset(G1Dataset):
    """Manifest of independent validated episodes; balanced task windows.

    No copying or relabeling raw LeRobot files. Source task strings stay intact.
    Round-robin oversampling prevents a long trajectory dominating a short one.
    """

    def __init__(self, manifest, horizon=24, *, image_cache=None):
        manifest = Path(manifest).resolve()
        spec = json.loads(manifest.read_text(encoding="utf-8"))
        if spec.get("schema_version") != 1 or not spec.get("datasets"):
            raise ValueError("Invalid multitask manifest")
        self.episodes, self.indices, self.tasks = [], [], {}
        self.horizon = horizon
        self.sources = []
        task_windows = {}
        seen_paths = set()
        for source in spec["datasets"]:
            path = (manifest.parent / source["path"]).resolve()
            evidence_path = (manifest.parent / source["evaluation"]).resolve()
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            if (
                evidence["evaluation"]["proxy_success"] is not True
                or evidence["goal"]["target_id"] != source["target_id"]
            ):
                raise ValueError(
                    "Source does not pass its task-specific developmental quality gate"
                )
            if path in seen_paths:
                raise ValueError("Duplicate source dataset")
            seen_paths.add(path)
            ds = (
                G1Dataset(path, horizon)
                if image_cache is None
                else G1Dataset(path, horizon, image_cache=image_cache)
            )
            if set(ds.tasks.values()) != {source["expected_prompt"]}:
                raise ValueError("Manifest task does not match actual recorded task")
            prompt = source["expected_prompt"]
            task_id = next((i for i, p in self.tasks.items() if p == prompt), len(self.tasks))
            self.tasks[task_id] = prompt
            start = len(self.episodes)
            for ep in ds.episodes:
                self.episodes.append(
                    {
                        **ep,
                        "id": len(self.episodes),
                        "source": str(path),
                        "tasks": [task_id] * len(ep["states"]),
                    }
                )
            windows = [(start + e, t) for e, t in ds.indices]
            task_windows.setdefault(task_id, []).extend(windows)
            self.sources.append(
                {
                    "name": source["name"],
                    "path": str(path),
                    "evaluation": str(evidence_path),
                    "official_success": evidence["evaluation"]["official_success"],
                    "prompt": source["expected_prompt"],
                    "windows": len(windows),
                    "frames": ds.info["total_frames"],
                }
            )
        per_task = list(task_windows.values())
        for i in range(max(map(len, per_task))):
            self.indices.extend(w[i % len(w)] for w in per_task)
        self.info = {
            "fps": 24,
            "total_frames": sum(s["frames"] for s in self.sources),
            "total_episodes": len(self.episodes),
            "total_tasks": len(self.tasks),
        }


def load_g1_dataset(path, horizon=24, *, image_cache=None):
    cls = G1MultiTaskDataset if Path(path).is_file() else G1Dataset
    return cls(path, horizon, image_cache=image_cache)
