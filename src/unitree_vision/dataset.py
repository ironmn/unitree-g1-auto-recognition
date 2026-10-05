"""Transactional samples and offline validation/recovery.

Committed sample directories are authoritative; index.jsonl is a rebuildable view.
Atomic rename prevents partial samples becoming visible. This is not a claim of
power-loss durability on every filesystem; keep acquisition on a local disk.
"""

import importlib.metadata as metadata
import json
import os
import platform
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import __version__


def atomic_json(path, value):
    path = Path(path)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def runtime_provenance():
    versions = {}
    for name in ("orca-gym", "av", "opencv-python", "numpy", "websockets"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "package_version": __version__,
        "python": platform.python_version(),
        "platform": platform.system(),
        "dependencies": versions,
    }


class DatasetWriter:
    """One writer per run, on the controller thread. No implicit resume."""

    def __init__(self, output, manifest):
        self.folder = Path(output) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        (self.folder / "samples").mkdir(parents=True, exist_ok=False)
        self.manifest = manifest
        self.count = 0
        self.last_index = -1
        atomic_json(self.folder / "manifest.json", manifest)
        (self.folder / "index.jsonl").touch(exist_ok=False)

    def save(self, entry):
        index = entry["state"]["simulate_index"]
        if type(index) is not int or not 0 <= index <= 2**31 - 1 or index <= self.last_index:
            raise ValueError("simulate_index must be increasing and fit signed int32")
        expected = set(self.manifest.get("cameras", entry["images"]))
        if not expected or set(entry["images"]) != expected:
            raise ValueError("Sample must contain all configured cameras")
        if not isinstance(entry["prompt"], str) or not entry["prompt"].strip():
            raise ValueError("Sample prompt must be a nonempty string")
        schema = self.manifest.get("state_manifest")
        if schema:
            state = entry["state"]
            dimension = len(schema["state_names"])
            if state["schema_id"] != schema["schema_id"]:
                raise ValueError("State schema mismatch")
            for key in ("state", "joint_velocity"):
                values = np.asarray(state[key], dtype=float)
                if values.shape != (dimension,) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid {key} shape or non-finite values")
        for role, (frame, meta) in entry["images"].items():
            if role not in ("head", "wrist_r", "wrist_l"):
                raise ValueError("Invalid camera role")
            if meta["simulate_index"] != index:
                raise ValueError("Image/state frame mismatch")
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError("Images must be HWC uint8 BGR")
            camera = self.manifest.get("cameras", {}).get(role, {})
            if "height" in camera and "width" in camera:
                if frame.shape[:2] != (camera["height"], camera["width"]):
                    raise ValueError("Image resolution changed; start a new collection run")
        json.dumps(entry["state"], allow_nan=False)
        sample_id = f"{self.count:06d}_{index:010d}"
        temporary = self.folder / "samples" / (".partial_" + sample_id)
        final = self.folder / "samples" / sample_id
        temporary.mkdir(exist_ok=False)
        try:
            images = {}
            for role, (frame, meta) in entry["images"].items():
                ok, encoded = cv2.imencode(".png", frame)
                if not ok:
                    raise RuntimeError(f"PNG encoding failed: {role}")
                with (temporary / f"{role}.png").open("wb") as image_file:
                    image_file.write(encoded.tobytes())
                    image_file.flush()
                    os.fsync(image_file.fileno())
                images[role] = dict(
                    meta,
                    file=f"{role}.png",
                    shape_hwc=list(frame.shape),
                    color_order_on_disk="RGB PNG",
                )
            record = {
                "sample_id": sample_id,
                "prompt": entry["prompt"],
                "observation": entry["state"],
                "images": images,
                "action": None,
                "annotation": None,
            }
            atomic_json(temporary / "observation.json", record)
            temporary.rename(final)
        except BaseException:
            # Only remove this writer's uncommitted temporary directory.
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        self.count += 1
        self.last_index = index
        # If append fails, the committed sample is retained. Stop and rebuild index.
        with (self.folder / "index.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(index_record(record)) + "\n")
            file.flush()
            os.fsync(file.fileno())
        return final


def index_record(record):
    sample_id = record["sample_id"]
    return {
        "sample_id": sample_id,
        "simulate_index": record["observation"]["simulate_index"],
        "file": f"samples/{sample_id}/observation.json",
    }


def read_json(path):
    def reject(value):
        raise ValueError(f"Non-finite JSON value: {value}")

    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def inspect_dataset(folder, *, check_index=True):
    """Read-only scan; checks original PNGs, schema, ordering and index completeness."""
    folder = Path(folder)
    manifest = read_json(folder / "manifest.json")
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported dataset schema_version")
    state_schema = manifest["state_manifest"]
    dimension = len(state_schema["state_names"])
    expected = set(manifest["cameras"])
    if not dimension or not expected:
        raise ValueError("Manifest must declare state and cameras")
    rows = []
    previous = -1
    committed = sorted(p for p in (folder / "samples").iterdir() if not p.name.startswith("."))
    for sample in committed:
        record = read_json(sample / "observation.json")
        state = record["observation"]
        index = state["simulate_index"]
        if record["sample_id"] != sample.name:
            raise ValueError(f"{sample.name}: sample id mismatch")
        if type(index) is not int or not previous < index <= 2**31 - 1:
            raise ValueError(f"{sample.name}: unordered or invalid simulate_index")
        previous = index
        if state["schema_id"] != state_schema["schema_id"]:
            raise ValueError(f"{sample.name}: state schema mismatch")
        if len(state["state"]) != dimension or len(state["joint_velocity"]) != dimension:
            raise ValueError(f"{sample.name}: state dimension mismatch")
        for key in ("state", "joint_velocity"):
            values = np.asarray(state[key], dtype=float)
            if values.shape != (dimension,) or not np.isfinite(values).all():
                raise ValueError(f"{sample.name}: invalid {key}")
        if set(record["images"]) != expected:
            raise ValueError(f"{sample.name}: missing/extra cameras")
        for role, image in record["images"].items():
            if role not in ("head", "wrist_r", "wrist_l") or image["file"] != f"{role}.png":
                raise ValueError(f"{sample.name}: invalid image filename")
            if image["simulate_index"] != index:
                raise ValueError(f"{sample.name}: image/state index mismatch")
            decoded = cv2.imdecode(
                np.fromfile(sample / image["file"], dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if decoded is None or list(decoded.shape) != image["shape_hwc"]:
                raise ValueError(f"{sample.name}: corrupt or wrong-size PNG")
            camera = manifest["cameras"][role]
            if "height" in camera and "width" in camera:
                if decoded.shape[:2] != (camera["height"], camera["width"]):
                    raise ValueError(f"{sample.name}: image resolution differs from manifest")
        rows.append(index_record(record))
    if check_index:
        index_path = folder / "index.jsonl"
        actual = [
            json.loads(line)
            for line in index_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if actual != rows:
            raise ValueError("index.jsonl differs from committed samples; rebuild index offline")
    return {
        "samples": len(rows),
        "images": len(rows) * len(expected),
        "state_dimension": dimension,
        "partial_samples": len(list((folder / "samples").glob(".partial_*"))),
        "rows": rows,
    }


def rebuild_index(folder):
    """Call only after acquisition stops. Never removes samples or partial folders."""
    folder = Path(folder)
    result = inspect_dataset(folder, check_index=False)
    path = folder / "index.jsonl"
    if path.exists():
        backup = path.with_name("index.jsonl.backup_" + uuid.uuid4().hex)
        shutil.copy2(path, backup)
    temporary = folder / (".index_" + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for row in result["rows"]:
                stream.write(json.dumps(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument(
        "--rebuild-index", action="store_true", help="Offline only; backs up old index"
    )
    args = parser.parse_args(argv)
    try:
        result = rebuild_index(args.folder) if args.rebuild_index else inspect_dataset(args.folder)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Dataset validation failed: {exc}\n")
    result.pop("rows")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
