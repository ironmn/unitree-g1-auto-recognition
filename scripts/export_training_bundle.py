"""Export a local data-only transfer ZIP, or verify all its SHA-256 entries.

Does not upload data. Extract a verified archive into the project root.
"""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [Path("data/batch_20261005"), Path("data/multitask_training_20261006")]


def sha_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(path):
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("bundle_manifest.json"))
        names = archive.namelist()
        expected = set(manifest["files"]) | {"bundle_manifest.json"}
        if len(names) != len(set(names)) or set(names) != expected:
            raise ValueError("Duplicate, missing or extra archive members")
        for name, expected_hash in manifest["files"].items():
            member = PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts or ":" in name or "\\" in name:
                raise ValueError("Unsafe archive member")
            with archive.open(name) as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected_hash:
                raise ValueError("Hash mismatch: " + name)
    return {"verified": True, "files": len(manifest["files"]), "sha256": sha_file(path)}


def export(output):
    summary = json.loads((ROOT / SOURCES[0] / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "complete":
        raise ValueError("Batch must be complete")
    if (ROOT / SOURCES[0] / ".running.lock").exists():
        raise ValueError("Collection must be stopped before export")
    paths = []
    for source in SOURCES:
        directory = ROOT / source
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        paths.extend(p for p in directory.rglob("*") if p.is_file())
    output = output.resolve()
    if any(output.is_relative_to(ROOT / source) for source in SOURCES):
        raise ValueError("Output must be outside source directories")
    output.parent.mkdir(parents=True, exist_ok=True)
    hashes = {}
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in sorted(paths):
            if path.is_symlink():
                raise ValueError("Symlink source is not supported")
            name = path.relative_to(ROOT).as_posix()
            hashes[name] = sha_file(path)
            archive.write(
                path,
                name,
                compress_type=zipfile.ZIP_STORED if path.suffix == ".mp4" else zipfile.ZIP_DEFLATED,
            )
        archive.writestr(
            "bundle_manifest.json", json.dumps({"schema_version": 1, "files": hashes}, indent=2)
        )
    result = verify(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        result["sha256"] + "  " + output.name + "\n", encoding="utf-8"
    )
    return {**result, "archive": str(output), "bytes": output.stat().st_size}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.archive) if args.verify else export(args.archive), indent=2))
