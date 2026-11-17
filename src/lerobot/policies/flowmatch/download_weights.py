"""Stdlib-only download of the original private Release assets; no model/hardware imports."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify(path, asset):
    if Path(path).stat().st_size != asset["bytes"] or digest(path) != asset["sha256"]:
        raise ValueError(f"SHA256/size mismatch: {path}; existing files are not overwritten")


def install(source, destination, asset):
    source, destination = Path(source), Path(destination)
    verify(source, asset)
    if destination.exists():
        verify(destination, asset)
        return "already_verified"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as target, source.open("rb") as origin:
        shutil.copyfileobj(origin, target, 4 * 1024 * 1024)
    verify(destination, asset)
    return "installed"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/flowmatch_100k/source_weights"))
    parser.add_argument("--gh", default="gh")
    parser.add_argument("--source-dir", type=Path, help="Read-only local assets instead of network download")
    parser.add_argument("--with-training-checkpoint", action="store_true")
    args = parser.parse_args()
    manifest_path = Path(__file__).parent / "assets/weights_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    roles = {"inference", "vision", "action"} | ({"training"} if args.with_training_checkpoint else set())
    for asset in manifest["assets"]:
        if asset["role"] not in roles:
            continue
        relative = Path(asset["path"]).relative_to("weights/100k")
        destination = args.output / relative
        if destination.exists():
            verify(destination, asset)
            print(json.dumps({"file": str(destination), "status": "already_verified"}))
            continue
        if args.source_dir:
            source = args.source_dir / relative
            if not source.exists():
                source = args.source_dir / asset["name"]
            status = install(source, destination, asset)
        else:
            with tempfile.TemporaryDirectory(prefix="flowmatch100k-") as temporary:
                subprocess.run(
                    [
                        args.gh,
                        "release",
                        "download",
                        manifest["tag"],
                        "--repo",
                        manifest["repository"],
                        "--pattern",
                        asset["name"],
                        "--dir",
                        temporary,
                    ],
                    check=True,
                )
                status = install(Path(temporary) / asset["name"], destination, asset)
        print(json.dumps({"file": str(destination), "status": status, "sha256": asset["sha256"]}))


if __name__ == "__main__":
    main()
