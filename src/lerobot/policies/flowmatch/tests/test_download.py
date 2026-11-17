import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

module_path = Path(__file__).resolve().parents[1] / "download_weights.py"
spec = importlib.util.spec_from_file_location("flowmatch_download", module_path)
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


def test_corrupt_source_or_existing_destination_never_replaces_original(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    asset = {"bytes": 4, "sha256": hashlib.sha256(b"good").hexdigest()}
    source.write_bytes(b"bad")
    with pytest.raises(ValueError):
        download.install(source, destination, asset)
    assert not destination.exists()
    source.write_bytes(b"good")
    destination.write_bytes(b"original")
    with pytest.raises(ValueError):
        download.install(source, destination, asset)
    assert destination.read_bytes() == b"original"


def test_verified_download_copy_and_repeat_are_idempotent(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "model.pt"
    source.write_bytes(b"good")
    asset = {"bytes": 4, "sha256": hashlib.sha256(b"good").hexdigest()}
    assert download.install(source, destination, asset) == "installed"
    assert download.install(source, destination, asset) == "already_verified"


def test_folder_local_downloader_main_loads_its_own_manifest(tmp_path, monkeypatch):
    assets = tmp_path / "entry/assets"
    assets.mkdir(parents=True)
    sources = tmp_path / "already_downloaded"
    sources.mkdir()
    (sources / "policy_100k.pt").write_bytes(b"good")
    manifest = {
        "assets": [
            {
                "name": "policy_100k.pt",
                "role": "inference",
                "path": "weights/100k/policy_100k.pt",
                "bytes": 4,
                "sha256": hashlib.sha256(b"good").hexdigest(),
            }
        ]
    }
    (assets / "weights_manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(download, "__file__", str(assets.parent / "download_weights.py"))
    monkeypatch.setattr(
        download.subprocess, "run", lambda *a, **k: pytest.fail("Unexpected network download")
    )
    import sys

    monkeypatch.setattr(
        sys, "argv", ["download_weights", "--source-dir", str(sources), "--output", str(tmp_path / "weights")]
    )
    download.main()
    assert (tmp_path / "weights/policy_100k.pt").read_bytes() == b"good"
