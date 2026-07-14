import hashlib
import json
from pathlib import Path


def content_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def write_manifest(path: Path, config: dict, artifacts: dict) -> Path:
    manifest = {"schema_version": 1, "config_hash": content_hash(config), "artifact_hashes": {name: content_hash(value) for name, value in artifacts.items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path
