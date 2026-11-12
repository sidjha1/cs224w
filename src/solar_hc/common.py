from __future__ import annotations

import hashlib
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    """Atomic replacement: interrupted runs never leave partial JSON checkpoints."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(path):
    path = Path(path).resolve()
    config = read_json(path)
    # Relative data paths resolve from the project root, not the invoking directory.
    config["data_dir"] = str((path.parent.parent / config["data_dir"]).resolve())
    if not 0 < config["voltage_min_pu"] < config["voltage_max_pu"]:
        raise ValueError("Invalid voltage interval")
    if not 0 < config["tolerance_kw"] < config["initial_kw"] <= config["ceiling_kw"]:
        raise ValueError("Require 0 < tolerance < initial <= ceiling")
    if config["audit_points"] < 3:
        raise ValueError("audit_points must be >= 3")
    lo, hi = config["daytime_hours"]
    if not 0 <= lo < hi < 24:
        raise ValueError("Invalid daytime window")
    return config


def paths(config):
    root = Path(config["data_dir"])
    return root / "raw" / config["feeder_id"], root / "processed" / config["feeder_id"]
