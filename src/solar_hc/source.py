"""Fetch only the chosen feeder's referenced DSS and annual CSV files from OEDI."""
from __future__ import annotations

import concurrent.futures
import posixpath
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

from .common import paths, read_json, sha256, write_json

BUCKET = "https://oedi-data-lake.s3.amazonaws.com/"


def references(text):
    refs = []
    for line in text.splitlines():
        line = line.split("!", 1)[0].strip()
        match = re.match(r'(?i)(?:redirect|compile|buscoords)\s+["\[]?([^"\]\s]+)', line)
        if match:
            refs.append(match[1])
        refs.extend(m[1] or m[2] for m in re.finditer(r'(?i)\bfile\s*=\s*(?:"([^"]+)"|([^\s,)]+))', line))
    return refs


def dependency_key(parent, reference):
    key = posixpath.normpath(posixpath.join(posixpath.dirname(parent), reference))
    if key.startswith("../") or key.startswith("/") or ":" in key:
        raise ValueError(f"Reference escapes source prefix: {reference}")
    return key


def download(config):
    raw, _ = paths(config)
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = raw / "manifest.json"
    old = read_json(manifest_path) if manifest_path.exists() else {}
    if old and old["source_prefix"] != config["source_prefix"]:
        raise ValueError("Existing source prefix differs; use a different data directory")
    files = {}

    def fetch(key):
        destination = raw / key
        previous = old.get("files", {}).get(key)
        if not (previous and destination.is_file() and sha256(destination) == previous["sha256"]):
            destination.parent.mkdir(parents=True, exist_ok=True)
            url = BUCKET + urllib.parse.quote(config["source_prefix"] + key, safe="/")
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=60) as response:
                        content = response.read()
                    temporary = destination.with_suffix(destination.suffix + ".part")
                    temporary.write_bytes(content)
                    temporary.replace(destination)
                    break
                except OSError:
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
        return key, {"bytes": destination.stat().st_size, "sha256": sha256(destination)}

    pending = {config["master_key"]}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        while pending:
            batch = sorted(pending - files.keys())
            if not batch:
                break
            pending = set()
            for key, record in pool.map(fetch, batch):
                files[key] = record
                if key.lower().endswith(".dss"):
                    pending.update(dependency_key(key, ref) for ref in references((raw / key).read_text()))
            write_json(manifest_path, {"source_prefix": config["source_prefix"],
                                      "master_key": config["master_key"], "files": files})
            print(f"Downloaded/verified {len(files)} source files", flush=True)
    return read_json(manifest_path)


def verify(config):
    raw, _ = paths(config)
    manifest = read_json(raw / "manifest.json")
    if manifest["master_key"] != config["master_key"] or manifest["source_prefix"] != config["source_prefix"]:
        raise ValueError("Configuration does not match downloaded source")
    for key, record in manifest["files"].items():
        if sha256(raw / key) != record["sha256"]:
            raise ValueError(f"Source checksum mismatch: {key}")
    return manifest
