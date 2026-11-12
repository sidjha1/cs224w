from __future__ import annotations

from collections import Counter
import csv
import importlib.metadata
import platform
import random
from pathlib import Path

from .common import digest, paths, read_json, sha256, write_json
from .export import export_graph
from .graph import extract_graph
from .search import find_capacity
from .simulator import Simulator, prepare_snapshot
from .source import download, verify


def split_feeders(feeder_ids, seed=224):
    feeders = sorted(set(feeder_ids))
    random.Random(seed).shuffle(feeders)
    if len(feeders) < 3:
        return {"train": feeders, "val": [], "test": [],
                "note": "Pilot only: need at least three physical feeders for held-out validation/test."}
    nval = max(1, round(0.15 * len(feeders)))
    ntest = max(1, round(0.15 * len(feeders)))
    return {"train": feeders[nval + ntest:], "val": feeders[:nval], "test": feeders[nval:nval + ntest]}


def run_identity(config, source):
    return {"config": config, "source": source,
            "code": {path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
            "versions": {name: importlib.metadata.version(name) for name in
                         ("solar-hc", "OpenDSSDirect.py", "dss-python", "dss-python-backend", "numpy", "torch", "torch-geometric")}}


def prepare(config):
    source = verify(config)
    _, output = paths(config)
    run_path = output / "run.json"
    identity = run_identity(config, source)
    if run_path.exists():
        previous = read_json(run_path)
        if previous.get("identity") != identity:
            raise ValueError("Prepared run uses different settings/source/code/dependencies. Use a new data_dir.")
        snapshot = read_json(output / "snapshot.json")
        graph = read_json(output / "graph.json")
        fingerprint = digest({"identity": identity, "snapshot": snapshot, "graph": graph})
        if fingerprint != previous["fingerprint"]:
            raise ValueError("Prepared snapshot or graph was modified; checkpoint cannot be reused")
        simulator = Simulator(config)
        if simulator.dss.Basic.Version() != snapshot["engine_version"]:
            raise ValueError("Engine version differs from prepared run")
        simulator.apply_snapshot(snapshot)
        return simulator, snapshot, graph, fingerprint
    simulator, snapshot = prepare_snapshot(config)
    graph = extract_graph(simulator, snapshot)
    write_json(output / "graph.json", graph)
    fingerprint = digest({"identity": identity, "snapshot": snapshot, "graph": graph})
    write_json(run_path, {"fingerprint": fingerprint, "identity": identity, "config": config,
                         "python": platform.python_version(), "platform": platform.platform(),
                         "versions": identity["versions"]})
    write_json(output / "splits.json", split_feeders([config["feeder_id"]], config["seed"]))
    return simulator, snapshot, graph, fingerprint


def label(config, max_buses=None, prepared=None):
    if prepared is None:
        prepared = prepare(config)
    simulator, snapshot, graph, fingerprint = prepared
    _, output = paths(config)
    checkpoint_path = output / "labels.json"
    checkpoint = read_json(checkpoint_path) if checkpoint_path.exists() else {"fingerprint": fingerprint, "buses": {}}
    if checkpoint["fingerprint"] != fingerprint:
        raise ValueError("Checkpoint mismatch; use a different data_dir")
    labels = checkpoint["buses"]
    candidates = [node["name"] for node in graph["nodes"] if node["is_load"]]
    pending = [name for name in candidates if name not in labels]
    if max_buses is not None:
        pending = pending[:max_buses]
    for name in pending:
        if simulator.baseline["status"] != "ok" or not simulator.baseline["feasible"]:
            labels[name] = {"status": "baseline_violation", "hc_kw": None,
                            "baseline": simulator.baseline, "solve_count": 0, "elapsed_seconds": 0}
        else:
            simulator.set_injection_bus(name)
            labels[name] = find_capacity(simulator.evaluate, **{key: config[key] for key in
                                        ("initial_kw", "ceiling_kw", "tolerance_kw", "audit_points")})
        write_json(checkpoint_path, checkpoint)
        print(f'{len(labels)}/{len(candidates)} {name}: {labels[name]["status"]} '
              f'HC={labels[name]["hc_kw"]}', flush=True)
    data = export_graph(graph, labels, output)
    with (output / "labels.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["bus", "status", "hc_kw", "lower_bound_kw", "upper_bound_kw", "solve_count", "elapsed_seconds"])
        writer.writeheader()
        for name in candidates:
            record = labels.get(name, {"status": "pending"})
            writer.writerow({"bus": name, **{key: record.get(key) for key in writer.fieldnames if key != "bus"}})
    good = [record["hc_kw"] for record in labels.values() if record["status"] == "ok"]
    summary = {"feeder_id": config["feeder_id"], "topology": graph["topology"],
               "candidate_buses": len(candidates), "processed_buses": len(labels),
               "pending_buses": len(candidates) - len(labels), "valid_labels": int(data.label_mask.sum()),
               "statuses": dict(Counter(record["status"] for record in labels.values())),
               "hc_min_kw": min(good) if good else None, "hc_max_kw": max(good) if good else None,
               "total_label_seconds": sum(record["elapsed_seconds"] for record in labels.values()),
               "total_power_flow_solves": sum(record["solve_count"] for record in labels.values()),
               "baseline": snapshot["baseline"], "profile_index": snapshot["profile_index"],
               "existing_pv_systems": len(snapshot.get("pvs", [])),
               "existing_pv_kw": snapshot.get("existing_pv_kw", 0),
               "existing_pv_kvar": snapshot.get("existing_pv_kvar", 0),
               "existing_pv_dc_nameplate_kw": snapshot.get("existing_pv_dc_nameplate_kw", 0),
               "snapshot_rejections": len(snapshot["rejected_lower_load_snapshots"])}
    write_json(output / "summary.json", summary)
    return summary


def run(config, max_buses=None):
    download(config)
    return label(config, max_buses=max_buses, prepared=prepare(config))
