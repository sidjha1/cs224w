"""Independent boundary re-solves using a fresh engine for each labeled bus."""
from __future__ import annotations

import torch

from .common import paths, read_json, write_json
from .pipeline import prepare
from .simulator import Simulator


def validate(config):
    _, snapshot, graph, fingerprint = prepare(config)
    _, output = paths(config)
    checkpoint = read_json(output / "labels.json")
    if checkpoint["fingerprint"] != fingerprint:
        raise ValueError("Label checkpoint mismatch")
    tensor = torch.load(output / "graph.pt", weights_only=True)
    ids = {node["name"]: i for i, node in enumerate(graph["nodes"])}
    expected_mask = torch.zeros(len(ids), dtype=torch.bool)
    checks = {}
    for bus, record in checkpoint["buses"].items():
        if record["status"] != "ok":
            continue
        expected_mask[ids[bus]] = True
        if not 0 <= record["hc_kw"] < record["upper_bound_kw"] <= config["ceiling_kw"]:
            raise ValueError(f"Invalid capacity bracket at {bus}")
        if record["upper_bound_kw"] - record["hc_kw"] > config["tolerance_kw"]:
            raise ValueError(f"Capacity bracket too wide at {bus}")
        simulator = Simulator(config)
        simulator.apply_snapshot(snapshot)
        if not simulator.baseline["feasible"]:
            raise ValueError(f"Fresh baseline infeasible for {bus}")
        simulator.set_injection_bus(bus)
        low = simulator.evaluate(record["hc_kw"])
        high = simulator.evaluate(record["upper_bound_kw"])
        if low["status"] != "ok" or not low["feasible"] or high["status"] != "ok" or high["feasible"]:
            raise ValueError(f"Independent boundary validation failed for {bus}")
        if abs(tensor["y"][ids[bus]].item() - record["hc_kw"]) > 1e-4:
            raise ValueError(f"Exported target mismatch for {bus}")
        checks[bus] = {"lower_kw": record["hc_kw"], "upper_kw": record["upper_bound_kw"],
                       "at_lower": low, "at_upper": high}
    if not torch.equal(tensor["label_mask"], expected_mask):
        raise ValueError("Exported label mask mismatch")
    result = {"status": "ok", "validated_buses": len(checks),
              "pending_buses": sum(node["is_load"] for node in graph["nodes"]) - len(checkpoint["buses"]),
              "method": "fresh engine per bus; independent feasible-lower and physically infeasible-upper solves",
              "fingerprint": fingerprint, "checks": checks}
    write_json(output / "validation.json", result)
    return {key: value for key, value in result.items() if key not in ("checks", "fingerprint")}
