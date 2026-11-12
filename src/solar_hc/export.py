from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from .common import write_json

NODE_FEATURES = ["load_kw", "load_kvar", "existing_pv_kw", "kv_base_ln", "phase_1", "phase_2", "phase_3",
                 "is_root", "is_load", "is_transformer_terminal", "is_regulator_terminal", "depth",
                 "voltage_min_pu", "voltage_max_pu", "existing_pv_kvar"]
EDGE_KINDS = ["line", "switch", "transformer", "regulator"]
EDGE_FEATURES = (["is_" + k for k in EDGE_KINDS] +
                 ["length_m", "normal_amps", "normal_kva", "xhl_percent", "xht_percent", "xlt_percent"] +
                 [f"r_{i}{j}_ohm" for i in range(3) for j in range(3)] +
                 [f"x_{i}{j}_ohm" for i in range(3) for j in range(3)] +
                 [f"wdg_{i}_{field}" for i in range(1, 4) for field in ("kv", "kva", "r_percent", "tap", "delta")] +
                 [f"terminal_{t}_conductor_{c}_node" for t in range(1, 4) for c in range(1, 5)] +
                 [f"terminal_{t}_is_parent_bus" for t in range(1, 4)] + ["is_reverse_message"])


def edge_vector(edge):
    vector = [float(edge["kind"] == kind) for kind in EDGE_KINDS]
    vector += [edge.get(field, 0.0) for field in EDGE_FEATURES[4:10]]
    for key in ("r_matrix_ohm", "x_matrix_ohm"):
        matrix = np.zeros((3, 3))
        if key in edge:
            source = np.asarray(edge[key])
            matrix[:source.shape[0], :source.shape[1]] = source
        vector.extend(matrix.ravel().tolist())
    for i in range(3):
        winding = edge["windings"][i] if i < len(edge["windings"]) else {}
        vector += [float(winding.get(field, 0)) for field in ("kv", "kva", "r_percent", "tap", "delta")]
    for i in range(3):
        terminal = edge["terminal_nodes"][i] if i < len(edge["terminal_nodes"]) else []
        vector += terminal + [-1] * (4 - len(terminal))
    for i in range(3):
        vector.append(float(i < len(edge["connections"]) and edge["connections"][i].split(".")[0].lower() == edge["parent"]))
    vector.append(0.0)
    return vector


def export_graph(graph, labels, output):
    nodes = graph["nodes"]
    ids = {node["name"]: i for i, node in enumerate(nodes)}
    x = []
    for node in nodes:
        values = {**node, **{f"phase_{i}": i in node["phases"] for i in (1, 2, 3)}}
        x.append([float(values[key]) for key in NODE_FEATURES])
    forward = [[ids[edge["parent"]], ids[edge["child"]]] for edge in graph["edges"]]
    # Both message directions share physical component features; edge_direction
    # identifies root-to-leaf (+1) and leaf-to-root (-1) without duplicating metadata.
    bidirectional = forward + [[b, a] for a, b in forward]
    edge_attr = [edge_vector(edge) for edge in graph["edges"]]
    reverse_attr = [vector[:-1] + [1.0] for vector in edge_attr]
    y = [0.0] * len(nodes)
    mask = [False] * len(nodes)
    for name, record in labels.items():
        if record["status"] == "ok":
            y[ids[name]] = record["hc_kw"]
            mask[ids[name]] = True
    parents = [-1 if node["parent"] is None else ids[node["parent"]] for node in nodes]
    data = Data(x=torch.tensor(x, dtype=torch.float32),
                edge_index=torch.tensor(bidirectional, dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(edge_attr + reverse_attr, dtype=torch.float32),
                y=torch.tensor(y, dtype=torch.float32), label_mask=torch.tensor(mask),
                candidate_mask=torch.tensor([node["is_load"] for node in nodes]),
                parent_index=torch.tensor(parents, dtype=torch.long),
                depth=torch.tensor([node["depth"] for node in nodes], dtype=torch.long),
                root_index=torch.tensor([ids[graph["root"]]], dtype=torch.long),
                edge_direction=torch.tensor([1] * len(forward) + [-1] * len(forward)),
                down_order=torch.arange(len(nodes)), up_order=torch.arange(len(nodes) - 1, -1, -1),
                feeder_id=graph["feeder_id"], bus_names=[node["name"] for node in nodes])
    data.validate(raise_on_error=True)
    if not torch.isfinite(data.x).all() or not torch.isfinite(data.edge_attr).all():
        raise ValueError("Nonfinite graph features")
    # Save tensor primitives rather than a pickled PyG class (weights_only load works).
    temporary = output / "graph.pt.tmp"
    torch.save(data.to_dict(), temporary)
    temporary.replace(output / "graph.pt")
    write_json(output / "feature_schema.json", {
        "node_features": NODE_FEATURES, "edge_features": EDGE_FEATURES,
        "target": "independent incremental AC PV capacity, kW, conservative feasible lower bound",
        "label_mask": "True only for finite bracketed status=ok labels; y=0 elsewhere is a placeholder",
        "edge_direction": "+1 root-to-leaf, -1 leaf-to-root; matrix terminal ordering is in graph.json",
        "phase_ids": "DSS local conductor identifiers, including split-phase legs; not global ABC phase labels",
        "matrix_order": "original DSS conductor order, padded to 3x3; see raw connections in graph.json",
        "terminal_nodes": "DSS node IDs per original terminal/conductor: 0 ground, -1 padding; parent-bus flags map terminal order to tree order",
        "transformer_impedance": "winding %R and XHL/XHT/XLT percent on component base; not line ohms",
        "normalization": "raw units; fit transforms using training feeders only",
        "existing_pv": "snapshot AC terminal injection, kW/kvar; inverter controls settled at baseline then fixed for HC searches",
        "schema_version": 2,
        "batching": "parent_index/down_order/up_order/root_index are local indices; offset explicitly for PyG batches",
    })
    return data
