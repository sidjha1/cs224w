"""Load tensor exports and batch trees without PyG's automatic index offsets."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import torch

from ..common import read_json, sha256


@dataclass
class FeederGraph:
    x: torch.Tensor
    y: torch.Tensor
    label_mask: torch.Tensor
    parent_index: torch.Tensor
    depth: torch.Tensor
    down_attr: torch.Tensor  # Incoming parent -> child component features, indexed by child.
    up_attr: torch.Tensor  # Incoming child -> parent features; root rows are placeholders.
    feeder_id: str
    path: str
    schema: dict


@dataclass
class TreeBatch:
    x: torch.Tensor
    y: torch.Tensor
    label_mask: torch.Tensor
    down_attr: torch.Tensor
    up_attr: torch.Tensor
    level_nodes: tuple[torch.Tensor, ...]
    parent_slots: tuple[torch.Tensor, ...]
    restore_order: torch.Tensor
    graph_index: torch.Tensor
    feeder_ids: tuple[str, ...]

    def to(self, device):
        """Return a device copy; schedules move along with the feature tensors."""
        fields = {}
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                fields[name] = value.to(device)
            elif name in ("level_nodes", "parent_slots"):
                fields[name] = tuple(item.to(device) for item in value)
        return replace(self, **fields)


def load_graph(path):
    path = Path(path).resolve()
    data = torch.load(path, map_location="cpu", weights_only=True)
    schema_path = path.with_name("feature_schema.json")
    schema = read_json(schema_path)
    x, y, mask = data["x"].float(), data["y"].float(), data["label_mask"].bool()
    parent, depth = data["parent_index"].long(), data["depth"].long()
    edge_index, direction = data["edge_index"].long(), data["edge_direction"].long()
    attr = data["edge_attr"].float()
    n = x.shape[0]
    if n == 0 or x.ndim != 2 or any(t.shape != (n,) for t in (y, mask, parent, depth)):
        raise ValueError(f"Invalid node tensor shapes in {path}")
    if x.shape[1] != len(schema["node_features"]) or attr.ndim != 2 or attr.shape[1] != len(schema["edge_features"]):
        raise ValueError(f"Feature schema mismatch in {path}")
    if not torch.isfinite(x).all() or not torch.isfinite(attr).all() or not torch.isfinite(y[mask]).all() or (y[mask] < 0).any():
        raise ValueError(f"Invalid features or labeled targets in {path}")
    roots = torch.where(parent == -1)[0]
    nonroot = parent >= 0
    if roots.numel() != 1 or (parent < -1).any() or (parent >= n).any() or (depth < 0).any():
        raise ValueError(f"Expected one rooted tree in {path}")
    if depth[roots].item() != 0 or not torch.equal(depth[nonroot], depth[parent[nonroot]] + 1):
        raise ValueError(f"Inconsistent tree depths/parents in {path}")
    if edge_index.shape != (2, 2 * (n - 1)) or direction.shape != (2 * (n - 1),) or attr.shape[0] != direction.numel():
        raise ValueError(f"Expected both directions of each tree edge in {path}")
    if (edge_index < 0).any() or (edge_index >= n).any() or not ((direction == 1) | (direction == -1)).all():
        raise ValueError(f"Invalid directed edges in {path}")
    down_attr = attr.new_zeros(n, attr.shape[1])
    up_attr = torch.zeros_like(down_attr)
    expected_children = torch.where(nonroot)[0]
    for sign, target in ((1, down_attr), (-1, up_attr)):
        selected = direction == sign
        parents, children = edge_index[:, selected] if sign == 1 else edge_index[:, selected].flip(0)
        if not torch.equal(children.sort().values, expected_children) or not torch.equal(parent[children], parents):
            raise ValueError(f"Directed edges disagree with rooted tree in {path}")
        target[children] = attr[selected]
    return FeederGraph(x, y, mask, parent, depth, down_attr, up_attr,
                       str(data["feeder_id"]), str(path), schema)


def collate_trees(graphs):
    """Batch a forest; root sentinels stay -1 and all other indices are offset.

    parent_slots[d] maps each node at depth d to its parent's position at depth
    d-1. Each sweep therefore processes one depth at a time without recursion,
    dense adjacency, or copying an N-by-hidden state tensor at every depth.
    """
    if not graphs:
        raise ValueError("Cannot batch zero graphs")
    offset, parents = 0, []
    for graph in graphs:
        parents.append(torch.where(graph.parent_index >= 0, graph.parent_index + offset, -1))
        offset += len(graph.x)
    parent = torch.cat(parents)
    depth = torch.cat([graph.depth for graph in graphs])
    levels = tuple(torch.where(depth == d)[0] for d in range(int(depth.max()) + 1))
    slots = torch.empty(len(parent), dtype=torch.long)
    for nodes in levels:
        slots[nodes] = torch.arange(len(nodes))
    parent_slots = (torch.empty(0, dtype=torch.long),) + tuple(slots[parent[nodes]] for nodes in levels[1:])
    return TreeBatch(
        x=torch.cat([g.x for g in graphs]), y=torch.cat([g.y for g in graphs]),
        label_mask=torch.cat([g.label_mask for g in graphs]),
        down_attr=torch.cat([g.down_attr for g in graphs]), up_attr=torch.cat([g.up_attr for g in graphs]),
        level_nodes=levels, parent_slots=parent_slots, restore_order=torch.argsort(torch.cat(levels)),
        graph_index=torch.repeat_interleave(torch.arange(len(graphs)), torch.tensor([len(g.x) for g in graphs])),
        feeder_ids=tuple(g.feeder_id for g in graphs))


def fit_feature_stats(graphs):
    """Population moments over training nodes and directed components only."""
    # Accumulate in float64 without concatenating all training graphs in memory.
    def moments(tensors, width):
        count = 0
        total, squares = torch.zeros(width, dtype=torch.float64), torch.zeros(width, dtype=torch.float64)
        for tensor in tensors:
            values = tensor.double()
            count += len(values)
            total += values.sum(0)
            squares += values.square().sum(0)
        if count == 0:
            return torch.zeros(width), torch.ones(width)
        mean = total / count
        std = (squares / count - mean.square()).clamp_min(0).sqrt()
        # Constant features keep their information and avoid division by tiny noise.
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        return mean.float(), std.float()
    node_mean, node_std = moments((g.x for g in graphs), graphs[0].x.shape[1])
    edge_mean, edge_std = moments((attrs[g.parent_index >= 0] for g in graphs
                                   for attrs in (g.down_attr, g.up_attr)), graphs[0].down_attr.shape[1])
    return dict(node_mean=node_mean, node_std=node_std, edge_mean=edge_mean, edge_std=edge_std)


def load_splits(graph_paths, split_file):
    """All snapshots/scenarios of a physical feeder follow its split assignment."""
    resolved = [str(Path(path).resolve()) for path in graph_paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Duplicate graph paths")
    splits = read_json(split_file)
    assignment = {}
    for split in ("train", "val", "test"):
        for feeder in splits[split]:
            if feeder in assignment:
                raise ValueError(f"Repeated feeder assignment: {feeder}")
            assignment[feeder] = split
    groups = {key: [] for key in ("train", "val", "test")}
    reference_schema = None
    records = []
    for path in resolved:
        graph = load_graph(path)
        if graph.feeder_id not in assignment:
            raise ValueError(f"Feeder absent from split manifest: {graph.feeder_id}")
        schema = {key: graph.schema[key] for key in ("node_features", "edge_features")}
        if reference_schema is not None and reference_schema != schema:
            raise ValueError(f"Feature order differs between graphs: {path}")
        reference_schema = schema
        split = assignment[graph.feeder_id]
        groups[split].append(graph)
        records.append({"path": path, "sha256": sha256(path), "feeder_id": graph.feeder_id, "split": split})
    loaded_ids = {graph.feeder_id for graphs in groups.values() for graph in graphs}
    if set(assignment) - loaded_ids:
        raise ValueError(f"Split manifest references unloaded feeders: {sorted(set(assignment) - loaded_ids)}")
    if not groups["train"]:
        raise ValueError("At least one training graph is required")
    for split, graphs in groups.items():
        if graphs and not any(graph.label_mask.any() for graph in graphs):
            raise ValueError(f"{split} split has no valid labels")
    return groups, reference_schema, records
