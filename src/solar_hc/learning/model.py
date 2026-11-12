"""SweepGNN: shared leaf-to-root and root-to-leaf MLP recurrences."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .data import TreeBatch


def mlp(input_dim, hidden_dim, output_dim, dropout, activate_output=True):
    layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout),
              nn.Linear(hidden_dim, output_dim)]
    if activate_output:
        layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class SweepGNN(nn.Module):
    """One prediction in AC kW per node, in the input's original node order.

    u_v = phi_up([x_v, sum_c psi_up([u_c, e_cv])])
    d_v = phi_down([u_v, psi_down([d_parent, e_parent_v])])
    HC_v = target_scale_kw * softplus(readout([u_v, d_v]))

    Root downward context is a shared learned vector. The two sweeps have
    separate parameters, shared across all depths/feeders. Siblings at one depth
    are processed together. TreeBatch supplies separate edge directions so the
    exported is_reverse_message feature is respected.
    """

    def __init__(self, node_dim, edge_dim, hidden_dim=64, dropout=0.0, target_scale_kw=100.0):
        super().__init__()
        if min(node_dim, edge_dim, hidden_dim) <= 0 or target_scale_kw <= 0 or not 0 <= dropout < 1:
            raise ValueError("Invalid model dimensions, dropout, or target scale")
        self.config = dict(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim,
                           dropout=dropout, target_scale_kw=target_scale_kw)
        self.hidden_dim = hidden_dim
        self.target_scale_kw = target_scale_kw
        self.phi_up = mlp(node_dim + hidden_dim, hidden_dim, hidden_dim, dropout)
        self.psi_up = mlp(hidden_dim + edge_dim, hidden_dim, hidden_dim, dropout)
        self.phi_down = mlp(2 * hidden_dim, hidden_dim, hidden_dim, dropout)
        self.psi_down = mlp(hidden_dim + edge_dim, hidden_dim, hidden_dim, dropout)
        self.root_context = nn.Parameter(torch.zeros(hidden_dim))
        self.readout = mlp(2 * hidden_dim, hidden_dim, 1, dropout, activate_output=False)
        # These buffers are fitted only on training data and saved with weights.
        self.register_buffer("node_mean", torch.zeros(node_dim))
        self.register_buffer("node_std", torch.ones(node_dim))
        self.register_buffer("edge_mean", torch.zeros(edge_dim))
        self.register_buffer("edge_std", torch.ones(edge_dim))

    @torch.no_grad()
    def set_feature_stats(self, stats):
        for name in ("node_mean", "node_std", "edge_mean", "edge_std"):
            target = getattr(self, name)
            target.copy_(stats[name].to(target))

    def forward(self, graph: TreeBatch):
        x = (graph.x - self.node_mean) / self.node_std
        edge_up = (graph.up_attr - self.edge_mean) / self.edge_std
        edge_down = (graph.down_attr - self.edge_mean) / self.edge_std
        levels = graph.level_nodes
        upward = [None] * len(levels)
        for depth in range(len(levels) - 1, -1, -1):
            nodes = levels[depth]
            aggregate = x.new_zeros(len(nodes), self.hidden_dim)
            if depth + 1 < len(levels):
                children = levels[depth + 1]
                messages = self.psi_up(torch.cat((upward[depth + 1], edge_up[children]), dim=-1))
                aggregate = aggregate.index_add(0, graph.parent_slots[depth + 1], messages)
            upward[depth] = self.phi_up(torch.cat((x[nodes], aggregate), dim=-1))

        downward = []
        for depth, nodes in enumerate(levels):
            if depth == 0:
                context = self.root_context.expand(len(nodes), -1)
            else:
                parent_states = downward[depth - 1][graph.parent_slots[depth]]
                context = self.psi_down(torch.cat((parent_states, edge_down[nodes]), dim=-1))
            downward.append(self.phi_down(torch.cat((upward[depth], context), dim=-1)))

        up = torch.cat(upward)[graph.restore_order]
        down = torch.cat(downward)[graph.restore_order]
        raw = self.readout(torch.cat((up, down), dim=-1)).squeeze(-1)
        return F.softplus(raw) * self.target_scale_kw


def load_checkpoint(path, device="cpu"):
    """Reconstruct a model with its fitted normalization; no optimizer required."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = SweepGNN(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), checkpoint
