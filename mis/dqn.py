"""S2V-DQN-style Q-network for MIS, adapted from Dai et al., NeurIPS 2017.

The original paper (arXiv:1704.01665) attacks Min Vertex Cover, Max Cut, and TSP
with a graph-embedding Q-network and n-step Q-learning. MIS is the complement
of MVC, so the same iterative-greedy framework applies directly.

Key adaptations from the paper:

  - State    : the input graph G plus a per-node *tag* in {in_set, blocked,
               available}. `in_set` = already added to the IS; `blocked`
               = has an `in_set` neighbor and can never be added; `available`
               = the legal action space at this step.
  - Action   : pick one available node; mark it in_set; mark all neighbors blocked.
  - Reward   : +1 per addition (so the total return equals |IS|).
  - Terminal : every node is in_set or blocked.

We swap the paper's structure2vec embedding for the project's existing
ConditionedResidualGAT. Architecturally similar (residual message passing on
attribute-tagged nodes), but reuses the codebase's well-debugged GAT plumbing
and 11-D graph-level conditioning.

The Q-head follows the paper's form:
    Q(s, v) = theta_5 . ReLU([theta_6 . sum_u(mu_u),  theta_7 . mu_v])
where mu_v are the per-node embeddings after T residual GAT blocks on the
tagged graph.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, GraphNorm
from torch_geometric.utils import scatter


# ---- node tag --------------------------------------------------------------

# Tag layout: [in_set, blocked]. "available" is the implicit (0, 0).
TAG_DIM = 2
TAG_IN_SET = 0
TAG_BLOCKED = 1


def initial_tag(num_nodes: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """All nodes start out available (tag = (0, 0))."""
    return torch.zeros((num_nodes, TAG_DIM), dtype=torch.float, device=device)


def apply_action(tag: torch.Tensor, edge_index: torch.Tensor,
                 v: int) -> torch.Tensor:
    """Mark v as in_set and every neighbor of v as blocked. Returns a NEW tag
    tensor (does not mutate the input). v is assumed to be currently available.
    """
    out = tag.clone()
    out[v, TAG_IN_SET] = 1.0
    if edge_index.numel():
        # neighbors of v are dst entries where src == v
        mask = edge_index[0] == v
        nbrs = edge_index[1][mask]
        if nbrs.numel():
            out[nbrs, TAG_BLOCKED] = 1.0
    return out


def available_mask(tag: torch.Tensor) -> torch.Tensor:
    """Boolean mask of nodes that are still legal actions."""
    return (tag[:, TAG_IN_SET] == 0) & (tag[:, TAG_BLOCKED] == 0)


# ---- Q-network -------------------------------------------------------------

class MisQNet(nn.Module):
    """Q-network: per-node Q-value on a tagged graph.

    Args mirror ConditionedResidualGAT's, with one addition:
      base_node_in : the original (un-tagged) node feature dim (default 2).
                     Tag features (TAG_DIM=2) are concatenated INSIDE.
    """

    def __init__(self, base_node_in: int = 2, graph_in: int = 11,
                 hidden: int = 128, num_layers: int = 4, heads: int = 4,
                 dropout: float = 0.0, norm: str = "graph"):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(f"hidden ({hidden}) must be divisible by heads ({heads})")
        self.kind = "mis_qnet"
        self.dropout = dropout
        self.num_layers = num_layers
        self.base_node_in = base_node_in
        self.graph_in = graph_in
        self.hidden = hidden

        in_dim = base_node_in + TAG_DIM + graph_in
        self.in_proj = nn.Linear(in_dim, hidden)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GATConv(hidden, hidden // heads,
                                      heads=heads, concat=True, dropout=dropout))
            self.norms.append(GraphNorm(hidden) if norm == "graph"
                              else nn.LayerNorm(hidden))

        # Q-head: theta_5 . ReLU(concat(theta_6 . pooled, theta_7 . per_node))
        self.theta_pool = nn.Linear(hidden, hidden, bias=False)
        self.theta_node = nn.Linear(hidden, hidden, bias=False)
        self.theta_q = nn.Linear(2 * hidden, 1)

    def encode(self, data: Data | Batch, tag: torch.Tensor) -> torch.Tensor:
        """Run the residual GAT stack on the tagged graph; return [n, hidden]."""
        x = data.x
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else None
        graph_x = data.graph_x

        # Broadcast graph-level features to every node
        if batch is None:
            if graph_x.dim() == 1:
                graph_x = graph_x.unsqueeze(0)
            broadcast = graph_x.expand(x.size(0), -1)
        else:
            broadcast = graph_x[batch]

        h = F.relu(self.in_proj(torch.cat([x, tag, broadcast], dim=-1)))
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index)
            if isinstance(norm, GraphNorm):
                h_new = norm(h_new, batch) if batch is not None else norm(h_new)
            else:
                h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new
        return h

    def forward(self, data: Data | Batch, tag: torch.Tensor) -> torch.Tensor:
        """Per-node Q-values [n] on the tagged graph."""
        mu = self.encode(data, tag)                       # [n, hidden]
        # Pooled state: sum over the (per-graph) node embeddings
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else None
        if batch is None:
            pooled = mu.sum(dim=0, keepdim=True).expand_as(mu)
        else:
            pool_per_graph = scatter(mu, batch, dim=0, reduce="sum")
            pooled = pool_per_graph[batch]
        h = torch.cat([F.relu(self.theta_pool(pooled)),
                       F.relu(self.theta_node(mu))], dim=-1)
        return self.theta_q(h).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
