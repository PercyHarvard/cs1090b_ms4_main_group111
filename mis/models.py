"""GNN architectures for node-level MIS prediction.

The headline model — and the architecture used in the report's main results —
is :class:`ConditionedResidualGAT`, which mirrors the diagram in the MS4
slide deck:

    1. Per-node features         x_v = [deg(v), deg(v)/(n-1)]                (R^2)
    2. Graph-level metadata      g   = [density, log|V|, one-hot family]      (R^11)
    3. Broadcast g to every node, concat with x_v, project to hidden dim
       => conditioned node embedding H^(0)
    4. Stack of K GAT conv layers, each with multi-head attention,
       a residual skip from input to output, GraphNorm/LayerNorm, ReLU,
       and dropout
    5. 2-layer MLP node head -> 1 logit per node
    6. sigmoid -> per-node MIS probability score (consumed by the decoder)

Two simpler baselines (:func:`make_baseline_gcn`, :func:`make_baseline_sage`)
are kept around purely for ablation tables — they do **not** receive the
graph-level conditioning, so the architecture comparison isolates the effect
of the GAT/residual/conditioning combination.

Forward signatures take a PyG ``Data`` or ``Batch`` directly so that we can
plumb ``data.graph_x`` (graph-level features) without breaking PyG's batching.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, GCNConv, GraphNorm, SAGEConv


# ---- main model: conditioned residual GAT ---------------------------------

class ConditionedResidualGAT(nn.Module):
    """GAT-based MIS scorer with graph-level conditioning + residual blocks.

    Args:
        node_in    : per-node feature dim (default 2: degree + normalized degree).
        graph_in   : graph-level feature dim broadcast to every node
                     (default 11: density + log|V| + 9 family one-hot).
        hidden     : hidden width of every GAT layer (must be divisible by ``heads``).
        num_layers : number of GAT layers (3-5 in the slide; default 4).
        heads      : attention heads per layer (default 4).
        dropout    : dropout between layers and inside the head MLP.
        norm       : "graph" -> GraphNorm, "layer" -> LayerNorm.
        readout_hidden : hidden width of the 2-layer scoring MLP.
    """

    def __init__(self, node_in: int = 2, graph_in: int = 11,
                 hidden: int = 64, num_layers: int = 4, heads: int = 4,
                 dropout: float = 0.2, norm: str = "graph",
                 readout_hidden: int | None = None):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(f"hidden ({hidden}) must be divisible by heads ({heads})")
        self.kind = "gat_cond"
        self.dropout = dropout
        self.num_layers = num_layers

        self.in_proj = nn.Linear(node_in + graph_in, hidden)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GATConv(hidden, hidden // heads,
                                      heads=heads, concat=True,
                                      dropout=dropout))
            self.norms.append(GraphNorm(hidden) if norm == "graph"
                              else nn.LayerNorm(hidden))
        self._norm_kind = norm

        readout_hidden = readout_hidden or hidden
        self.head = nn.Sequential(
            nn.Linear(hidden, readout_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(readout_hidden, 1),
        )

    # -- forward ------------------------------------------------------------

    def _broadcast_graph(self, graph_x: torch.Tensor, n_nodes: int,
                         batch: torch.Tensor | None) -> torch.Tensor:
        """Replicate graph-level features to every node."""
        if batch is None:
            # single graph: graph_x is [1, graph_in] or [graph_in]
            if graph_x.dim() == 1:
                graph_x = graph_x.unsqueeze(0)
            return graph_x.expand(n_nodes, -1)
        # batched: graph_x is [num_graphs, graph_in]
        return graph_x[batch]

    def forward(self, data: Data | Batch) -> torch.Tensor:
        x = data.x
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else None
        graph_x = data.graph_x

        broadcast = self._broadcast_graph(graph_x, x.size(0), batch)
        h = F.relu(self.in_proj(torch.cat([x, broadcast], dim=-1)))

        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index)
            if isinstance(norm, GraphNorm):
                h_new = norm(h_new, batch) if batch is not None else norm(h_new)
            else:
                h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new   # residual

        return self.head(h).squeeze(-1)


# ---- baselines (no graph-level conditioning) -------------------------------

class _SimpleGNN(nn.Module):
    """GCN- or SAGE-based baseline for ablation. No graph-level conditioning."""

    def __init__(self, kind: str, node_in: int = 2, hidden: int = 64,
                 num_layers: int = 4, dropout: float = 0.2):
        super().__init__()
        self.kind = kind
        self.dropout = dropout
        self.in_proj = nn.Linear(node_in, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            if kind == "gcn":
                self.convs.append(GCNConv(hidden, hidden))
            elif kind == "sage":
                self.convs.append(SAGEConv(hidden, hidden, aggr="mean"))
            else:
                raise ValueError(f"unknown baseline kind {kind!r}")
            self.norms.append(GraphNorm(hidden))
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, data: Data | Batch) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else None
        h = F.relu(self.in_proj(x))
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index)
            h_new = norm(h_new, batch) if batch is not None else norm(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new
        return self.head(h).squeeze(-1)


def make_baseline_gcn(**kwargs) -> _SimpleGNN:
    return _SimpleGNN("gcn", **kwargs)


def make_baseline_sage(**kwargs) -> _SimpleGNN:
    return _SimpleGNN("sage", **kwargs)


# ---- factory ---------------------------------------------------------------

def make_model(kind: str, **kwargs) -> nn.Module:
    """Convenience factory.

    Kinds:
        ``gat_cond`` (default headline model) -> ConditionedResidualGAT
        ``gcn``  -> baseline GCN (no conditioning)
        ``sage`` -> baseline GraphSAGE (no conditioning)
    """
    if kind in ("gat_cond", "gat", "gnn"):
        return ConditionedResidualGAT(**kwargs)
    if kind == "gcn":
        return make_baseline_gcn(**kwargs)
    if kind == "sage":
        return make_baseline_sage(**kwargs)
    raise ValueError(f"unknown kind {kind!r}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
