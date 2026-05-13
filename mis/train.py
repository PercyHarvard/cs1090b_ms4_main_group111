"""Training loops for unsupervised + supervised MIS GNNs.

Unsupervised loss (Erdős-GNN-style relaxation, Karalias & Loukas, NeurIPS 2020):

    L_unsup(p) = - sum_v p_v + lambda * sum_{(u,v) in E} p_u * p_v

The size term rewards larger soft sets; the edge term penalizes adjacent
pairs that are both selected. We sum only one direction of each undirected
edge (src < dst) so the bidirectional COO doesn't double-count.

Supervised loss (used as a sanity baseline only):

    L_sup(p, y) = BCE(p, y) + lambda * edge_penalty / n
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader


# ---- losses ----------------------------------------------------------------

def edge_penalty(probs: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Sum of p_u * p_v over undirected edges (one direction only)."""
    if edge_index.numel() == 0:
        return torch.zeros((), device=probs.device)
    src, dst = edge_index[0], edge_index[1]
    mask = src < dst
    return (probs[src[mask]] * probs[dst[mask]]).sum()


def unsupervised_loss(logits: torch.Tensor, edge_index: torch.Tensor,
                      lam: float = 1.0,
                      n_nodes: int | None = None,
                      logit_l2: float = 0.0) -> dict:
    """Erdős-GNN-style unsupervised IS loss with an anti-saturation regularizer.

    L = - sum p_v + lam * sum_{(u,v) in E, u<v} p_u p_v + logit_l2 * mean(z^2)

    where p_v = sigmoid(z_v). The first two terms are the standard
    differentiable independent-set objective (size reward + edge penalty).
    The ``logit_l2`` term is an L2 penalty on the *logits* themselves; it
    addresses the practical collapse mode where the model drives all p_v
    toward 0. Once that happens, sigmoid'(z_v) = p_v(1-p_v) → 0 and the
    gradient back through to z_v vanishes — a saturation trap. The L2 term
    has gradient 2 * logit_l2 * z_v / N, which is *non-vanishing* at any
    saturation point and pulls logits back toward 0, keeping the rest of
    the loss differentiable. Default 0.0 to opt in explicitly.

    If ``n_nodes`` is given, divides the size + edge terms by it so the
    per-batch loss scale is comparable across graphs of different sizes.
    """
    p = torch.sigmoid(logits)
    size = p.sum()
    edge = edge_penalty(p, edge_index)
    if n_nodes is not None and n_nodes > 0:
        raw = -size / n_nodes + lam * edge / n_nodes
    else:
        raw = -size + lam * edge
    if logit_l2 > 0:
        loss = raw + logit_l2 * (logits ** 2).mean()
    else:
        loss = raw
    # ``loss`` is what gets backpropped (includes regularizer);
    # ``loss_raw`` is the IS objective alone — use it for monitoring + early stop.
    return {"loss": loss, "loss_raw": raw.detach(),
            "size_term": size.detach(), "edge_term": edge.detach()}


def supervised_loss(logits: torch.Tensor, y: torch.Tensor,
                    edge_index: torch.Tensor, lam: float = 1.0) -> dict:
    bce = F.binary_cross_entropy_with_logits(logits, y)
    p = torch.sigmoid(logits)
    edge = edge_penalty(p, edge_index)
    n = max(1, y.numel())
    loss = bce + lam * edge / n
    return {"loss": loss, "loss_raw": loss.detach(),
            "bce": bce.detach(), "edge_term": edge.detach()}


# ---- training driver -------------------------------------------------------

@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32
    lr: float = 5e-4
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    lam: float = 1.0
    logit_l2: float = 1e-2  # anti-saturation regularizer; 0 disables
    mode: str = "unsupervised"   # "unsupervised" | "supervised"
    patience: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    verbose: bool = True
    normalize_size: bool = False  # divide -size + lam*edge by total batch nodes
    checkpoint_path: str | None = None  # if set, save best weights here on every improvement


@dataclass
class TrainHistory:
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    train_size: list = field(default_factory=list)
    val_size: list = field(default_factory=list)
    epoch_time: list = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("inf")


def _step(model: nn.Module, batch: Batch, cfg: TrainConfig, train: bool):
    """Forward + (optional) backward on one minibatch. Returns metric dict."""
    logits = model(batch)
    n_nodes = batch.num_nodes if cfg.normalize_size else None
    if cfg.mode == "supervised":
        if not hasattr(batch, "y") or batch.y is None:
            raise ValueError("supervised mode requires y on every Data")
        out = supervised_loss(logits, batch.y, batch.edge_index, lam=cfg.lam)
    else:
        out = unsupervised_loss(logits, batch.edge_index, lam=cfg.lam,
                                n_nodes=n_nodes, logit_l2=cfg.logit_l2)
    loss = out["loss"]
    if train:
        loss.backward()
    return {k: (float(v.detach()) if torch.is_tensor(v) else v)
            for k, v in out.items()}


def train_model(model: nn.Module, train_set: Iterable[Data],
                val_set: Iterable[Data], cfg: TrainConfig | None = None
                ) -> TrainHistory:
    """Train ``model`` for cfg.epochs; return TrainHistory. Restores best-val state."""
    cfg = cfg or TrainConfig()
    model = model.to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    train_loader = DataLoader(list(train_set), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(list(val_set), batch_size=cfg.batch_size, shuffle=False)

    history = TrainHistory()
    best_state = None
    bad_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        model.train()
        train_acc = {"loss": 0.0, "raw": 0.0, "size": 0.0, "n": 0}
        for batch in train_loader:
            batch = batch.to(cfg.device)
            opt.zero_grad()
            out = _step(model, batch, cfg, train=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            train_acc["loss"] += out["loss"]
            train_acc["raw"]  += out.get("loss_raw", out["loss"])
            train_acc["size"] += out.get("size_term", 0.0)
            train_acc["n"] += 1

        model.eval()
        val_acc = {"loss": 0.0, "raw": 0.0, "size": 0.0, "n": 0}
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(cfg.device)
                out = _step(model, batch, cfg, train=False)
                val_acc["loss"] += out["loss"]
                val_acc["raw"]  += out.get("loss_raw", out["loss"])
                val_acc["size"] += out.get("size_term", 0.0)
                val_acc["n"] += 1

        train_loss = train_acc["raw"] / max(1, train_acc["n"])
        val_loss = (val_acc["raw"] / max(1, val_acc["n"])
                    if val_acc["n"] else float("inf"))
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.train_size.append(train_acc["size"] / max(1, train_acc["n"]))
        history.val_size.append(val_acc["size"] / max(1, val_acc["n"]))
        history.epoch_time.append(time.time() - t0)

        if cfg.verbose:
            print(f"  epoch {epoch:3d} | train {train_loss:8.3f} "
                  f"| val {val_loss:8.3f} "
                  f"| p-mass tr {history.train_size[-1]:6.2f} val {history.val_size[-1]:6.2f} "
                  f"| {history.epoch_time[-1]:5.1f}s")

        if val_loss < history.best_val_loss - 1e-4:
            history.best_val_loss = val_loss
            history.best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
            # Persist best-so-far to disk so interruption doesn't lose progress.
            if cfg.checkpoint_path is not None:
                torch.save(best_state, cfg.checkpoint_path)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                if cfg.verbose:
                    print(f"  early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


# ---- inference -------------------------------------------------------------

@torch.no_grad()
def predict_scores(model: nn.Module, data: Data,
                   device: str | None = None) -> torch.Tensor:
    """Run ``model`` on a single graph; return per-node sigmoid scores [n]."""
    device = device or next(model.parameters()).device
    model.eval()
    data = data.to(device)
    logits = model(data)
    return torch.sigmoid(logits).cpu()
