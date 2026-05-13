"""DQN training loop for MIS, following Dai et al. NeurIPS 2017 (S2V-DQN).

Components:
  - episode rollout with epsilon-greedy action selection on a single graph
  - n-step return computation
  - experience replay buffer (graph_id, tag, action, n_step_return, next_tag, done)
  - target network with periodic hard update
  - SmoothL1 (Huber) loss on the Bellman error

Reward: +1 per node addition. gamma defaults to 1.0 since the return then
exactly equals the IS size — we want the policy to maximize |IS|, not a
geometrically-weighted sum.
"""
from __future__ import annotations
import copy
import random
import time
from collections import deque, namedtuple
from dataclasses import dataclass, field
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from . import dqn as dqnmod


Transition = namedtuple("Transition",
                        ["graph_idx", "tag", "action", "n_step_reward",
                         "next_tag", "n_steps_to_terminal"])


# ---- replay buffer ---------------------------------------------------------

class ReplayBuffer:
    """Bounded ring buffer of Transitions plus a parallel list of the graphs
    they came from (so we can re-look-up edge_index/x at sample time). Graphs
    are kept on CPU and moved to the device per-batch."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buf: deque[Transition] = deque(maxlen=capacity)

    def push(self, t: Transition) -> None:
        self.buf.append(t)

    def sample(self, batch_size: int, rng: random.Random) -> list[Transition]:
        return rng.sample(self.buf, min(batch_size, len(self.buf)))

    def __len__(self) -> int:
        return len(self.buf)


# ---- episode rollout -------------------------------------------------------

@torch.no_grad()
def rollout_episode(model: nn.Module, data: Data, epsilon: float,
                     device: str, rng: random.Random
                     ) -> tuple[list[int], list[torch.Tensor], list[float]]:
    """Run one greedy/eps-greedy episode on `data`, return:
        actions     : list[int]              the order of selected nodes
        tags        : list[Tensor [n, 2]]    pre-action tag at each step
                                              (tags[i] is BEFORE actions[i])
        rewards     : list[float]            +1 per added node

    Episode stops when no node is available.
    """
    model.eval()
    data = data.to(device)
    tag = dqnmod.initial_tag(int(data.num_nodes), device=device)
    actions, tags, rewards = [], [], []

    while True:
        avail = dqnmod.available_mask(tag)
        if not avail.any():
            break

        if rng.random() < epsilon:
            # uniform random over available
            avail_idx = avail.nonzero(as_tuple=True)[0]
            v = int(avail_idx[rng.randrange(avail_idx.numel())].item())
        else:
            q = model(data, tag)
            q = q.masked_fill(~avail, float("-inf"))
            v = int(q.argmax().item())

        tags.append(tag.cpu().clone())
        actions.append(v)
        rewards.append(1.0)
        tag = dqnmod.apply_action(tag, data.edge_index, v)

    return actions, tags, rewards


def store_episode(buf: ReplayBuffer, graph_idx: int, actions: list[int],
                   tags: list[torch.Tensor], rewards: list[float],
                   n_step: int) -> None:
    """Convert one episode trajectory into n-step transitions, push to buffer.

    n_step_reward[t] = sum_{i=0}^{n-1} r_{t+i}, with the cap at terminal.
    next_tag[t] = tag at step t+n if it exists, else None (encoded by
    n_steps_to_terminal field; if it equals (T - t), the transition was
    terminal within the n-window and the bootstrap term is omitted).
    """
    T = len(actions)
    for t in range(T):
        end = min(t + n_step, T)
        nsr = sum(rewards[t:end])
        next_tag = tags[end] if end < T else None
        # If next_tag is None we don't have a bootstrap state — store a
        # zeros placeholder; the loss code branches on n_steps_to_terminal.
        if next_tag is None:
            placeholder = torch.zeros_like(tags[t])
            buf.push(Transition(graph_idx=graph_idx, tag=tags[t],
                                action=actions[t], n_step_reward=nsr,
                                next_tag=placeholder,
                                n_steps_to_terminal=end - t))
        else:
            buf.push(Transition(graph_idx=graph_idx, tag=tags[t],
                                action=actions[t], n_step_reward=nsr,
                                next_tag=next_tag,
                                n_steps_to_terminal=n_step))


# ---- loss / update ---------------------------------------------------------

def compute_loss(model: nn.Module, target: nn.Module,
                 batch: list[Transition], graphs: list[Data],
                 gamma: float, n_step: int, device: str) -> torch.Tensor:
    """Per-transition Bellman target with the target network for bootstrap.

    For non-terminal transitions:
        y = sum_{i=0..n-1} gamma^i * r_{t+i}  +  gamma^n * max_v Q_target(s_{t+n}, v)
    For transitions that ended within the n-window:
        y = sum_{i=0..k-1} gamma^i * r_{t+i}    (no bootstrap)

    With gamma=1.0 (the default), this collapses to the path return.
    """
    losses = []
    target.eval()
    for tr in batch:
        data = graphs[tr.graph_idx].to(device)
        tag = tr.tag.to(device)
        next_tag = tr.next_tag.to(device)

        q_pred = model(data, tag)[tr.action]

        is_terminal = (tr.n_steps_to_terminal < n_step) or \
                      (not dqnmod.available_mask(next_tag).any().item())
        if is_terminal:
            y = torch.tensor(tr.n_step_reward, device=device, dtype=q_pred.dtype)
        else:
            with torch.no_grad():
                q_next = target(data, next_tag)
                avail = dqnmod.available_mask(next_tag)
                q_next = q_next.masked_fill(~avail, float("-inf"))
                bootstrap = q_next.max()
            y = tr.n_step_reward + (gamma ** n_step) * bootstrap

        losses.append(F.smooth_l1_loss(q_pred, y))
    return torch.stack(losses).mean()


# ---- training driver -------------------------------------------------------

@dataclass
class DqnConfig:
    episodes: int = 5000           # total training episodes (graph rollouts)
    batch_size: int = 64           # transitions per gradient update
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    gamma: float = 1.0
    n_step: int = 4
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_episodes: int = 2000  # linear decay over this many episodes
    replay_capacity: int = 50_000
    warmup_transitions: int = 2_000  # fill buffer before training starts
    target_update_episodes: int = 100
    updates_per_episode: int = 4
    log_every: int = 100
    eval_every: int = 500
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class DqnHistory:
    episode_size: list = field(default_factory=list)     # IS size per episode
    episode_reward: list = field(default_factory=list)   # = size
    losses: list = field(default_factory=list)            # mean loss per episode
    eval_size: list = field(default_factory=list)         # eval IS at checkpoints
    eval_at: list = field(default_factory=list)           # episode # of each eval


def epsilon_at(ep: int, cfg: DqnConfig) -> float:
    if ep >= cfg.eps_decay_episodes:
        return cfg.eps_end
    frac = ep / max(1, cfg.eps_decay_episodes)
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


def train_dqn(model: nn.Module, train_graphs: list[Data],
              cfg: DqnConfig | None = None,
              val_graphs: list[Data] | None = None,
              checkpoint_path: str | None = None) -> DqnHistory:
    cfg = cfg or DqnConfig()
    model = model.to(cfg.device)
    target = copy.deepcopy(model).to(cfg.device)
    target.eval()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                             weight_decay=cfg.weight_decay)
    rng = random.Random(0)
    buf = ReplayBuffer(cfg.replay_capacity)
    history = DqnHistory()
    best_eval = -1.0

    print(f"[dqn] device={cfg.device} train_graphs={len(train_graphs)} "
          f"episodes={cfg.episodes} n_step={cfg.n_step} gamma={cfg.gamma}")

    for ep in range(1, cfg.episodes + 1):
        # 1. rollout
        eps = epsilon_at(ep, cfg)
        graph_idx = rng.randrange(len(train_graphs))
        actions, tags, rewards = rollout_episode(
            model, train_graphs[graph_idx], eps, cfg.device, rng)
        store_episode(buf, graph_idx, actions, tags, rewards, cfg.n_step)

        is_size = len(actions)
        history.episode_size.append(is_size)
        history.episode_reward.append(sum(rewards))

        # 2. learning updates (only after warmup)
        ep_losses = []
        if len(buf) >= cfg.warmup_transitions:
            model.train()
            for _ in range(cfg.updates_per_episode):
                batch = buf.sample(cfg.batch_size, rng)
                loss = compute_loss(model, target, batch, train_graphs,
                                     cfg.gamma, cfg.n_step, cfg.device)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                ep_losses.append(float(loss.detach()))
        history.losses.append(sum(ep_losses) / max(1, len(ep_losses)))

        # 3. periodic target update
        if ep % cfg.target_update_episodes == 0:
            target.load_state_dict(model.state_dict())

        # 4. logging
        if ep % cfg.log_every == 0 or ep == cfg.episodes:
            recent = history.episode_size[-cfg.log_every:]
            print(f"[dqn] ep {ep:5d} | eps {eps:.3f} | size {sum(recent)/len(recent):6.2f} "
                  f"(min {min(recent)} max {max(recent)}) | "
                  f"loss {history.losses[-1]:.4f} | buf {len(buf):5d}")

        # 5. validation: greedy decode, no exploration
        if val_graphs is not None and (ep % cfg.eval_every == 0 or ep == cfg.episodes):
            sizes = []
            for d in val_graphs:
                acts, _, _ = rollout_episode(model, d, 0.0, cfg.device, rng)
                sizes.append(len(acts))
            mean_size = sum(sizes) / max(1, len(sizes))
            history.eval_size.append(mean_size)
            history.eval_at.append(ep)
            print(f"[dqn] ep {ep:5d} | val mean IS {mean_size:.3f}")
            if checkpoint_path is not None and mean_size > best_eval:
                best_eval = mean_size
                torch.save(model.state_dict(), checkpoint_path)
                print(f"[dqn] saved best (val IS {mean_size:.3f}) to {checkpoint_path}")

    return history


# ---- inference -------------------------------------------------------------

@torch.no_grad()
def dqn_decode(model: nn.Module, data: Data, device: str | None = None
                ) -> list[int]:
    """Greedy IS construction by repeatedly picking argmax Q over available
    nodes. Equivalent to rollout_episode with epsilon=0."""
    device = device or next(model.parameters()).device
    model.eval()
    data = data.to(device)
    tag = dqnmod.initial_tag(int(data.num_nodes), device=device)
    chosen = []
    while True:
        avail = dqnmod.available_mask(tag)
        if not avail.any():
            break
        q = model(data, tag).masked_fill(~avail, float("-inf"))
        v = int(q.argmax().item())
        chosen.append(v)
        tag = dqnmod.apply_action(tag, data.edge_index, v)
    return sorted(chosen)
