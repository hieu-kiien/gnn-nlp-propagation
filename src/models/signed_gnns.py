"""Tier 2 Macro Signed Social Network Link Prediction Models and Graph Parity Utilities.

Implements:
  - Unified message-passing graph construction (`prepare_mp_graph`) supporting
    coalesced and multi-edge parity across undirected and directed signed GNNs.
  - Non-parametric chronological prior baseline (`PriorPairSignPersistence`).
  - Unsigned and signed undirected baselines (`UserTextMLP`, `StandardGCNLink`,
    `StandardGATv2Link`, `SGCNLink` with Layer-0 cross-space residual distortion
    removed, `FAGCNLink`).
  - Bidirectional continuous 2D `[s_uv, h_uv]` attention comparators
    (`BiEdgeGATv2Link`, `SBiFAGATLink` with zeroed/frozen `lin_root`).
  - Directed signed SOTA models:
      * `SiGAT` (Huang et al., ICDM 2019) — 4 directed signed motif channels + channel attention.
      * `SDGNN` (Huang et al., AAAI 2021) — 4 directed signed aggregators + zero-leakage
        status-theory & balance structural auxiliary loss.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import FAConv, GATConv, GATv2Conv, GCNConv, SignedConv
from torch_geometric.utils import coalesce, to_undirected

from src.models.sbi_fagat import StanceGuidedFAConv

LINK_PRED_MODEL_NAMES: tuple[str, ...] = (
    "UserTextMLP",
    "PriorPairSignPersistence",
    "StandardGCN",
    "StandardGATv2",
    "SGCN",
    "FAGCN",
    "BiEdgeGATv2-Link",
    "SiGAT",
    "SDGNN",
    "SBiFAGAT-Link",
)


def prepare_mp_graph(
    edge_index: Tensor,
    edge_attr: Tensor | None,
    num_nodes: int,
    *,
    directed: bool,
    coalesce_edges: bool = True,
) -> tuple[Tensor, Tensor]:
    """Canonicalize message-passing edges and 2D `[s_uv, h_uv]` attributes across GNNs.

    When `coalesce_edges=True` (`"coalesced"` mode), directed multi-edges `(u -> v)`
    are first coalesced by mean, then either kept directed or reduced symmetrically
    via `to_undirected(..., reduce="mean")`. When `coalesce_edges=False` (`"multiedge"`
    mode), multi-edge multiplicities are preserved for both directed and undirected models.
    """
    if edge_index.numel() == 0:
        return (
            edge_index.new_empty((2, 0), dtype=torch.long),
            edge_index.new_zeros((0, 2), dtype=torch.float32),
        )

    ei = edge_index.long()
    num_edges = int(ei.size(1))
    if edge_attr is None or edge_attr.numel() == 0:
        ea = ei.new_zeros((num_edges, 2), dtype=torch.float32)
    else:
        ea = edge_attr.to(device=ei.device, dtype=torch.float32)
        if ea.ndim == 1:
            ea = ea.unsqueeze(-1)
        if ea.size(1) < 2:
            pad = ea.new_zeros((ea.size(0), 2 - ea.size(1)))
            ea = torch.cat([ea, pad], dim=-1)
        elif ea.size(1) > 2:
            ea = ea[:, :2]

    if coalesce_edges:
        ei_dir, ea_dir = coalesce(ei, ea, num_nodes=num_nodes, reduce="mean")
        if directed:
            return ei_dir, ea_dir
        ei_undir, ea_undir = to_undirected(ei_dir, edge_attr=ea_dir, num_nodes=num_nodes, reduce="mean")
        return ei_undir, ea_undir

    if directed:
        return ei, ea
    ei_undir = torch.cat([ei, ei[[1, 0]]], dim=1)
    ea_undir = torch.cat([ea, ea], dim=0)
    return ei_undir, ea_undir


class UserNodeStem(nn.Module):
    """Balanced encoder projecting user historical SpanPool embeddings + 4-D unsigned activity stats."""

    def __init__(self, in_dim: int = 772, hidden_dim: int = 64, beh_dim: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_dim < 8:
            raise ValueError(f"hidden_dim must be >= 8 for UserNodeStem, got {hidden_dim}")
        self.beh_dim = min(int(beh_dim), max(1, int(in_dim) - 1))
        self.sem_dim = max(1, int(in_dim) - self.beh_dim)
        beh_out = max(2, hidden_dim // 4)
        sem_out = hidden_dim - beh_out

        self.sem_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.sem_dim, sem_out),
            nn.GELU(),
            nn.LayerNorm(sem_out),
        )
        self.beh_proj = nn.Sequential(
            nn.Linear(self.beh_dim, beh_out),
            nn.GELU(),
            nn.LayerNorm(beh_out),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        x_f = x.float()
        sem_h = self.sem_proj(x_f[:, : self.sem_dim])
        beh_in = x_f[:, self.sem_dim : self.sem_dim + self.beh_dim]
        if beh_in.size(1) < self.beh_dim:
            pad = beh_in.new_zeros((beh_in.size(0), self.beh_dim - beh_in.size(1)))
            beh_in = torch.cat([beh_in, pad], dim=-1)
        beh_h = self.beh_proj(beh_in)
        return self.out_norm(torch.cat([sem_h, beh_h], dim=-1))


class PairwiseSignedHead(nn.Module):
    """Predict opposing-stance logit P(s_uv < 0) from endpoint user representations."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: Tensor, pairs: Tensor) -> Tensor:
        if pairs.numel() == 0:
            return h.new_zeros((0,))
        hu = h[pairs[0]]
        hv = h[pairs[1]]
        feats = torch.cat([hu, hv, hu * hv, (hu - hv).abs()], dim=-1)
        return self.mlp(feats).squeeze(-1)


class UserTextMLP(nn.Module):
    """Graph-agnostic node-embedding baseline for chronological signed link prediction."""

    def __init__(self, in_dim: int = 772, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.stem = UserNodeStem(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.head = PairwiseSignedHead(hidden_dim, dropout=dropout)

    def encode(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None = None) -> Tensor:
        del edge_index, edge_attr
        h0 = self.stem(x)
        return h0 + self.encoder(h0)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None, pairs: Tensor) -> Tensor:
        return self.head(self.encode(x, edge_index, edge_attr), pairs)


class PriorPairSignPersistence(nn.Module):
    """Non-parametric Bayesian m-estimate chronological Train pair/user sign persistence baseline."""

    def __init__(
        self,
        alpha: float = 2.0,
        user_alpha: float = 2.0,
        undirected_fallback: bool = True,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.user_alpha = float(user_alpha)
        self.undirected_fallback = bool(undirected_fallback)
        self.global_prior: float = 0.5
        self.num_users: int = 0
        self.dir_stats: dict[tuple[int, int], tuple[float, float]] = {}
        self.undir_stats: dict[tuple[int, int], tuple[float, float]] = {}
        self.out_stats: dict[int, tuple[float, float]] = {}
        self.in_stats: dict[int, tuple[float, float]] = {}
        self.all_stats: dict[int, tuple[float, float]] = {}

    def fit(
        self,
        train_pairs: Tensor,
        train_y_oppose: Tensor,
        num_users: int,
    ) -> "PriorPairSignPersistence":
        """Accumulate empirical negative/total counts strictly from chronological Train edges."""
        self.num_users = int(num_users)
        self.dir_stats.clear()
        self.undir_stats.clear()
        self.out_stats.clear()
        self.in_stats.clear()
        self.all_stats.clear()

        if train_pairs.numel() == 0 or train_y_oppose.numel() == 0:
            self.global_prior = 0.5
            return self

        src_arr = train_pairs[0].detach().cpu().tolist()
        dst_arr = train_pairs[1].detach().cpu().tolist()
        y_arr = [float(v) for v in train_y_oppose.detach().cpu().tolist()]

        self.global_prior = (sum(y_arr) + 0.5) / (float(len(y_arr)) + 1.0)

        dir_acc: dict[tuple[int, int], list[float]] = defaultdict(lambda: [0.0, 0.0])
        undir_acc: dict[tuple[int, int], list[float]] = defaultdict(lambda: [0.0, 0.0])
        out_acc: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
        in_acc: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
        all_acc: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])

        for u_raw, v_raw, y in zip(src_arr, dst_arr, y_arr):
            u = int(u_raw)
            v = int(v_raw)
            y_bin = 1.0 if y >= 0.5 else 0.0
            dir_acc[(u, v)][0] += y_bin
            dir_acc[(u, v)][1] += 1.0

            ukey = (u, v) if u <= v else (v, u)
            undir_acc[ukey][0] += y_bin
            undir_acc[ukey][1] += 1.0

            out_acc[u][0] += y_bin
            out_acc[u][1] += 1.0
            in_acc[v][0] += y_bin
            in_acc[v][1] += 1.0

            all_acc[u][0] += y_bin
            all_acc[u][1] += 1.0
            all_acc[v][0] += y_bin
            all_acc[v][1] += 1.0

        self.dir_stats = {k: (v[0], v[1]) for k, v in dir_acc.items()}
        self.undir_stats = {k: (v[0], v[1]) for k, v in undir_acc.items()}
        self.out_stats = {k: (v[0], v[1]) for k, v in out_acc.items()}
        self.in_stats = {k: (v[0], v[1]) for k, v in in_acc.items()}
        self.all_stats = {k: (v[0], v[1]) for k, v in all_acc.items()}
        return self

    def _smoothed_rate(self, stats: tuple[float, float]) -> float:
        neg_cnt, tot_cnt = stats
        return (neg_cnt + self.user_alpha * self.global_prior) / (tot_cnt + self.user_alpha)

    def _user_prior(self, u: int, v: int) -> float:
        out_u = self.out_stats.get(u)
        in_v = self.in_stats.get(v)
        if out_u is not None and in_v is not None and out_u[1] > 0.0 and in_v[1] > 0.0:
            return 0.5 * (self._smoothed_rate(out_u) + self._smoothed_rate(in_v))
        all_u = self.all_stats.get(u)
        all_v = self.all_stats.get(v)
        if all_u is not None and all_v is not None and all_u[1] > 0.0 and all_v[1] > 0.0:
            return 0.5 * (self._smoothed_rate(all_u) + self._smoothed_rate(all_v))
        if all_u is not None and all_u[1] > 0.0:
            return self._smoothed_rate(all_u)
        if all_v is not None and all_v[1] > 0.0:
            return self._smoothed_rate(all_v)
        return self.global_prior

    def predict_proba(self, pairs: Tensor) -> Tensor:
        if pairs.numel() == 0:
            return torch.empty((0,), dtype=torch.float32, device=pairs.device)
        src_arr = pairs[0].detach().cpu().tolist()
        dst_arr = pairs[1].detach().cpu().tolist()
        probs: list[float] = []
        for u_raw, v_raw in zip(src_arr, dst_arr):
            u = int(u_raw)
            v = int(v_raw)
            pair_stat = self.dir_stats.get((u, v))
            if pair_stat is None and self.undirected_fallback:
                ukey = (u, v) if u <= v else (v, u)
                pair_stat = self.undir_stats.get(ukey)
            neg_cnt, tot_cnt = pair_stat if pair_stat is not None else (0.0, 0.0)
            prior = self._user_prior(u, v)
            p = (neg_cnt + self.alpha * prior) / (tot_cnt + self.alpha)
            probs.append(min(1.0 - 1e-6, max(1e-6, float(p))))
        return torch.tensor(probs, dtype=torch.float32, device=pairs.device)

    def predict_logits(self, pairs: Tensor) -> Tensor:
        p = self.predict_proba(pairs)
        return torch.log(p / (1.0 - p))

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None,
        pairs: Tensor,
    ) -> Tensor:
        del x, edge_index, edge_attr
        return self.predict_logits(pairs)


class StandardGCNLink(nn.Module):
    """Unsigned undirected GCN encoder (Kipf & Welling, ICLR 2017) for link sign prediction."""

    def __init__(
        self,
        in_dim: int = 772,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        coalesce_edges: bool = True,
    ) -> None:
        super().__init__()
        self.coalesce_edges = bool(coalesce_edges)
        self.stem = UserNodeStem(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.drop = nn.Dropout(dropout)
        self.head = PairwiseSignedHead(hidden_dim, dropout=dropout)

    def encode(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None = None) -> Tensor:
        ei, _ = prepare_mp_graph(
            edge_index,
            edge_attr,
            x.size(0),
            directed=False,
            coalesce_edges=self.coalesce_edges,
        )
        h = self.stem(x)
        for conv, norm in zip(self.convs, self.norms):
            h = h + self.drop(norm(F.relu(conv(h, ei))))
        return h

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None, pairs: Tensor) -> Tensor:
        return self.head(self.encode(x, edge_index, edge_attr), pairs)


class StandardGATv2Link(nn.Module):
    """Unsigned undirected GATv2 encoder (Brody et al., ICLR 2022) for link sign prediction."""

    def __init__(
        self,
        in_dim: int = 772,
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        coalesce_edges: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})")
        self.coalesce_edges = bool(coalesce_edges)
        self.stem = UserNodeStem(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
        head_dim = hidden_dim // heads
        self.convs = nn.ModuleList(
            [
                GATv2Conv(hidden_dim, head_dim, heads=heads, concat=True, dropout=dropout, add_self_loops=True)
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.drop = nn.Dropout(dropout)
        self.head = PairwiseSignedHead(hidden_dim, dropout=dropout)

    def encode(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None = None) -> Tensor:
        ei, _ = prepare_mp_graph(
            edge_index,
            edge_attr,
            x.size(0),
            directed=False,
            coalesce_edges=self.coalesce_edges,
        )
        h = self.stem(x)
        for conv, norm in zip(self.convs, self.norms):
            h = h + self.drop(norm(F.elu(conv(h, ei))))
        return h

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None, pairs: Tensor) -> Tensor:
        return self.head(self.encode(x, edge_index, edge_attr), pairs)


class SGCNLink(nn.Module):
    """Signed Graph Convolutional Network (Derr et al., ICDM 2018) with clean Layer-0 projection."""

    def __init__(
        self,
        in_dim: int = 772,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        coalesce_edges: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim % 2 != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be even for SGCN")
        self.coalesce_edges = bool(coalesce_edges)
        half = hidden_dim // 2
        self.stem = UserNodeStem(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
        convs = [SignedConv(hidden_dim, half, first_aggr=True)]
        for _ in range(max(0, num_layers - 1)):
            convs.append(SignedConv(half, half, first_aggr=False))
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(len(convs))])
        self.drop = nn.Dropout(dropout)
        self.head = PairwiseSignedHead(hidden_dim, dropout=dropout)

    def encode(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None = None) -> Tensor:
        num_nodes = x.size(0)
        ei_dir, ea_dir = prepare_mp_graph(
            edge_index,
            edge_attr,
            num_nodes,
            directed=True,
            coalesce_edges=self.coalesce_edges,
        )
        if ei_dir.numel() > 0:
            signs = ea_dir[:, 0]
            pos_dir = ei_dir[:, signs >= 0.0]
            neg_dir = ei_dir[:, signs < 0.0]
            if self.coalesce_edges:
                pos_ei = to_undirected(pos_dir, num_nodes=num_nodes) if pos_dir.numel() > 0 else ei_dir.new_empty((2, 0))
                neg_ei = to_undirected(neg_dir, num_nodes=num_nodes) if neg_dir.numel() > 0 else ei_dir.new_empty((2, 0))
            else:
                pos_ei = (
                    torch.cat([pos_dir, pos_dir[[1, 0]]], dim=1)
                    if pos_dir.numel() > 0
                    else ei_dir.new_empty((2, 0))
                )
                neg_ei = (
                    torch.cat([neg_dir, neg_dir[[1, 0]]], dim=1)
                    if neg_dir.numel() > 0
                    else ei_dir.new_empty((2, 0))
                )
        else:
            pos_ei = ei_dir.new_empty((2, 0))
            neg_ei = ei_dir.new_empty((2, 0))

        # Layer 0 projects [sem || beh] cleanly into [balanced || unbalanced] with NO cross-space skip;
        # Layers l >= 1 apply residual skips strictly within [balanced || unbalanced].
        h = self.stem(x)
        for l_idx, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            out = self.drop(norm(torch.tanh(conv(h, pos_ei, neg_ei))))
            h = out if l_idx == 0 else (h + out)
        return h

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None, pairs: Tensor) -> Tensor:
        return self.head(self.encode(x, edge_index, edge_attr), pairs)


class FAGCNLink(nn.Module):
    """Frequency Adaptation GCN (Bo et al., AAAI 2021) for signed link prediction."""

    def __init__(
        self,
        in_dim: int = 772,
        hidden_dim: int = 64,
        num_layers: int = 2,
        eps: float = 0.5,
        dropout: float = 0.1,
        coalesce_edges: bool = True,
    ) -> None:
        super().__init__()
        self.coalesce_edges = bool(coalesce_edges)
        self.stem = UserNodeStem(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.convs = nn.ModuleList(
            [FAConv(channels=hidden_dim, eps=eps, dropout=dropout, add_self_loops=True) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.drop = nn.Dropout(dropout)
        self.head = PairwiseSignedHead(hidden_dim, dropout=dropout)

    def encode(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None = None) -> Tensor:
        ei, _ = prepare_mp_graph(
            edge_index,
            edge_attr,
            x.size(0),
            directed=False,
            coalesce_edges=self.coalesce_edges,
        )
        h0 = self.stem(x)
        h = h0
        for conv, norm in zip(self.convs, self.norms):
            h = h + self.drop(norm(F.elu(conv(h, h0, ei))))
        return h

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None, pairs: Tensor) -> Tensor:
        return self.head(self.encode(x, edge_index, edge_attr), pairs)


class BiEdgeGATv2Link(nn.Module):
    """Bidirectional (TD+BU) GATv2 comparator with continuous 2D `[s_uv, h_uv]` edge features."""

    def __init__(
        self,
        in_dim: int = 772,
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        edge_dim: int = 2,
        eps: float = 0.5,
        dropout: float = 0.1,
        coalesce_edges: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})")
        self.coalesce_edges = bool(coalesce_edges)
        self.eps = float(eps)
        self.stem = UserNodeStem(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
        head_dim = hidden_dim // heads

        self.td_convs = nn.ModuleList(
            [
                GATv2Conv(
                    hidden_dim,
                    head_dim,
                    heads=heads,
                    concat=True,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    add_self_loops=True,
                    fill_value=0.0,
                )
                for _ in range(num_layers)
            ]
        )
        self.bu_convs = nn.ModuleList(
            [
                GATv2Conv(
                    hidden_dim,
                    head_dim,
                    heads=heads,
                    concat=True,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    add_self_loops=True,
                    fill_value=0.0,
                )
                for _ in range(num_layers)
            ]
        )
        self.td_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.bu_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.drop = nn.Dropout(dropout)
        self.head = PairwiseSignedHead(hidden_dim, dropout=dropout)

    def encode(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None = None) -> Tensor:
        ei_td, ea_td = prepare_mp_graph(
            edge_index,
            edge_attr,
            x.size(0),
            directed=True,
            coalesce_edges=self.coalesce_edges,
        )
        ei_bu = ei_td[[1, 0]] if ei_td.numel() > 0 else ei_td
        h0 = self.stem(x)
        h_td = h0
        h_bu = h0
        for td_conv, bu_conv, td_norm, bu_norm in zip(
            self.td_convs, self.bu_convs, self.td_norms, self.bu_norms
        ):
            td_msg = F.elu(td_conv(h_td, ei_td, edge_attr=ea_td))
            bu_msg = F.elu(bu_conv(h_bu, ei_bu, edge_attr=ea_td))
            h_td = self.drop(td_norm(h_td + td_msg + self.eps * h0))
            h_bu = self.drop(bu_norm(h_bu + bu_msg + self.eps * h0))
        return 0.5 * (h_td + h_bu)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None, pairs: Tensor) -> Tensor:
        return self.head(self.encode(x, edge_index, edge_attr), pairs)


def _split_directed_signed_channels(
    ei_dir: Tensor,
    ea_dir: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return `(pos_out, neg_out, pos_in, neg_in)` directed edge tensors for PyG `source_to_target` flow."""
    if ei_dir.numel() == 0:
        empty = ei_dir.new_empty((2, 0))
        return empty, empty, empty, empty
    signs = ea_dir[:, 0]
    pos_dir = ei_dir[:, signs >= 0.0]
    neg_dir = ei_dir[:, signs < 0.0]
    pos_in = pos_dir
    neg_in = neg_dir
    pos_out = pos_dir[[1, 0]] if pos_dir.numel() > 0 else pos_dir
    neg_out = neg_dir[[1, 0]] if neg_dir.numel() > 0 else neg_dir
    return pos_out, neg_out, pos_in, neg_in


class _SiGATLayer(nn.Module):
    """Single SiGAT motif-channel attention layer (+out, -out, +in, -in)."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.branch_dim = max(4, hidden_dim // 4)
        branch_heads = min(2, max(1, self.branch_dim // 8))
        while self.branch_dim % branch_heads != 0 and branch_heads > 1:
            branch_heads -= 1
        head_dim = self.branch_dim // branch_heads

        self.conv_pos_out = GATConv(
            hidden_dim, head_dim, heads=branch_heads, concat=True, dropout=dropout, add_self_loops=False
        )
        self.conv_neg_out = GATConv(
            hidden_dim, head_dim, heads=branch_heads, concat=True, dropout=dropout, add_self_loops=False
        )
        self.conv_pos_in = GATConv(
            hidden_dim, head_dim, heads=branch_heads, concat=True, dropout=dropout, add_self_loops=False
        )
        self.conv_neg_in = GATConv(
            hidden_dim, head_dim, heads=branch_heads, concat=True, dropout=dropout, add_self_loops=False
        )
        self.attn_proj = nn.Linear(self.branch_dim, self.branch_dim)
        self.attn_vec = nn.Linear(self.branch_dim, 1, bias=False)
        self.self_lin = nn.Linear(hidden_dim, hidden_dim)
        self.fusion_lin = nn.Linear(hidden_dim + 4 * self.branch_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def _apply_channel(self, conv: nn.Module, h: Tensor, ei: Tensor, num_nodes: int) -> Tensor:
        if ei.numel() == 0:
            return h.new_zeros((num_nodes, self.branch_dim))
        return conv(h, ei)

    def forward(
        self,
        h: Tensor,
        pos_out: Tensor,
        neg_out: Tensor,
        pos_in: Tensor,
        neg_in: Tensor,
    ) -> Tensor:
        n = h.size(0)
        m_po = self._apply_channel(self.conv_pos_out, h, pos_out, n)
        m_no = self._apply_channel(self.conv_neg_out, h, neg_out, n)
        m_pi = self._apply_channel(self.conv_pos_in, h, pos_in, n)
        m_ni = self._apply_channel(self.conv_neg_in, h, neg_in, n)

        stacked = torch.stack([m_po, m_no, m_pi, m_ni], dim=1)  # [N, 4, branch_dim]
        scores = self.attn_vec(torch.tanh(self.attn_proj(stacked)))  # [N, 4, 1]
        alpha_c = torch.softmax(scores, dim=1)
        attended = (alpha_c * 4.0 * stacked).reshape(n, 4 * self.branch_dim)
        combined = torch.cat([self.self_lin(h), attended], dim=-1)
        return self.drop(self.norm(torch.tanh(self.fusion_lin(combined))))


class SiGAT(nn.Module):
    """Signed Graph Attention Network (Huang et al., ICDM 2019) for directed signed links."""

    def __init__(
        self,
        in_dim: int = 772,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        coalesce_edges: bool = True,
    ) -> None:
        super().__init__()
        self.coalesce_edges = bool(coalesce_edges)
        self.stem = UserNodeStem(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.layers = nn.ModuleList([_SiGATLayer(hidden_dim, dropout=dropout) for _ in range(num_layers)])
        self.head = PairwiseSignedHead(hidden_dim, dropout=dropout)

    def encode(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None = None) -> Tensor:
        ei_dir, ea_dir = prepare_mp_graph(
            edge_index,
            edge_attr,
            x.size(0),
            directed=True,
            coalesce_edges=self.coalesce_edges,
        )
        pos_out, neg_out, pos_in, neg_in = _split_directed_signed_channels(ei_dir, ea_dir)
        h = self.stem(x)
        for l_idx, layer in enumerate(self.layers):
            out = layer(h, pos_out, neg_out, pos_in, neg_in)
            h = out if l_idx == 0 else (h + out)
        return h

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None, pairs: Tensor) -> Tensor:
        return self.head(self.encode(x, edge_index, edge_attr), pairs)


class _SDGNNLayer(nn.Module):
    """Single SDGNN directed signed aggregator layer (+out, -out, +in, -in)."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.branch_dim = max(4, hidden_dim // 4)
        branch_heads = min(2, max(1, self.branch_dim // 8))
        while self.branch_dim % branch_heads != 0 and branch_heads > 1:
            branch_heads -= 1
        head_dim = self.branch_dim // branch_heads

        self.conv_pos_out = GATv2Conv(
            hidden_dim, head_dim, heads=branch_heads, concat=True, dropout=dropout, add_self_loops=False
        )
        self.conv_neg_out = GATv2Conv(
            hidden_dim, head_dim, heads=branch_heads, concat=True, dropout=dropout, add_self_loops=False
        )
        self.conv_pos_in = GATv2Conv(
            hidden_dim, head_dim, heads=branch_heads, concat=True, dropout=dropout, add_self_loops=False
        )
        self.conv_neg_in = GATv2Conv(
            hidden_dim, head_dim, heads=branch_heads, concat=True, dropout=dropout, add_self_loops=False
        )
        self.self_lin = nn.Linear(hidden_dim, hidden_dim)
        self.fusion_lin = nn.Linear(hidden_dim + 4 * self.branch_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def _apply_channel(self, conv: nn.Module, h: Tensor, ei: Tensor, num_nodes: int) -> Tensor:
        if ei.numel() == 0:
            return h.new_zeros((num_nodes, self.branch_dim))
        return conv(h, ei)

    def forward(
        self,
        h: Tensor,
        pos_out: Tensor,
        neg_out: Tensor,
        pos_in: Tensor,
        neg_in: Tensor,
    ) -> Tensor:
        n = h.size(0)
        m_po = self._apply_channel(self.conv_pos_out, h, pos_out, n)
        m_no = self._apply_channel(self.conv_neg_out, h, neg_out, n)
        m_pi = self._apply_channel(self.conv_pos_in, h, pos_in, n)
        m_ni = self._apply_channel(self.conv_neg_in, h, neg_in, n)
        combined = torch.cat([self.self_lin(h), m_po, m_no, m_pi, m_ni], dim=-1)
        return self.drop(self.norm(F.gelu(self.fusion_lin(combined))))


class SDGNN(nn.Module):
    """Signed Directed GNN (Huang et al., AAAI 2021) with zero-leakage status/balance aux loss."""

    def __init__(
        self,
        in_dim: int = 772,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        coalesce_edges: bool = True,
    ) -> None:
        super().__init__()
        self.coalesce_edges = bool(coalesce_edges)
        self.stem = UserNodeStem(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.layers = nn.ModuleList([_SDGNNLayer(hidden_dim, dropout=dropout) for _ in range(num_layers)])
        self.status_head = nn.Linear(hidden_dim, 1)
        self.head = PairwiseSignedHead(hidden_dim, dropout=dropout)

    def encode(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None = None) -> Tensor:
        ei_dir, ea_dir = prepare_mp_graph(
            edge_index,
            edge_attr,
            x.size(0),
            directed=True,
            coalesce_edges=self.coalesce_edges,
        )
        pos_out, neg_out, pos_in, neg_in = _split_directed_signed_channels(ei_dir, ea_dir)
        h = self.stem(x)
        for layer in self.layers:
            h = h + layer(h, pos_out, neg_out, pos_in, neg_in)
        return h

    def compute_aux_loss(
        self,
        h: Tensor,
        mp_edge_index: Tensor,
        mp_edge_attr: Tensor | None,
        *,
        max_samples: int = 2048,
    ) -> Tensor:
        """Compute status-theory & balance-alignment auxiliary loss strictly on masked `mp_edge_index`."""
        if mp_edge_index.numel() == 0 or mp_edge_attr is None or mp_edge_attr.numel() == 0:
            return h.sum() * 0.0

        ei_dir, ea_dir = prepare_mp_graph(
            mp_edge_index,
            mp_edge_attr,
            h.size(0),
            directed=True,
            coalesce_edges=self.coalesce_edges,
        )
        if ei_dir.numel() == 0:
            return h.sum() * 0.0

        if ei_dir.size(1) > max_samples:
            perm = torch.randperm(ei_dir.size(1), device=ei_dir.device)[:max_samples]
            ei_dir = ei_dir[:, perm]
            ea_dir = ea_dir[perm]

        src, dst = ei_dir[0], ei_dir[1]
        s_uv = ea_dir[:, 0]
        y_pos = (s_uv >= 0.0).float()

        status_scores = self.status_head(h).squeeze(-1)
        loss_status = F.binary_cross_entropy_with_logits(status_scores[dst] - status_scores[src], y_pos)

        h_norm = F.normalize(h, p=2, dim=-1, eps=1e-6)
        cos_uv = (h_norm[src] * h_norm[dst]).sum(dim=-1)
        target_sign = torch.where(s_uv >= 0.0, 1.0, -1.0)
        loss_bal = F.mse_loss(cos_uv, target_sign)
        return 0.5 * loss_status + 0.5 * loss_bal

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None, pairs: Tensor) -> Tensor:
        return self.head(self.encode(x, edge_index, edge_attr), pairs)


class SBiFAGATLink(nn.Module):
    """Bidirectional Stance-Guided Frequency-Adaptive GAT for Macro Signed Link Prediction."""

    def __init__(
        self,
        in_dim: int = 772,
        hidden_dim: int = 64,
        num_layers: int = 2,
        eps: float = 0.5,
        dropout: float = 0.1,
        coalesce_edges: bool = True,
    ) -> None:
        super().__init__()
        self.coalesce_edges = bool(coalesce_edges)
        self.stem = UserNodeStem(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.convs_td = nn.ModuleList(
            [
                StanceGuidedFAConv(
                    hidden_dim=hidden_dim,
                    edge_dim=2,
                    eps=eps,
                    gate_mode="tanh",
                    dropout=dropout,
                    use_dual_channel=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.convs_bu = nn.ModuleList(
            [
                StanceGuidedFAConv(
                    hidden_dim=hidden_dim,
                    edge_dim=2,
                    eps=eps,
                    gate_mode="tanh",
                    dropout=dropout,
                    use_dual_channel=True,
                )
                for _ in range(num_layers)
            ]
        )
        # User-user macro graphs have no cascade root node; zero & freeze lin_root so
        # StanceGuidedFAConv receives exact zero root-broadcast without modifying sbi_fagat.py.
        for conv in (*self.convs_td, *self.convs_bu):
            conv.lin_root.weight.requires_grad_(False)
            conv.lin_root.bias.requires_grad_(False)
            with torch.no_grad():
                conv.lin_root.weight.zero_()
                conv.lin_root.bias.zero_()

        self.mix_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in range(num_layers)
            ]
        )
        self.head = PairwiseSignedHead(hidden_dim, dropout=dropout)

    def encode(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None = None) -> Tensor:
        ei_td, ea_td = prepare_mp_graph(
            edge_index,
            edge_attr,
            x.size(0),
            directed=True,
            coalesce_edges=self.coalesce_edges,
        )
        ei_bu = ei_td[[1, 0]] if ei_td.numel() > 0 else ei_td
        h0 = self.stem(x)
        zero_root = h0.new_zeros(h0.shape)
        h = h0
        for conv_td, conv_bu, mix in zip(self.convs_td, self.convs_bu, self.mix_projs):
            h_td, _ = conv_td(h, h0, zero_root, ei_td, ea_td)
            h_bu, _ = conv_bu(h, h0, zero_root, ei_bu, ea_td)
            h = h + mix(torch.cat([h_td, h_bu], dim=-1))
        return h

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None, pairs: Tensor) -> Tensor:
        return self.head(self.encode(x, edge_index, edge_attr), pairs)


def _resolve_attention_heads(hidden_dim: int, max_heads: int = 4) -> int:
    candidate = min(max_heads, max(1, hidden_dim // 16))
    while candidate > 1 and hidden_dim % candidate != 0:
        candidate -= 1
    return candidate


def build_link_model(
    model_name: str,
    *,
    in_dim: int,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    coalesce_edges: bool = True,
) -> nn.Module:
    """Instantiate a signed link prediction architecture by canonical name."""
    if model_name == "UserTextMLP":
        return UserTextMLP(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
    if model_name == "PriorPairSignPersistence":
        return PriorPairSignPersistence()
    if model_name == "StandardGCN":
        return StandardGCNLink(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            coalesce_edges=coalesce_edges,
        )
    if model_name == "StandardGATv2":
        return StandardGATv2Link(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=_resolve_attention_heads(hidden_dim, 4),
            dropout=dropout,
            coalesce_edges=coalesce_edges,
        )
    if model_name == "SGCN":
        return SGCNLink(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            coalesce_edges=coalesce_edges,
        )
    if model_name == "FAGCN":
        return FAGCNLink(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            coalesce_edges=coalesce_edges,
        )
    if model_name == "BiEdgeGATv2-Link":
        return BiEdgeGATv2Link(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=_resolve_attention_heads(hidden_dim, 4),
            dropout=dropout,
            coalesce_edges=coalesce_edges,
        )
    if model_name == "SiGAT":
        return SiGAT(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            coalesce_edges=coalesce_edges,
        )
    if model_name == "SDGNN":
        return SDGNN(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            coalesce_edges=coalesce_edges,
        )
    if model_name == "SBiFAGAT-Link":
        return SBiFAGATLink(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            coalesce_edges=coalesce_edges,
        )
    raise ValueError(f"Unknown link prediction model '{model_name}'")


__all__ = [
    "BiEdgeGATv2Link",
    "FAGCNLink",
    "LINK_PRED_MODEL_NAMES",
    "PairwiseSignedHead",
    "PriorPairSignPersistence",
    "SBiFAGATLink",
    "SDGNN",
    "SGCNLink",
    "SiGAT",
    "StandardGATv2Link",
    "StandardGCNLink",
    "UserNodeStem",
    "UserTextMLP",
    "build_link_model",
    "prepare_mp_graph",
]
