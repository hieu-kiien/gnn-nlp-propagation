"""Baseline models and shared multi-task components for SBi-FAGAT evaluation.

Implements standardized baselines sharing `hidden_dim = 64` and identical
multi-task prediction heads (`y_size`, `y_polar`, `rumor_logits`):
  1. `TextMLP`: Mean-pooled NLP embeddings + MLP (graph-agnostic baseline).
  2. `TemporalStanceMLP`: Root + mean-pooled node/temporal MLP fused with pooled 2D `[s_uv, h_uv]` stats (no GNN).
  3. `StandardGCN`: Undirected 2-layer GCN (Kipf & Welling, ICLR 2017).
  4. `StandardGATv2`: Edge-sign-free dynamic GATv2 (Brody et al., ICLR 2022).
  5. `EdgeGATv2`: Edge-aware GATv2 consuming 2D `[s_uv, h_uv]` (or latent edge MLP).
  6. `FAGCNBaseline`: Frequency-adaptive signed GCN using PyG `FAConv` (Bo et al., AAAI 2021).
  7. `BiGCN`: Top-Down + Bottom-Up GCN with DropEdge and Root Feature Enhancement (Bian et al., AAAI 2020).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import FAConv, GATv2Conv, GCNConv, global_mean_pool
from torch_geometric.utils import degree, dropout_edge, scatter, softmax, to_undirected


def resolve_root_indices(data: Any, num_nodes: int) -> tuple[Tensor, Tensor]:
    """Resolve per-graph root node indices and batch assignment vector safely."""
    device = data.x.device
    if hasattr(data, "batch") and data.batch is not None:
        batch_vec = data.batch.long()
    else:
        batch_vec = torch.zeros(num_nodes, dtype=torch.long, device=device)

    if hasattr(data, "root_index") and data.root_index is not None:
        root_idx = data.root_index.view(-1).long().to(device)
    elif hasattr(data, "ptr") and data.ptr is not None:
        root_idx = data.ptr[:-1].long().to(device)
    else:
        num_graphs = int(batch_vec.max().item()) + 1 if batch_vec.numel() > 0 else 1
        roots = []
        for g in range(num_graphs):
            matches = (batch_vec == g).nonzero(as_tuple=False)
            roots.append(int(matches[0].item()) if matches.numel() > 0 else 0)
        root_idx = torch.tensor(roots, dtype=torch.long, device=device)

    return root_idx, batch_vec


def broadcast_root_features(h: Tensor, root_idx: Tensor, batch_vec: Tensor) -> Tensor:
    """Broadcast each graph's root feature vector to all nodes in that graph."""
    return h[root_idx][batch_vec]


class TwoStreamNodeStem(nn.Module):
    """Modality-balanced encoder projecting semantic embeddings separately from causal node metadata."""

    def __init__(
        self,
        in_dim: int = 770,
        hidden_dim: int = 64,
        struct_dim: int = 10,
        dropout: float = 0.1,
        mask_topology: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim < 4:
            raise ValueError(f"hidden_dim must be >= 4 for TwoStreamNodeStem, got {hidden_dim}")
        struct_out = max(2, hidden_dim // 4)
        sem_out = hidden_dim - struct_out
        self.sem_in_dim = 768 if int(in_dim) == 770 else int(in_dim)
        self.struct_dim = int(struct_dim)
        self.mask_topology = bool(mask_topology)

        self.sem_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.sem_in_dim, sem_out),
            nn.GELU(),
            nn.LayerNorm(sem_out),
        )
        self.struct_proj = nn.Sequential(
            nn.Linear(self.struct_dim, struct_out),
            nn.GELU(),
            nn.LayerNorm(struct_out),
        )

    def _apply_topology_mask(self, s_vec: Tensor) -> Tensor:
        if not self.mask_topology:
            return s_vec
        masked = s_vec.clone()
        cols = [c for c in (1, 4, 6, 7) if c < masked.size(1)]
        if cols:
            masked[:, cols] = 0.0
        return masked

    def _resolve_struct_x(self, x: Tensor, data: Any, root_idx: Tensor) -> Tensor:
        raw_struct = getattr(data, "struct_x", None)
        if (
            isinstance(raw_struct, Tensor)
            and raw_struct.dim() == 2
            and raw_struct.size(0) == x.size(0)
            and raw_struct.size(1) == self.struct_dim
        ):
            return self._apply_topology_mask(raw_struct.to(device=x.device, dtype=x.dtype))

        num_nodes = x.size(0)
        s_vec = x.new_zeros((num_nodes, self.struct_dim))
        if num_nodes == 0:
            return s_vec

        if x.size(1) >= 769:
            s_vec[:, 0] = x[:, 768].clamp(0.0, 1.0)
        if x.size(1) >= 770 and not self.mask_topology:
            s_vec[:, 4] = x[:, 769].clamp(0.0, 1.0)

        valid_roots = root_idx[(root_idx >= 0) & (root_idx < num_nodes)]
        is_root = x.new_zeros((num_nodes,), dtype=torch.bool)
        if valid_roots.numel() > 0:
            is_root[valid_roots] = True
            s_vec[valid_roots, 5] = 1.0

        if not self.mask_topology:
            raw_ei = getattr(data, "edge_index_td", getattr(data, "edge_index", None))
            if isinstance(raw_ei, Tensor) and raw_ei.numel() > 0:
                src_nodes = raw_ei[0].long()
                valid_mask = (src_nodes >= 0) & (src_nodes < num_nodes)
                if bool(valid_mask.any()):
                    deg_out = degree(src_nodes[valid_mask], num_nodes=num_nodes, dtype=x.dtype)
                else:
                    deg_out = x.new_zeros((num_nodes,))
            else:
                deg_out = x.new_zeros((num_nodes,))

            s_vec[:, 6] = torch.clamp(torch.log1p(deg_out) / math.log1p(20.0), max=1.0)
            s_vec[:, 7] = ((deg_out == 0.0) & (~is_root)).to(dtype=x.dtype)
        return self._apply_topology_mask(s_vec)

    def forward(self, x: Tensor, data: Any, root_idx: Tensor) -> Tensor:
        sem_in = x[:, : self.sem_in_dim]
        s_vec = self._resolve_struct_x(x, data, root_idx)
        return torch.cat([self.sem_proj(sem_in), self.struct_proj(s_vec)], dim=-1)


class MultiTaskHead(nn.Module):
    """Shared multi-task head predicting cascade size, polarization, rumor class, and uncertainty."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_rumor_classes: int = 4,
        dropout: float = 0.1,
        cascade_feat_dim: int = 0,
        predict_uncertainty: bool = False,
        mask_semantic_stats: bool = False,
        mask_topology_stats: bool = False,
        use_polar_persistence_anchor: bool = False,
    ) -> None:
        super().__init__()
        self.cascade_feat_dim = int(cascade_feat_dim)
        self.predict_uncertainty = bool(predict_uncertainty)
        self.mask_semantic_stats = bool(mask_semantic_stats)
        self.mask_topology_stats = bool(mask_topology_stats)
        self.use_polar_persistence_anchor = bool(use_polar_persistence_anchor)
        self.persist_weight = (
            nn.Parameter(torch.zeros(1)) if self.use_polar_persistence_anchor else None
        )
        if self.use_polar_persistence_anchor:
            self.register_buffer("persist_ep_logit_ref", torch.zeros(1))
        if self.cascade_feat_dim > 0:
            stats_out_dim = max(16, hidden_dim // 2)
            self.cascade_proj = nn.Sequential(
                nn.Linear(self.cascade_feat_dim, stats_out_dim),
                nn.GELU(),
                nn.LayerNorm(stats_out_dim),
            )
            reg_in_dim = in_dim + stats_out_dim
        else:
            self.cascade_proj = None
            reg_in_dim = in_dim

        self.size_head = nn.Sequential(
            nn.Linear(reg_in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.polar_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.rumor_head = nn.Sequential(
            nn.Linear(reg_in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_rumor_classes),
        )
        if self.predict_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(reg_in_dim, max(32, hidden_dim // 2)),
                nn.GELU(),
                nn.Linear(max(32, hidden_dim // 2), 2),
            )
        else:
            self.uncertainty_head = None

    def forward(
        self,
        graph_emb: Tensor,
        node_emb: Tensor,
        cascade_stats: Tensor | None = None,
        early_y_polar: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if self.cascade_proj is not None:
            if cascade_stats is None:
                stats_t = graph_emb.new_zeros((graph_emb.size(0), self.cascade_feat_dim))
            else:
                stats_t = cascade_stats.float().to(graph_emb.device)
                if stats_t.dim() == 1:
                    stats_t = stats_t.view(graph_emb.size(0), -1)
                if stats_t.size(-1) < self.cascade_feat_dim:
                    stats_t = F.pad(stats_t, (0, self.cascade_feat_dim - stats_t.size(-1)))
                elif stats_t.size(-1) > self.cascade_feat_dim:
                    stats_t = stats_t[:, : self.cascade_feat_dim]
            if (self.mask_semantic_stats or self.mask_topology_stats) and stats_t.size(-1) >= 8:
                stats_t = stats_t.clone()
                if self.mask_semantic_stats:
                    stats_t[:, 7] = 0.0
                if self.mask_topology_stats:
                    stats_t[:, 3] = 0.0
                    stats_t[:, 4] = 0.0
            reg_emb = torch.cat([graph_emb, self.cascade_proj(stats_t)], dim=-1)
        else:
            reg_emb = graph_emb

        y_size_pred = F.softplus(self.size_head(reg_emb)).squeeze(-1)
        y_polar_raw_logit = self.polar_head(graph_emb).squeeze(-1)
        if self.persist_weight is not None and early_y_polar is not None:
            ep = early_y_polar.float().to(device=graph_emb.device, dtype=graph_emb.dtype).view(-1)
            if ep.size(0) == graph_emb.size(0):
                ep_clamped = ep.clamp(0.02, 0.98)
                ep_logit = torch.log(ep_clamped / (1.0 - ep_clamped))
                ep_ref = getattr(self, "persist_ep_logit_ref", None)
                if ep_ref is not None:
                    ep_logit = ep_logit - ep_ref.to(device=ep_logit.device, dtype=ep_logit.dtype)
                alpha_p = torch.tanh(self.persist_weight.view(1))
                y_polar_logit = alpha_p * ep_logit + y_polar_raw_logit
            else:
                y_polar_logit = y_polar_raw_logit + 0.0 * self.persist_weight.sum()
        elif self.persist_weight is not None:
            y_polar_logit = y_polar_raw_logit + 0.0 * self.persist_weight.sum()
        else:
            y_polar_logit = y_polar_raw_logit
        y_polar_pred = torch.sigmoid(y_polar_logit)
        rumor_logits = self.rumor_head(reg_emb)
        result: dict[str, Tensor] = {
            "y_size_pred": y_size_pred,
            "y_polar_logit": y_polar_logit,
            "y_polar_pred": y_polar_pred,
            "rumor_logits": rumor_logits,
            "graph_emb": graph_emb,
            "node_emb": node_emb,
        }
        if self.uncertainty_head is not None:
            raw_logvars = -6.0 + 9.0 * torch.sigmoid(self.uncertainty_head(reg_emb.detach()))
            size_logvar = raw_logvars[:, 0]
            polar_logvar = raw_logvars[:, 1]
            polar_sigma = torch.exp(0.5 * polar_logvar)
            confidence = torch.exp(-polar_sigma / 0.25).clamp(0.0, 1.0)
            result["size_logvar"] = size_logvar
            result["polar_logvar"] = polar_logvar
            result["confidence_score"] = confidence
        return result


class TextMLP(nn.Module):
    """Graph-agnostic baseline: mean-pools node NLP representations into an MLP."""

    def __init__(
        self,
        in_dim: int = 770,
        hidden_dim: int = 64,
        num_rumor_classes: int = 4,
        dropout: float = 0.1,
        use_cascade_stats: bool = False,
        cascade_feat_dim: int = 8,
        predict_uncertainty: bool = False,
    ) -> None:
        super().__init__()
        self.use_cascade_stats = bool(use_cascade_stats)
        self.cascade_feat_dim = int(cascade_feat_dim)
        self.stem = TwoStreamNodeStem(
            in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout, mask_topology=True
        )
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = MultiTaskHead(
            2 * hidden_dim,
            hidden_dim,
            num_rumor_classes,
            dropout,
            cascade_feat_dim=self.cascade_feat_dim if self.use_cascade_stats else 0,
            predict_uncertainty=bool(predict_uncertainty),
            mask_semantic_stats=True,
            mask_topology_stats=True,
        )

    def forward(self, data: Any) -> dict[str, Tensor]:
        x = data.x.float()
        root_idx, batch_vec = resolve_root_indices(data, x.size(0))
        num_graphs = int(root_idx.size(0))
        h = self.stem(x, data, root_idx)
        node_emb = self.encoder(h)
        graph_emb = torch.cat(
            [node_emb[root_idx], global_mean_pool(node_emb, batch_vec, size=num_graphs)],
            dim=-1,
        )
        c_stats = getattr(data, "cascade_stats", None) if self.use_cascade_stats else None
        return self.head(graph_emb, node_emb, cascade_stats=c_stats)


class TemporalStanceMLP(nn.Module):
    """Non-graph ablation baseline fusing node temporal features and pooled 2D/4D edge statistics."""

    def __init__(
        self,
        in_dim: int = 770,
        hidden_dim: int = 64,
        edge_dim: int = 2,
        num_rumor_classes: int = 4,
        dropout: float = 0.1,
        use_cascade_stats: bool = False,
        cascade_feat_dim: int = 8,
        predict_uncertainty: bool = False,
        use_polar_persistence_anchor: bool = True,
    ) -> None:
        super().__init__()
        self.edge_dim = edge_dim
        self.use_cascade_stats = bool(use_cascade_stats)
        self.cascade_feat_dim = int(cascade_feat_dim)
        self.use_polar_persistence_anchor = bool(use_polar_persistence_anchor)
        self.stem = TwoStreamNodeStem(
            in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout, mask_topology=False
        )
        self.node_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim + 4 * edge_dim, 2 * hidden_dim),
            nn.LayerNorm(2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = MultiTaskHead(
            2 * hidden_dim,
            hidden_dim,
            num_rumor_classes,
            dropout,
            cascade_feat_dim=self.cascade_feat_dim if self.use_cascade_stats else 0,
            predict_uncertainty=bool(predict_uncertainty),
            mask_semantic_stats=False,
            use_polar_persistence_anchor=self.use_polar_persistence_anchor,
        )

    def forward(self, data: Any) -> dict[str, Tensor]:
        x = data.x.float()
        root_idx, batch_vec = resolve_root_indices(data, x.size(0))
        num_graphs = int(root_idx.size(0))

        h_stem = self.stem(x, data, root_idx)
        node_emb = self.node_encoder(h_stem)
        h_root = node_emb[root_idx]
        h_pool = global_mean_pool(node_emb, batch_vec, size=num_graphs)

        e_mean = torch.zeros((num_graphs, self.edge_dim), device=x.device, dtype=node_emb.dtype)
        e_std = torch.zeros_like(e_mean)
        e_min = torch.zeros_like(e_mean)
        e_max = torch.zeros_like(e_mean)

        raw_edge_index = getattr(data, "edge_index_td", getattr(data, "edge_index", None))
        edge_attr = getattr(data, "edge_attr", None)
        if (
            edge_attr is not None
            and raw_edge_index is not None
            and edge_attr.numel() > 0
            and raw_edge_index.numel() > 0
        ):
            attr = edge_attr.to(device=x.device, dtype=node_emb.dtype)
            if attr.dim() == 1:
                attr = attr.unsqueeze(-1)
            if attr.size(1) < self.edge_dim:
                attr = F.pad(attr, (0, self.edge_dim - attr.size(1)))
            elif attr.size(1) > self.edge_dim:
                attr = attr[:, : self.edge_dim]
            edge_batch = batch_vec[raw_edge_index[0]]
            ones = torch.ones((attr.size(0), 1), device=x.device, dtype=node_emb.dtype)
            edge_counts = scatter(ones, edge_batch, dim=0, dim_size=num_graphs, reduce="sum").squeeze(-1)
            has_edges = (edge_counts > 0).unsqueeze(-1)
            has_multi_edges = (edge_counts >= 2).unsqueeze(-1)

            e_mean_raw = scatter(attr, edge_batch, dim=0, dim_size=num_graphs, reduce="mean")
            e_sq_raw = scatter(attr.pow(2), edge_batch, dim=0, dim_size=num_graphs, reduce="mean")
            e_min_raw = scatter(attr, edge_batch, dim=0, dim_size=num_graphs, reduce="min")
            e_max_raw = scatter(attr, edge_batch, dim=0, dim_size=num_graphs, reduce="max")

            e_mean = torch.where(has_edges, e_mean_raw, e_mean)
            e_min = torch.where(has_edges, e_min_raw, e_min)
            e_max = torch.where(has_edges, e_max_raw, e_max)
            e_std_raw = (e_sq_raw - e_mean_raw.pow(2)).clamp(min=0.0).sqrt()
            e_std = torch.where(has_multi_edges, e_std_raw, e_std)

        fused_in = torch.cat([h_root, h_pool, e_mean, e_std, e_min, e_max], dim=-1)
        graph_emb = self.fusion(fused_in)
        c_stats = getattr(data, "cascade_stats", None) if self.use_cascade_stats else None
        early_yp = getattr(data, "early_y_polar", None) if self.use_polar_persistence_anchor else None
        return self.head(graph_emb, node_emb, cascade_stats=c_stats, early_y_polar=early_yp)


class StandardGCN(nn.Module):
    """Standard undirected GCN baseline (Kipf & Welling, ICLR 2017)."""

    def __init__(
        self,
        in_dim: int = 770,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_rumor_classes: int = 4,
        dropout: float = 0.1,
        use_cascade_stats: bool = False,
        cascade_feat_dim: int = 8,
        predict_uncertainty: bool = False,
    ) -> None:
        super().__init__()
        self.use_cascade_stats = bool(use_cascade_stats)
        self.cascade_feat_dim = int(cascade_feat_dim)
        self.input_proj = TwoStreamNodeStem(
            in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout
        )
        self.convs = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.head = MultiTaskHead(
            2 * hidden_dim,
            hidden_dim,
            num_rumor_classes,
            dropout,
            cascade_feat_dim=self.cascade_feat_dim if self.use_cascade_stats else 0,
            predict_uncertainty=bool(predict_uncertainty),
            mask_semantic_stats=True,
        )

    def forward(self, data: Any) -> dict[str, Tensor]:
        x = data.x.float()
        root_idx, batch_vec = resolve_root_indices(data, x.size(0))
        raw_edge_index = getattr(data, "edge_index_td", data.edge_index)
        edge_index = (
            to_undirected(raw_edge_index, num_nodes=x.size(0))
            if raw_edge_index.numel() > 0
            else raw_edge_index
        )

        h = self.input_proj(x, data, root_idx)
        for conv, norm in zip(self.convs, self.norms):
            h = norm(F.relu(conv(h, edge_index)))
            h = self.dropout(h)

        graph_emb = torch.cat([h[root_idx], global_mean_pool(h, batch_vec)], dim=-1)
        c_stats = getattr(data, "cascade_stats", None) if self.use_cascade_stats else None
        return self.head(graph_emb, h, cascade_stats=c_stats)


class StandardGATv2(nn.Module):
    """Dynamic GATv2 baseline without edge stance signals (Brody et al., ICLR 2022)."""

    def __init__(
        self,
        in_dim: int = 770,
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        num_rumor_classes: int = 4,
        dropout: float = 0.1,
        use_cascade_stats: bool = False,
        cascade_feat_dim: int = 8,
        predict_uncertainty: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})"
            )
        self.use_cascade_stats = bool(use_cascade_stats)
        self.cascade_feat_dim = int(cascade_feat_dim)
        self.input_proj = TwoStreamNodeStem(
            in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout
        )
        head_dim = hidden_dim // heads
        self.convs = nn.ModuleList(
            [
                GATv2Conv(
                    hidden_dim,
                    head_dim,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    add_self_loops=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.head = MultiTaskHead(
            2 * hidden_dim,
            hidden_dim,
            num_rumor_classes,
            dropout,
            cascade_feat_dim=self.cascade_feat_dim if self.use_cascade_stats else 0,
            predict_uncertainty=bool(predict_uncertainty),
            mask_semantic_stats=True,
        )

    def forward(self, data: Any) -> dict[str, Tensor]:
        x = data.x.float()
        root_idx, batch_vec = resolve_root_indices(data, x.size(0))
        raw_edge_index = getattr(data, "edge_index_td", data.edge_index)
        edge_index = (
            to_undirected(raw_edge_index, num_nodes=x.size(0))
            if raw_edge_index.numel() > 0
            else raw_edge_index
        )

        h = self.input_proj(x, data, root_idx)
        for conv, norm in zip(self.convs, self.norms):
            h = norm(F.elu(conv(h, edge_index)))
            h = self.dropout(h)

        graph_emb = torch.cat([h[root_idx], global_mean_pool(h, batch_vec)], dim=-1)
        c_stats = getattr(data, "cascade_stats", None) if self.use_cascade_stats else None
        return self.head(graph_emb, h, cascade_stats=c_stats)


class _BiGCNBranch(nn.Module):
    """Single directional GCN branch with DropEdge & Root Feature Enhancement."""

    def __init__(self, hidden_dim: int, num_layers: int, drop_edge_p: float, dropout: float) -> None:
        super().__init__()
        self.drop_edge_p = drop_edge_p
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.root_projs = nn.ModuleList(
            [nn.Linear(2 * hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h0: Tensor,
        edge_index: Tensor,
        root_idx: Tensor,
        batch_vec: Tensor,
    ) -> Tensor:
        if self.training and self.drop_edge_p > 0.0 and edge_index.numel() > 0:
            edge_index, _ = dropout_edge(
                edge_index, p=self.drop_edge_p, force_undirected=False, training=True
            )
        root_anchor = broadcast_root_features(h0, root_idx, batch_vec)
        h = h0
        for conv, root_proj, norm in zip(self.convs, self.root_projs, self.norms):
            h_conv = F.relu(conv(h, edge_index))
            h = norm(F.relu(root_proj(torch.cat([h_conv, root_anchor], dim=-1))))
            h = self.dropout(h)
        return h


class BiGCN(nn.Module):
    """Bi-Directional GCN baseline (Bian et al., AAAI 2020) without conflict gating."""

    def __init__(
        self,
        in_dim: int = 770,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_rumor_classes: int = 4,
        drop_edge_p: float = 0.1,
        dropout: float = 0.1,
        use_cascade_stats: bool = False,
        cascade_feat_dim: int = 8,
        predict_uncertainty: bool = False,
    ) -> None:
        super().__init__()
        self.use_cascade_stats = bool(use_cascade_stats)
        self.cascade_feat_dim = int(cascade_feat_dim)
        self.input_proj = TwoStreamNodeStem(
            in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout
        )
        self.td_branch = _BiGCNBranch(hidden_dim, num_layers, drop_edge_p, dropout)
        self.bu_branch = _BiGCNBranch(hidden_dim, num_layers, drop_edge_p, dropout)
        self.head = MultiTaskHead(
            2 * hidden_dim,
            hidden_dim,
            num_rumor_classes,
            dropout,
            cascade_feat_dim=self.cascade_feat_dim if self.use_cascade_stats else 0,
            predict_uncertainty=bool(predict_uncertainty),
            mask_semantic_stats=True,
        )

    def forward(self, data: Any) -> dict[str, Tensor]:
        x = data.x.float()
        root_idx, batch_vec = resolve_root_indices(data, x.size(0))
        edge_index_td = getattr(data, "edge_index_td", data.edge_index)
        edge_index_bu = getattr(
            data,
            "edge_index_bu",
            edge_index_td[[1, 0], :] if edge_index_td.numel() > 0 else edge_index_td,
        )

        h0 = self.input_proj(x, data, root_idx)
        h_td = self.td_branch(h0, edge_index_td, root_idx, batch_vec)
        h_bu = self.bu_branch(h0, edge_index_bu, root_idx, batch_vec)

        node_emb = 0.5 * (h_td + h_bu)
        graph_emb = torch.cat(
            [global_mean_pool(h_td, batch_vec), global_mean_pool(h_bu, batch_vec)],
            dim=-1,
        )
        c_stats = getattr(data, "cascade_stats", None) if self.use_cascade_stats else None
        return self.head(graph_emb, node_emb, cascade_stats=c_stats)


class LatentConflictMLP(nn.Module):
    """Shared 2-layer latent edge inferrer (`[s_hat, h_hat]` or 4D) for text-free graphs (e.g. UPFD)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        edge_dim: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.edge_dim = int(edge_dim)
        self.net = nn.Sequential(
            nn.Linear(3 * in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.edge_dim),
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        if edge_index.numel() == 0:
            return x.new_zeros((0, self.edge_dim))
        src, dst = edge_index[0], edge_index[1]
        pair_feat = torch.cat([x[src], x[dst], x[src] - x[dst]], dim=-1)
        raw = self.net(pair_feat)
        if self.edge_dim == 1:
            return torch.tanh(raw[:, 0:1])
        return torch.cat([torch.tanh(raw[:, 0:1]), torch.sigmoid(raw[:, 1:])], dim=-1)


_BaselineLatentEdgeMLP = LatentConflictMLP


class EdgeGATv2(nn.Module):
    """Edge-aware GATv2 baseline receiving `[s_uv, h_uv]` or 4D edge features."""

    def __init__(
        self,
        in_dim: int = 770,
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        edge_dim: int = 2,
        num_rumor_classes: int = 4,
        dropout: float = 0.1,
        use_latent_edge_mlp: bool = False,
        use_cascade_stats: bool = False,
        cascade_feat_dim: int = 8,
        predict_uncertainty: bool = False,
        use_polar_persistence_anchor: bool | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})"
            )
        self.edge_dim = edge_dim
        self.use_latent_edge_mlp = use_latent_edge_mlp
        self.use_cascade_stats = bool(use_cascade_stats)
        self.cascade_feat_dim = int(cascade_feat_dim)
        self.use_polar_persistence_anchor = (
            bool(not use_latent_edge_mlp)
            if use_polar_persistence_anchor is None
            else bool(use_polar_persistence_anchor)
        )
        self.input_proj = TwoStreamNodeStem(
            in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout
        )
        self.latent_edge_mlp = (
            LatentConflictMLP(
                in_dim=in_dim, hidden_dim=hidden_dim, edge_dim=edge_dim, dropout=dropout
            )
            if use_latent_edge_mlp
            else None
        )
        head_dim = hidden_dim // heads
        self.convs = nn.ModuleList(
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
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.head = MultiTaskHead(
            2 * hidden_dim,
            hidden_dim,
            num_rumor_classes,
            dropout,
            cascade_feat_dim=self.cascade_feat_dim if self.use_cascade_stats else 0,
            predict_uncertainty=bool(predict_uncertainty),
            mask_semantic_stats=False,
            use_polar_persistence_anchor=self.use_polar_persistence_anchor,
        )

    def _prepare_edge_attr(self, x: Tensor, edge_index: Tensor, data: Any) -> Tensor:
        if self.latent_edge_mlp is not None:
            return self.latent_edge_mlp(x, edge_index)
        raw_attr = getattr(data, "edge_attr", None)
        if raw_attr is None:
            raw_attr = getattr(data, "edge_stance", None)
        if raw_attr is None:
            return x.new_zeros((edge_index.size(1), self.edge_dim))
        attr = raw_attr.float()
        if attr.dim() == 1:
            attr = attr.unsqueeze(-1)
        if attr.size(1) < self.edge_dim:
            attr = F.pad(attr, (0, self.edge_dim - attr.size(1)))
        elif attr.size(1) > self.edge_dim:
            attr = attr[:, : self.edge_dim]
        return attr

    def forward(self, data: Any) -> dict[str, Tensor]:
        x = data.x.float()
        root_idx, batch_vec = resolve_root_indices(data, x.size(0))
        raw_edge_index = getattr(data, "edge_index_td", data.edge_index)
        raw_edge_attr = self._prepare_edge_attr(x, raw_edge_index, data)
        if raw_edge_index.numel() > 0:
            edge_index, edge_attr = to_undirected(
                raw_edge_index,
                edge_attr=raw_edge_attr,
                num_nodes=x.size(0),
                reduce="mean",
            )
        else:
            edge_index = raw_edge_index
            edge_attr = raw_edge_attr

        h = self.input_proj(x, data, root_idx)
        for conv, norm in zip(self.convs, self.norms):
            h = norm(F.elu(conv(h, edge_index, edge_attr=edge_attr)))
            h = self.dropout(h)

        graph_emb = torch.cat([h[root_idx], global_mean_pool(h, batch_vec)], dim=-1)
        c_stats = getattr(data, "cascade_stats", None) if self.use_cascade_stats else None
        early_yp = getattr(data, "early_y_polar", None) if self.use_polar_persistence_anchor else None
        return self.head(graph_emb, h, cascade_stats=c_stats, early_y_polar=early_yp)


class FAGCNBaseline(nn.Module):
    """Frequency Adaptation GCN baseline (Bo et al., AAAI 2021) using PyG `FAConv`."""

    def __init__(
        self,
        in_dim: int = 770,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_rumor_classes: int = 4,
        eps: float = 0.5,
        dropout: float = 0.1,
        use_cascade_stats: bool = False,
        cascade_feat_dim: int = 8,
        predict_uncertainty: bool = False,
    ) -> None:
        super().__init__()
        self.use_cascade_stats = bool(use_cascade_stats)
        self.cascade_feat_dim = int(cascade_feat_dim)
        self.input_proj = TwoStreamNodeStem(
            in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout
        )
        self.convs = nn.ModuleList(
            [
                FAConv(
                    channels=hidden_dim,
                    eps=eps,
                    dropout=dropout,
                    add_self_loops=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.head = MultiTaskHead(
            2 * hidden_dim,
            hidden_dim,
            num_rumor_classes,
            dropout,
            cascade_feat_dim=self.cascade_feat_dim if self.use_cascade_stats else 0,
            predict_uncertainty=bool(predict_uncertainty),
            mask_semantic_stats=True,
        )

    def forward(self, data: Any) -> dict[str, Tensor]:
        x = data.x.float()
        root_idx, batch_vec = resolve_root_indices(data, x.size(0))
        raw_edge_index = getattr(data, "edge_index_td", data.edge_index)
        edge_index = (
            to_undirected(raw_edge_index, num_nodes=x.size(0))
            if raw_edge_index.numel() > 0
            else raw_edge_index
        )

        h0 = self.input_proj(x, data, root_idx)
        h = h0
        for conv, norm in zip(self.convs, self.norms):
            h = norm(F.elu(conv(h, h0, edge_index)))
            h = self.dropout(h)

        graph_emb = torch.cat([h[root_idx], global_mean_pool(h, batch_vec)], dim=-1)
        c_stats = getattr(data, "cascade_stats", None) if self.use_cascade_stats else None
        return self.head(graph_emb, h, cascade_stats=c_stats)


class _BaselineAttentionReadout(nn.Module):
    """Per-graph softmax attention readout used by matched bidirectional baselines."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.query = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h: Tensor, batch_vec: Tensor, num_graphs: int) -> Tensor:
        scores = self.query(torch.tanh(self.proj(h))).squeeze(-1)
        attn = softmax(scores, batch_vec, num_nodes=num_graphs).unsqueeze(-1)
        out = h.new_zeros((num_graphs, h.size(-1)))
        out.index_add_(0, batch_vec, h * attn)
        return out


class BiEdgeGATv2(nn.Module):
    """Matched Bidirectional Edge-Aware GATv2 comparator (TD+BU + 2D edge_attr + attention readout).

    Controls for bidirectionality, edge attributes, and attention pooling so ablation
    gains of `SBiFAGAT` are cleanly attributed to signed frequency-adaptive gating and
    Camp Dipole readout rather than generic architecture capacity.
    """

    def __init__(
        self,
        in_dim: int = 770,
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        edge_dim: int = 2,
        eps: float = 0.5,
        num_rumor_classes: int = 4,
        dropout: float = 0.1,
        use_latent_edge_mlp: bool = False,
        use_cascade_stats: bool = False,
        cascade_feat_dim: int = 8,
        predict_uncertainty: bool = False,
        use_polar_persistence_anchor: bool | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})"
            )
        self.edge_dim = edge_dim
        self.eps = float(eps)
        self.use_latent_edge_mlp = use_latent_edge_mlp
        self.use_cascade_stats = bool(use_cascade_stats)
        self.cascade_feat_dim = int(cascade_feat_dim)
        self.use_polar_persistence_anchor = (
            bool(not use_latent_edge_mlp)
            if use_polar_persistence_anchor is None
            else bool(use_polar_persistence_anchor)
        )
        self.input_proj = TwoStreamNodeStem(
            in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout
        )
        self.root_lin = nn.Linear(hidden_dim, hidden_dim)
        self.latent_edge_mlp = (
            LatentConflictMLP(
                in_dim=in_dim, hidden_dim=hidden_dim, edge_dim=edge_dim, dropout=dropout
            )
            if use_latent_edge_mlp
            else None
        )
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
        self.td_readout = _BaselineAttentionReadout(hidden_dim)
        self.bu_readout = _BaselineAttentionReadout(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = MultiTaskHead(
            2 * hidden_dim,
            hidden_dim,
            num_rumor_classes,
            dropout,
            cascade_feat_dim=self.cascade_feat_dim if self.use_cascade_stats else 0,
            predict_uncertainty=bool(predict_uncertainty),
            mask_semantic_stats=False,
            use_polar_persistence_anchor=self.use_polar_persistence_anchor,
        )

    def _prepare_edge_attr(self, x: Tensor, edge_index: Tensor, data: Any) -> Tensor:
        if self.latent_edge_mlp is not None:
            return self.latent_edge_mlp(x, edge_index)
        raw_attr = getattr(data, "edge_attr", None)
        if raw_attr is None:
            raw_attr = getattr(data, "edge_stance", None)
        if raw_attr is None:
            return x.new_zeros((edge_index.size(1), self.edge_dim))
        attr = raw_attr.float()
        if attr.dim() == 1:
            attr = attr.unsqueeze(-1)
        if attr.size(1) < self.edge_dim:
            attr = F.pad(attr, (0, self.edge_dim - attr.size(1)))
        elif attr.size(1) > self.edge_dim:
            attr = attr[:, : self.edge_dim]
        return attr

    def forward(self, data: Any) -> dict[str, Tensor]:
        x = data.x.float()
        root_idx, batch_vec = resolve_root_indices(data, x.size(0))
        num_graphs = int(root_idx.size(0))
        edge_index_td = getattr(data, "edge_index_td", data.edge_index)
        edge_index_bu = getattr(
            data,
            "edge_index_bu",
            edge_index_td[[1, 0], :] if edge_index_td.numel() > 0 else edge_index_td,
        )
        edge_attr = self._prepare_edge_attr(x, edge_index_td, data)

        h0 = self.input_proj(x, data, root_idx)
        root_broadcast = self.root_lin(h0[root_idx][batch_vec])
        h_td = h0
        h_bu = h0
        for td_conv, bu_conv, td_norm, bu_norm in zip(
            self.td_convs, self.bu_convs, self.td_norms, self.bu_norms
        ):
            td_msg = F.elu(td_conv(h_td, edge_index_td, edge_attr=edge_attr))
            bu_msg = F.elu(bu_conv(h_bu, edge_index_bu, edge_attr=edge_attr))
            h_td = self.dropout(td_norm(h_td + td_msg + self.eps * h0 + root_broadcast))
            h_bu = self.dropout(bu_norm(h_bu + bu_msg + self.eps * h0 + root_broadcast))

        node_emb = 0.5 * (h_td + h_bu)
        z_root = 0.5 * (h_td[root_idx] + h_bu[root_idx])
        z_att = 0.5 * (
            self.td_readout(h_td, batch_vec, num_graphs)
            + self.bu_readout(h_bu, batch_vec, num_graphs)
        )
        graph_emb = torch.cat([z_root, z_att], dim=-1)
        c_stats = getattr(data, "cascade_stats", None) if self.use_cascade_stats else None
        early_yp = getattr(data, "early_y_polar", None) if self.use_polar_persistence_anchor else None
        return self.head(graph_emb, node_emb, cascade_stats=c_stats, early_y_polar=early_yp)
