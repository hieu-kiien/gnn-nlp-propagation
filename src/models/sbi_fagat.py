"""SBi-FAGAT: Stance-Guided Bidirectional Frequency-Adaptive Graph Attention Network.

Implements the core architecture specified in `docs/math_spec.md` using
`torch_geometric.nn.MessagePassing`:
  - Supports 2D edge attributes `e_uv = [s_uv, h_uv]` (stance polarity + hostility
    intensity) feeding the signed frequency-adaptive gate:
        gamma_uv = tanh(a^T [W h_u || W h_v || e_uv])
    (and `sigmoid` variant for conflict-gate ablation).
  - Runs independent Top-Down (propagation) and Bottom-Up (community feedback)
    message passing with root anchoring and degree normalization.
  - Provides `LatentConflictMLP(x_u, x_v, x_u - x_v)` for datasets without raw
    edge texts (e.g., PyG UPFD BERT features).
  - Exposes `extract_edge_gates(data)` to inspect `|gamma_uv|` and signed `gamma_uv`
    on every edge for model interpretability and cascade heatmaps.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import degree, softmax

from src.models.baselines import (
    MultiTaskHead,
    broadcast_root_features,
    resolve_root_indices,
)


class LatentConflictMLP(nn.Module):
    """Infer 2D latent edge signals `[s_hat, h_hat]` from endpoint node features.

    Used on benchmarks like UPFD where raw reply text is unavailable and only
    node embeddings (e.g., 768-d BERT) are provided.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        if edge_index.numel() == 0:
            return x.new_zeros((0, 2))
        src, dst = edge_index[0], edge_index[1]
        x_u, x_v = x[src], x[dst]
        pair_feat = torch.cat([x_u, x_v, x_u - x_v], dim=-1)
        raw = self.net(pair_feat)
        s_hat = torch.tanh(raw[:, 0:1])
        h_hat = torch.sigmoid(raw[:, 1:2])
        return torch.cat([s_hat, h_hat], dim=-1)


class StanceGuidedFAConv(MessagePassing):
    """Stance-guided frequency-adaptive graph attention convolution (`MessagePassing`)."""

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int = 2,
        eps: float = 0.5,
        gate_mode: Literal["tanh", "sigmoid"] = "tanh",
        dropout: float = 0.1,
    ) -> None:
        super().__init__(aggr="add", flow="source_to_target", node_dim=0)
        if gate_mode not in {"tanh", "sigmoid"}:
            raise ValueError(f"gate_mode must be 'tanh' or 'sigmoid', got {gate_mode!r}")

        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.eps = eps
        self.gate_mode = gate_mode

        self.lin = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lin_root = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.att = nn.Parameter(torch.empty(1, 2 * hidden_dim + edge_dim))
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.lin_root.weight)
        nn.init.zeros_(self.lin_root.bias)
        node_att_dim = 2 * self.hidden_dim
        nn.init.xavier_uniform_(self.att[:, :node_att_dim])
        if self.edge_dim > 0:
            edge_scale = 1.0 / math.sqrt(float(self.edge_dim))
            nn.init.constant_(self.att[:, node_att_dim:], edge_scale)

    def forward(
        self,
        h: Tensor,
        h_0: Tensor,
        h_root_broadcast: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> tuple[Tensor, Tensor]:
        num_nodes = h.size(0)
        z = self.lin(h)

        if edge_index.numel() > 0:
            src, dst = edge_index[0], edge_index[1]
            deg_out = degree(src, num_nodes=num_nodes, dtype=z.dtype)
            deg_in = degree(dst, num_nodes=num_nodes, dtype=z.dtype)
            edge_norm = ((deg_out[src] + 1.0) * (deg_in[dst] + 1.0)).pow(-0.5)
            cat_feat = torch.cat([z[src], z[dst], edge_attr], dim=-1)
            score = (cat_feat * self.att).sum(dim=-1)
            gamma = torch.tanh(score) if self.gate_mode == "tanh" else torch.sigmoid(score)
            edge_weight = edge_norm * gamma
            m = self.propagate(
                edge_index,
                z=z,
                edge_weight=edge_weight,
                size=(num_nodes, num_nodes),
            )
        else:
            m = torch.zeros_like(z)
            gamma = z.new_zeros((0,))

        out = self.norm(F.elu(self.eps * h_0 + m + self.lin_root(h_root_broadcast)))
        out = self.dropout(out)
        return out, gamma

    def message(
        self,
        z_j: Tensor,
        edge_weight: Tensor,
    ) -> Tensor:
        return edge_weight.unsqueeze(-1) * z_j


class GlobalAttentionReadout(nn.Module):
    """Per-graph softmax attention readout over all early nodes in V_K."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.query = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h: Tensor, batch_vec: Tensor, num_graphs: int) -> Tensor:
        scores = self.query(torch.tanh(self.proj(h))).squeeze(-1)
        attn = softmax(scores, batch_vec, num_nodes=num_graphs).unsqueeze(-1)
        weighted = h * attn
        out = h.new_zeros((num_graphs, h.size(-1)))
        out.index_add_(0, batch_vec, weighted)
        return out


class SBiFAGAT(nn.Module):
    """Stance-Guided Bidirectional Frequency-Adaptive Graph Attention Network."""

    def __init__(
        self,
        in_dim: int = 770,
        hidden_dim: int = 128,
        num_layers: int = 2,
        edge_dim: int = 2,
        num_rumor_classes: int = 4,
        eps: float = 0.5,
        dropout: float = 0.1,
        gate_mode: Literal["tanh", "sigmoid"] = "tanh",
        use_bottom_up: bool = True,
        ablate_stance: bool = False,
        ablate_hostility: bool = False,
        use_latent_edge_mlp: bool = False,
        readout_mode: Literal["attention", "mean"] = "attention",
    ) -> None:
        super().__init__()
        if readout_mode not in {"attention", "mean"}:
            raise ValueError(f"readout_mode must be 'attention' or 'mean', got {readout_mode!r}")
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.edge_dim = edge_dim
        self.use_bottom_up = use_bottom_up
        self.ablate_stance = ablate_stance
        self.ablate_hostility = ablate_hostility
        self.use_latent_edge_mlp = use_latent_edge_mlp
        self.gate_mode = gate_mode
        self.readout_mode = readout_mode

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.latent_edge_mlp = (
            LatentConflictMLP(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
            if use_latent_edge_mlp
            else None
        )

        self.td_convs = nn.ModuleList(
            [
                StanceGuidedFAConv(
                    hidden_dim=hidden_dim,
                    edge_dim=edge_dim,
                    eps=eps,
                    gate_mode=gate_mode,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.bu_convs = (
            nn.ModuleList(
                [
                    StanceGuidedFAConv(
                        hidden_dim=hidden_dim,
                        edge_dim=edge_dim,
                        eps=eps,
                        gate_mode=gate_mode,
                        dropout=dropout,
                    )
                    for _ in range(num_layers)
                ]
            )
            if use_bottom_up
            else None
        )

        self.td_readout = (
            GlobalAttentionReadout(hidden_dim) if readout_mode == "attention" else None
        )
        self.bu_readout = (
            GlobalAttentionReadout(hidden_dim)
            if (use_bottom_up and readout_mode == "attention")
            else None
        )
        self.head = MultiTaskHead(4 * hidden_dim, hidden_dim, num_rumor_classes, dropout)

    def _prepare_edge_attr(
        self,
        x: Tensor,
        edge_index_td: Tensor,
        data: Any,
    ) -> Tensor:
        num_edges = edge_index_td.size(1)
        if self.latent_edge_mlp is not None:
            edge_attr = self.latent_edge_mlp(x, edge_index_td)
        else:
            raw_attr = getattr(data, "edge_attr", None)
            if raw_attr is None:
                raw_attr = getattr(data, "edge_stance", None)

            if raw_attr is None:
                edge_attr = x.new_zeros((num_edges, self.edge_dim))
            else:
                edge_attr = raw_attr.float()
                if edge_attr.dim() == 1:
                    edge_attr = edge_attr.unsqueeze(-1)
                if edge_attr.size(1) < self.edge_dim:
                    edge_attr = F.pad(edge_attr, (0, self.edge_dim - edge_attr.size(1)))
                elif edge_attr.size(1) > self.edge_dim:
                    edge_attr = edge_attr[:, : self.edge_dim]

        if self.ablate_stance or self.ablate_hostility:
            edge_attr = edge_attr.clone()
            if self.ablate_stance and edge_attr.size(1) >= 1:
                edge_attr[:, 0] = 0.0
            if self.ablate_hostility and edge_attr.size(1) >= 2:
                edge_attr[:, 1] = 0.0

        return edge_attr

    def _encode_graphs(
        self, data: Any
    ) -> tuple[dict[str, Tensor], list[Tensor], list[Tensor]]:
        x = data.x.float()
        root_idx, batch_vec = resolve_root_indices(data, x.size(0))
        num_graphs = root_idx.size(0)

        edge_index_td = getattr(data, "edge_index_td", data.edge_index)
        edge_index_bu = getattr(
            data,
            "edge_index_bu",
            edge_index_td[[1, 0], :] if edge_index_td.numel() > 0 else edge_index_td,
        )
        edge_attr = self._prepare_edge_attr(x, edge_index_td, data)

        h0 = self.input_proj(x)
        h_root_0 = broadcast_root_features(h0, root_idx, batch_vec)

        h_td = h0
        gamma_td_layers: list[Tensor] = []
        for conv in self.td_convs:
            h_td, gamma_td = conv(h_td, h0, h_root_0, edge_index_td, edge_attr)
            gamma_td_layers.append(gamma_td)

        z_root_td = h_td[root_idx]
        z_att_td = (
            self.td_readout(h_td, batch_vec, num_graphs)
            if self.td_readout is not None
            else global_mean_pool(h_td, batch_vec)
        )

        gamma_bu_layers: list[Tensor] = []
        if self.use_bottom_up and self.bu_convs is not None:
            h_bu = h0
            for conv in self.bu_convs:
                h_bu, gamma_bu = conv(h_bu, h0, h_root_0, edge_index_bu, edge_attr)
                gamma_bu_layers.append(gamma_bu)
            z_root_bu = h_bu[root_idx]
            z_att_bu = (
                self.bu_readout(h_bu, batch_vec, num_graphs)
                if self.bu_readout is not None
                else global_mean_pool(h_bu, batch_vec)
            )
            node_emb = 0.5 * (h_td + h_bu)
        else:
            z_root_bu = z_root_td
            z_att_bu = z_att_td
            node_emb = h_td

        graph_emb = torch.cat([z_root_td, z_att_td, z_root_bu, z_att_bu], dim=-1)
        outputs = self.head(graph_emb, node_emb)
        return outputs, gamma_td_layers, gamma_bu_layers

    def forward(self, data: Any) -> dict[str, Tensor]:
        outputs, _, _ = self._encode_graphs(data)
        return outputs

    @torch.no_grad()
    def extract_edge_gates(self, data: Any) -> dict[str, Any]:
        """Extract signed and magnitude `|gamma_uv|` edge weights for interpretability."""
        was_training = self.training
        self.eval()
        try:
            _, gamma_td_layers, gamma_bu_layers = self._encode_graphs(data)
        finally:
            self.train(was_training)

        gamma_td = gamma_td_layers[-1]
        abs_gamma_td = gamma_td.abs()
        if gamma_bu_layers:
            gamma_bu = gamma_bu_layers[-1]
            abs_gamma_bu = gamma_bu.abs()
            edge_importance = 0.5 * (abs_gamma_td + abs_gamma_bu)
        else:
            gamma_bu = torch.zeros_like(gamma_td)
            abs_gamma_bu = torch.zeros_like(abs_gamma_td)
            edge_importance = abs_gamma_td

        return {
            "gamma_td": gamma_td,
            "gamma_bu": gamma_bu,
            "abs_gamma_td": abs_gamma_td,
            "abs_gamma_bu": abs_gamma_bu,
            "edge_importance": edge_importance,
            "gamma_td_layers": gamma_td_layers,
            "gamma_bu_layers": gamma_bu_layers,
        }
