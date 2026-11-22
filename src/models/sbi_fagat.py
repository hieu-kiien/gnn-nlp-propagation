"""SBi-FAGAT: Stance-Guided Bidirectional Frequency-Adaptive Graph Attention Network.

Implements the core architecture specified in `docs/math_spec.md` using
`torch_geometric.nn.MessagePassing`:
  - Supports 2D edge attributes `e_uv = [s_uv, h_uv]` (stance polarity + hostility
    intensity) feeding dual-channel message passing:
      * Polarity channel (signed low/high-pass gate `gamma_uv in [-1, 1]`):
            gamma_uv = tanh(node_score + beta_cos * cos(z_u, z_v) + a_e^T e_uv + delta_pol(phi(e_uv)))
      * Excitatory intensity channel (`alpha_int in [0, 1]`):
            alpha_int = sigmoid(0.8 |s_uv| + 0.8 h_uv + delta_int(phi(e_uv)))
    (plus `sigmoid` gate-mode and single-channel ablations).
  - Runs coupled Top-Down (propagation) and Bottom-Up (community feedback)
    message passing with Dir-GNN cross-directional mixing, root anchoring, and
    symmetric degree normalization.
  - Aggregates graph-level representations via Camp Dipole + gate-moment readout
    over root-aligned edge signals.
  - Provides `LatentConflictMLP(x_u, x_v, x_u - x_v)` for text-free structural
    transfer benchmarks (e.g., PyG UPFD BERT features).
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
from torch_geometric.utils import degree, scatter, softmax

from src.models.baselines import (
    LatentConflictMLP,
    MultiTaskHead,
    TwoStreamNodeStem,
    broadcast_root_features,
    resolve_root_indices,
)


class StanceGuidedFAConv(MessagePassing):
    """Stance-guided frequency-adaptive graph attention convolution (`MessagePassing`)."""

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int = 2,
        eps: float = 0.5,
        gate_mode: Literal["tanh", "sigmoid"] = "tanh",
        dropout: float = 0.1,
        use_dual_channel: bool = True,
    ) -> None:
        super().__init__(aggr="add", flow="source_to_target", node_dim=0)
        if gate_mode not in {"tanh", "sigmoid"}:
            raise ValueError(f"gate_mode must be 'tanh' or 'sigmoid', got {gate_mode!r}")
        if hidden_dim < 2:
            raise ValueError(f"hidden_dim must be >= 2, got {hidden_dim}")

        self.hidden_dim = hidden_dim
        self.pol_dim = hidden_dim // 2
        self.edge_dim = edge_dim
        self.eps = eps
        self.gate_mode = gate_mode
        self.use_dual_channel = bool(use_dual_channel)

        self.lin = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lin_root = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.att = nn.Parameter(torch.empty(1, 2 * hidden_dim + edge_dim))
        self.cos_weight = nn.Parameter(torch.tensor(0.10))

        if edge_dim >= 4:
            basis_dim = edge_dim + 2
        elif edge_dim == 2:
            basis_dim = 4
        else:
            basis_dim = 2 * max(1, edge_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(basis_dim, 16),
            nn.GELU(),
            nn.Linear(16, 2),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.lin_root.weight)
        nn.init.zeros_(self.lin_root.bias)
        node_att_dim = 2 * self.hidden_dim
        nn.init.xavier_uniform_(self.att[:, :node_att_dim], gain=0.2)
        if self.edge_dim == 4:
            with torch.no_grad():
                self.att[:, node_att_dim:].copy_(
                    torch.tensor(
                        [[0.5, -0.25, -0.25, 0.25]],
                        dtype=self.att.dtype,
                        device=self.att.device,
                    )
                )
        elif self.edge_dim == 2:
            with torch.no_grad():
                self.att[:, node_att_dim:].copy_(
                    torch.tensor(
                        [[0.6, -0.4]],
                        dtype=self.att.dtype,
                        device=self.att.device,
                    )
                )
        elif self.edge_dim > 0:
            edge_scale = 1.0 / math.sqrt(float(self.edge_dim))
            nn.init.constant_(self.att[:, node_att_dim:], edge_scale)

        with torch.no_grad():
            self.cos_weight.fill_(0.10)
        nn.init.xavier_uniform_(self.edge_mlp[0].weight)
        nn.init.zeros_(self.edge_mlp[0].bias)
        nn.init.normal_(self.edge_mlp[2].weight, std=0.02)
        nn.init.zeros_(self.edge_mlp[2].bias)

    def _edge_basis(self, edge_attr: Tensor) -> Tensor:
        if self.edge_dim == 0 or edge_attr.size(1) == 0:
            return edge_attr.new_zeros((edge_attr.size(0), 2))
        if self.edge_dim >= 4 and edge_attr.size(1) >= 4:
            s = edge_attr[:, 0:1]
            h = edge_attr[:, 1:2]
            r = edge_attr[:, 2:3]
            c = edge_attr[:, 3:4]
            parts = [s, h, r, c, s * h, s.abs()]
            if edge_attr.size(1) > 4:
                parts.append(edge_attr[:, 4:])
            return torch.cat(parts, dim=-1)
        if self.edge_dim == 2 and edge_attr.size(1) >= 2:
            s = edge_attr[:, 0:1]
            h = edge_attr[:, 1:2]
            return torch.cat([s, h, s * h, s.abs()], dim=-1)
        return torch.cat([edge_attr, edge_attr.abs()], dim=-1)

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

            z_gate = F.layer_norm(z, (self.hidden_dim,))
            node_att_dim = 2 * self.hidden_dim
            node_cat = torch.cat([z_gate[src], z_gate[dst]], dim=-1)
            node_score = (node_cat * self.att[:, :node_att_dim]).sum(dim=-1) / math.sqrt(
                float(max(1, node_att_dim))
            )
            cos_sim = F.cosine_similarity(z_gate[src], z_gate[dst], dim=-1, eps=1e-6)
            edge_delta = self.edge_mlp(self._edge_basis(edge_attr))
            if self.edge_dim > 0 and edge_attr.size(1) > 0:
                edge_score = (edge_attr * self.att[:, node_att_dim:]).sum(dim=-1) + edge_delta[:, 0]
            else:
                edge_score = edge_delta[:, 0]

            score = node_score + self.cos_weight * cos_sim + edge_score
            gamma = torch.tanh(score) if self.gate_mode == "tanh" else torch.sigmoid(score)

            s_abs = edge_attr[:, 0].abs() if edge_attr.size(1) >= 1 else gamma.abs()
            h_val = edge_attr[:, 1] if edge_attr.size(1) >= 2 else torch.zeros_like(s_abs)
            alpha_int = torch.sigmoid(0.8 * s_abs + 0.8 * h_val + edge_delta[:, 1])

            edge_weights_2ch = torch.stack([edge_norm * gamma, edge_norm * alpha_int], dim=-1)
            m = self.propagate(
                edge_index,
                z=z,
                edge_weights_2ch=edge_weights_2ch,
                size=(num_nodes, num_nodes),
            )
        else:
            m = torch.zeros_like(z)
            gamma = z.new_zeros((0,))

        raw = 0.5 * h + self.eps * h_0 + m + self.lin_root(h_root_broadcast)
        sym_act = raw + 0.2 * torch.tanh(raw)
        out = self.dropout(self.norm(sym_act))
        return out, gamma

    def message(
        self,
        z_j: Tensor,
        edge_weights_2ch: Tensor,
    ) -> Tensor:
        if not self.use_dual_channel:
            return edge_weights_2ch[:, 0:1] * z_j
        m_pol = edge_weights_2ch[:, 0:1] * z_j[:, : self.pol_dim]
        m_exc = edge_weights_2ch[:, 1:2] * z_j[:, self.pol_dim :]
        return torch.cat([m_pol, m_exc], dim=-1)


def _ste_parent_sign(u_ori: Tensor, neutral_eps: float) -> Tensor:
    """Piecewise-linear parent sign ramp: exact +/-1 outside neutral band, smooth u/eps inside."""
    eps_safe = max(float(neutral_eps), 1e-6)
    return torch.clamp(u_ori / eps_safe, -1.0, 1.0)


def _propagate_root_aligned_edge_signal(
    edge_index_td: Tensor,
    edge_sig: Tensor,
    root_idx: Tensor,
    num_nodes: int,
    neutral_eps: float = 0.1,
) -> Tensor:
    """Project local parent->reply gate/stance signals into root-aligned orientation via BFS."""
    if num_nodes <= 0 or edge_index_td.numel() == 0 or edge_sig.numel() == 0:
        return edge_sig

    device = edge_sig.device
    src = edge_index_td[0].long()
    dst = edge_index_td[1].long()

    valid_roots = root_idx.long()
    valid_roots = valid_roots[(valid_roots >= 0) & (valid_roots < num_nodes)]
    if valid_roots.numel() == 0:
        return edge_sig

    node_ori = edge_sig.new_zeros((num_nodes,))
    node_ori = node_ori.scatter(
        0, valid_roots, torch.ones_like(valid_roots, dtype=edge_sig.dtype)
    )
    visited = torch.zeros((num_nodes,), dtype=torch.bool, device=device)
    visited = visited.scatter(
        0, valid_roots, torch.ones_like(valid_roots, dtype=torch.bool)
    )

    max_hops = min(int(num_nodes), 64)
    for _ in range(max_hops):
        frontier = visited[src] & (~visited[dst])
        if not bool(frontier.any()):
            break
        f_src = src[frontier]
        f_dst = dst[frontier]
        # Detach parent orientation so BFS acts as bounded signed routing in [-1, 1]
        # without multiplying 1/eps Jacobians exponentially across deep chains.
        p_sign = _ste_parent_sign(node_ori[f_src].detach(), neutral_eps)
        aligned_f = p_sign * edge_sig[frontier]
        updated = scatter(aligned_f, f_dst, dim=0, dim_size=num_nodes, reduce="mean")
        new_vis = torch.zeros((num_nodes,), dtype=torch.bool, device=device).scatter(
            0, f_dst, torch.ones_like(f_dst, dtype=torch.bool)
        )
        node_ori = torch.where(new_vis, updated, node_ori)
        visited = visited | new_vis

    p_sign_all = _ste_parent_sign(node_ori[src].detach(), neutral_eps)
    return p_sign_all * edge_sig


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
        hidden_dim: int = 64,
        num_layers: int = 2,
        edge_dim: int = 2,
        num_rumor_classes: int = 4,
        eps: float = 0.5,
        dropout: float = 0.1,
        gate_mode: Literal["tanh", "sigmoid"] = "tanh",
        use_bottom_up: bool = True,
        ablate_stance: bool = False,
        ablate_hostility: bool = False,
        ablate_irrationality: bool = False,
        ablate_constructive: bool = False,
        use_latent_edge_mlp: bool = False,
        readout_mode: Literal["attention", "mean"] = "attention",
        use_cascade_stats: bool = False,
        cascade_feat_dim: int = 8,
        predict_uncertainty: bool = False,
        use_dir_mix: bool = True,
        use_camp_dipole: bool = True,
        use_dual_channel: bool = True,
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
        self.ablate_irrationality = ablate_irrationality
        self.ablate_constructive = ablate_constructive
        self.use_latent_edge_mlp = use_latent_edge_mlp
        self.gate_mode = gate_mode
        self.readout_mode = readout_mode
        self.use_cascade_stats = bool(use_cascade_stats)
        self.cascade_feat_dim = int(cascade_feat_dim)
        self.predict_uncertainty = bool(predict_uncertainty)
        self.use_dir_mix = bool(use_dir_mix)
        self.use_camp_dipole = bool(use_camp_dipole)
        self.use_dual_channel = bool(use_dual_channel)

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

        self.td_convs = nn.ModuleList(
            [
                StanceGuidedFAConv(
                    hidden_dim=hidden_dim,
                    edge_dim=edge_dim,
                    eps=eps,
                    gate_mode=gate_mode,
                    dropout=dropout,
                    use_dual_channel=use_dual_channel,
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
                        use_dual_channel=use_dual_channel,
                    )
                    for _ in range(num_layers)
                ]
            )
            if use_bottom_up
            else None
        )
        if use_bottom_up and use_dir_mix:
            self.dir_mix_td: nn.ModuleList | None = nn.ModuleList(
                [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers)]
            )
            self.dir_mix_bu: nn.ModuleList | None = nn.ModuleList(
                [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers)]
            )
            self.dir_mix_scale: nn.Parameter | None = nn.Parameter(
                torch.full((num_layers,), 0.10)
            )
        else:
            self.dir_mix_td = None
            self.dir_mix_bu = None
            self.dir_mix_scale = None

        self.td_readout = (
            GlobalAttentionReadout(hidden_dim) if readout_mode == "attention" else None
        )
        self.bu_readout = (
            GlobalAttentionReadout(hidden_dim)
            if (use_bottom_up and readout_mode == "attention")
            else None
        )
        if self.use_camp_dipole:
            dip_hid = max(16, hidden_dim // 2)
            self.dipole_proj: nn.Module | None = nn.Sequential(
                nn.Linear(3 * hidden_dim + 6, dip_hid),
                nn.LayerNorm(dip_hid),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dip_hid, 2 * hidden_dim),
            )
            self.dipole_scale: nn.Parameter | None = nn.Parameter(torch.tensor(0.25))
        else:
            self.dipole_proj = None
            self.dipole_scale = None
        self.readout_norm = nn.LayerNorm(2 * hidden_dim)
        self.head = MultiTaskHead(
            2 * hidden_dim,
            hidden_dim,
            num_rumor_classes,
            dropout,
            cascade_feat_dim=self.cascade_feat_dim if self.use_cascade_stats else 0,
            predict_uncertainty=self.predict_uncertainty,
            mask_semantic_stats=bool(self.ablate_stance or self.ablate_hostility),
        )

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

        if (
            self.ablate_stance
            or self.ablate_hostility
            or self.ablate_irrationality
            or self.ablate_constructive
        ):
            edge_attr = edge_attr.clone()
            if self.ablate_stance and edge_attr.size(1) >= 1:
                edge_attr[:, 0] = 0.0
            if self.ablate_hostility and edge_attr.size(1) >= 2:
                edge_attr[:, 1] = 0.0
            if self.ablate_irrationality and edge_attr.size(1) >= 3:
                edge_attr[:, 2] = 0.0
            if self.ablate_constructive and edge_attr.size(1) >= 4:
                edge_attr[:, 3] = 0.0

        return edge_attr

    def _encode_graphs(
        self, data: Any
    ) -> tuple[dict[str, Tensor], list[Tensor], list[Tensor]]:
        x = data.x.float()
        num_nodes = x.size(0)
        root_idx, batch_vec = resolve_root_indices(data, num_nodes)
        num_graphs = root_idx.size(0)

        edge_index_td = getattr(data, "edge_index_td", data.edge_index)
        edge_index_bu = getattr(
            data,
            "edge_index_bu",
            edge_index_td[[1, 0], :] if edge_index_td.numel() > 0 else edge_index_td,
        )
        edge_attr = self._prepare_edge_attr(x, edge_index_td, data)

        h0 = self.input_proj(x, data, root_idx)
        h_root_0 = broadcast_root_features(h0, root_idx, batch_vec)

        h_td = h0
        h_bu = h0
        gamma_td_layers: list[Tensor] = []
        gamma_bu_layers: list[Tensor] = []

        if self.use_bottom_up and self.bu_convs is not None:
            for l_idx in range(self.num_layers):
                h_td_next, gamma_td = self.td_convs[l_idx](
                    h_td, h0, h_root_0, edge_index_td, edge_attr
                )
                h_bu_next, gamma_bu = self.bu_convs[l_idx](
                    h_bu, h0, h_root_0, edge_index_bu, edge_attr
                )
                gamma_td_layers.append(gamma_td)
                gamma_bu_layers.append(gamma_bu)
                if (
                    self.use_dir_mix
                    and self.dir_mix_td is not None
                    and self.dir_mix_bu is not None
                    and self.dir_mix_scale is not None
                ):
                    mix_scale = torch.tanh(self.dir_mix_scale[l_idx])
                    h_td = h_td_next + mix_scale * self.dir_mix_td[l_idx](h_bu_next)
                    h_bu = h_bu_next + mix_scale * self.dir_mix_bu[l_idx](h_td_next)
                else:
                    h_td = h_td_next
                    h_bu = h_bu_next

            node_emb = 0.5 * (h_td + h_bu)
            z_root = 0.5 * (h_td[root_idx] + h_bu[root_idx])
            if self.td_readout is not None and self.bu_readout is not None:
                z_att = 0.5 * (
                    self.td_readout(h_td, batch_vec, num_graphs)
                    + self.bu_readout(h_bu, batch_vec, num_graphs)
                )
            else:
                z_att = global_mean_pool(node_emb, batch_vec, size=num_graphs)
        else:
            for conv in self.td_convs:
                h_td, gamma_td = conv(h_td, h0, h_root_0, edge_index_td, edge_attr)
                gamma_td_layers.append(gamma_td)
            node_emb = h_td
            z_root = h_td[root_idx]
            z_att = (
                self.td_readout(h_td, batch_vec, num_graphs)
                if self.td_readout is not None
                else global_mean_pool(h_td, batch_vec, size=num_graphs)
            )

        base_emb = torch.cat([z_root, z_att], dim=-1)
        if self.use_camp_dipole and self.dipole_proj is not None and self.dipole_scale is not None:
            node_cnt = scatter(
                x.new_ones((num_nodes, 1), dtype=node_emb.dtype),
                batch_vec,
                dim=0,
                dim_size=num_graphs,
                reduce="sum",
            )
            mean_h = global_mean_pool(node_emb, batch_vec, size=num_graphs)
            var_h = global_mean_pool((node_emb - mean_h[batch_vec]).pow(2), batch_vec, size=num_graphs)
            z_disp = torch.where(node_cnt > 1.0, torch.sqrt(var_h + 1e-6), torch.zeros_like(var_h))

            if edge_index_td.numel() > 0:
                dst = edge_index_td[1]
                edge_batch = batch_vec[dst]
                h_dst = node_emb[dst]
                gamma_td_last = gamma_td_layers[-1]
                # Center sigmoid gates (0, 1) -> (-1, 1) and run identical root-aligned BFS so
                # w/o Conflict Gate (Sigmoid) isolates ONLY the signed message-passing gate.
                raw_camp_sig = gamma_td_last if self.gate_mode == "tanh" else (2.0 * gamma_td_last - 1.0)
                camp_sig = _propagate_root_aligned_edge_signal(
                    edge_index_td,
                    raw_camp_sig,
                    root_idx,
                    num_nodes,
                    neutral_eps=0.1,
                )

                w_pos = F.relu(camp_sig).unsqueeze(-1)
                w_neg = F.relu(-camp_sig).unsqueeze(-1)
                sum_pos = scatter(w_pos, edge_batch, dim=0, dim_size=num_graphs, reduce="sum")
                sum_neg = scatter(w_neg, edge_batch, dim=0, dim_size=num_graphs, reduce="sum")
                ones_e = x.new_ones((edge_index_td.size(1), 1), dtype=node_emb.dtype)
                edge_cnt = scatter(ones_e, edge_batch, dim=0, dim_size=num_graphs, reduce="sum")
                has_edges = edge_cnt > 0.0

                mu_pos = scatter(w_pos * h_dst, edge_batch, dim=0, dim_size=num_graphs, reduce="sum") / (
                    sum_pos + 1.0
                )
                mu_neg = scatter(w_neg * h_dst, edge_batch, dim=0, dim_size=num_graphs, reduce="sum") / (
                    sum_neg + 1.0
                )
                denom_e = edge_cnt.clamp(min=1.0)
                frac_pos = torch.where(
                    has_edges, (sum_pos / denom_e).clamp(0.0, 1.0), torch.zeros_like(sum_pos)
                )
                frac_neg = torch.where(
                    has_edges, (sum_neg / denom_e).clamp(0.0, 1.0), torch.zeros_like(sum_neg)
                )
                camp_balance = 4.0 * frac_pos * frac_neg

                diff_camp = mu_pos - mu_neg
                z_dipole_polar = camp_balance * torch.sqrt(diff_camp.pow(2) + 1e-8)
                z_dipole_signed = diff_camp

                g_col = raw_camp_sig.unsqueeze(-1)
                g_sum = scatter(g_col, edge_batch, dim=0, dim_size=num_graphs, reduce="sum")
                g_sq_sum = scatter(g_col.pow(2), edge_batch, dim=0, dim_size=num_graphs, reduce="sum")
                g_abs_sum = scatter(
                    torch.sqrt(g_col.pow(2) + 1e-8), edge_batch, dim=0, dim_size=num_graphs, reduce="sum"
                )
                g_mean = torch.where(has_edges, g_sum / denom_e, torch.zeros_like(g_sum))
                g_sq_mean = torch.where(has_edges, g_sq_sum / denom_e, torch.zeros_like(g_sq_sum))
                g_abs_mean = torch.where(has_edges, g_abs_sum / denom_e, torch.zeros_like(g_abs_sum))
                g_var = (g_sq_mean - g_mean.pow(2)).clamp(min=0.0)
                g_std = torch.where(edge_cnt >= 2.0, torch.sqrt(g_var + 1e-8), torch.zeros_like(g_var))
                g_mom = torch.cat([g_mean, g_std, g_abs_mean, frac_pos, frac_neg, camp_balance], dim=-1)
            else:
                z_dipole_polar = node_emb.new_zeros((num_graphs, self.hidden_dim))
                z_dipole_signed = node_emb.new_zeros((num_graphs, self.hidden_dim))
                g_mom = node_emb.new_zeros((num_graphs, 6))

            dip_in = torch.cat([z_disp, z_dipole_polar, z_dipole_signed, g_mom], dim=-1)
            graph_emb = self.readout_norm(base_emb + torch.tanh(self.dipole_scale) * self.dipole_proj(dip_in))
        else:
            graph_emb = self.readout_norm(base_emb)
        c_stats = getattr(data, "cascade_stats", None) if self.use_cascade_stats else None
        outputs = self.head(graph_emb, node_emb, cascade_stats=c_stats)
        return outputs, gamma_td_layers, gamma_bu_layers

    def forward(self, data: Any) -> dict[str, Tensor]:
        outputs, _, _ = self._encode_graphs(data)
        return outputs

    @torch.no_grad()
    def predict_with_diagnostics(self, data: Any, *, top_k_edges: int = 5) -> dict[str, Any]:
        """Run inference and emit interpretable forecasts, confidence flags, and trigger edges."""
        was_training = self.training
        self.eval()
        try:
            outputs, gamma_td_layers, gamma_bu_layers = self._encode_graphs(data)
        finally:
            self.train(was_training)

        y_size = outputs["y_size_pred"].detach().cpu()
        y_polar = outputs["y_polar_pred"].detach().cpu()
        rumor_probs = torch.softmax(outputs["rumor_logits"].detach().cpu(), dim=-1)

        if "confidence_score" in outputs:
            conf = outputs["confidence_score"].detach().cpu()
        else:
            conf = torch.full_like(y_polar, 0.75)

        c_stats = getattr(data, "cascade_stats", None)
        if c_stats is not None and c_stats.numel() >= 8:
            feat_w = c_stats.size(-1) if c_stats.dim() >= 2 else max(8, self.cascade_feat_dim)
            stats_2d = c_stats.detach().cpu().float().view(-1, feat_w)
            substantive_ratio = stats_2d[:, 6]
            one_sided_host = stats_2d[:, 7]
        else:
            substantive_ratio = torch.ones_like(y_polar)
            one_sided_host = torch.zeros_like(y_polar)

        low_signal = (substantive_ratio < 0.35) | (conf < 0.45)

        gamma_td = gamma_td_layers[-1].detach().cpu() if gamma_td_layers else torch.zeros(0)
        gamma_bu = gamma_bu_layers[-1].detach().cpu() if gamma_bu_layers else torch.zeros_like(gamma_td)
        edge_imp = 0.5 * (gamma_td.abs() + gamma_bu.abs()) if gamma_bu_layers else gamma_td.abs()
        top_edges: list[dict[str, Any]] = []
        if edge_imp.numel() > 0:
            raw_attr = getattr(data, "edge_attr", None)
            attr_cpu = raw_attr.detach().cpu() if raw_attr is not None else None
            order = torch.argsort(edge_imp, descending=True)[: max(1, int(top_k_edges))]
            edge_pairs = getattr(data, "edge_pairs", None)
            flat_edge_pairs: list[Any] | None = None
            if isinstance(edge_pairs, (list, tuple)):
                if edge_pairs and isinstance(edge_pairs[0], list):
                    flat_edge_pairs = [pair for sub in edge_pairs for pair in sub]
                else:
                    flat_edge_pairs = list(edge_pairs)
            for idx_t in order:
                e_i = int(idx_t.item())
                item: dict[str, Any] = {
                    "edge_idx": e_i,
                    "importance": float(edge_imp[e_i].item()),
                    "gamma_td": float(gamma_td[e_i].item()),
                }
                if flat_edge_pairs is not None and e_i < len(flat_edge_pairs):
                    item["pair"] = flat_edge_pairs[e_i]
                if attr_cpu is not None and e_i < attr_cpu.size(0):
                    item["s_uv"] = float(attr_cpu[e_i, 0].item())
                    if attr_cpu.size(1) >= 2:
                        item["h_uv"] = float(attr_cpu[e_i, 1].item())
                    if attr_cpu.size(1) >= 3:
                        item["r_uv"] = float(attr_cpu[e_i, 2].item())
                    if attr_cpu.size(1) >= 4:
                        item["c_uv"] = float(attr_cpu[e_i, 3].item())
                top_edges.append(item)

        return {
            "y_size_pred": y_size,
            "predicted_total_replies": torch.expm1(y_size).clamp(min=0.0),
            "y_polar_pred": y_polar,
            "rumor_probs": rumor_probs,
            "confidence_score": conf,
            "substantive_text_ratio": substantive_ratio,
            "one_sided_hostility_index": one_sided_host,
            "low_signal_warning": low_signal,
            "top_trigger_edges": top_edges,
        }

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
