"""CLI and benchmark suite for Tier 2 Macro Signed User-Interaction Social Network.

Supports:
  1. Sparse CPU macro network polarization analytics (Structural Balance Theory
     + degree-preserving permutation z-scores, Signed Laplacian spectral bipartition,
     Signed Modularity Q_signed, Garimella Random Walk Controversy RWC, and fig5).
  2. Multi-seed Zero-Leakage Chronological Signed Link Prediction GNN benchmark
     comparing UserTextMLP, StandardGCN, StandardGATv2, SGCN (Derr et al., ICDM 2018),
     FAGCN (Bo et al., AAAI 2021), and SBiFAGAT-Link (bidirectional StanceGuidedFAConv).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch import Tensor
from torch_geometric.nn import FAConv, GATv2Conv, GCNConv, SignedConv
from torch_geometric.utils import to_undirected

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.user_network import (
    build_temporal_signed_link_dataset,
    compute_macro_signed_network_stats,
    format_macro_network_latex,
    plot_macro_signed_network,
)
from src.models.sbi_fagat import StanceGuidedFAConv
from src.training.reporting import export_latex_table, serialize_benchmark_json
from src.training.trainer import (
    compute_paired_bootstrap_pvalue,
    compute_paired_ttest,
    holm_bonferroni_adjust,
)
from src.utils.seed import OFFICIAL_SEEDS, seed_everything

LINK_PRED_METRICS: tuple[tuple[str, str, bool], ...] = (
    ("roc_auc", "ROC-AUC $\\uparrow$", False),
    ("pr_auc", "PR-AUC $\\uparrow$", False),
    ("macro_f1", "Macro-F1 $\\uparrow$", False),
    ("trans_roc_auc", "Trans.\\ ROC-AUC $\\uparrow$", False),
    ("trans_pr_auc", "Trans.\\ PR-AUC $\\uparrow$", False),
    ("trans_macro_f1", "Trans.\\ Macro-F1 $\\uparrow$", False),
)

LINK_PRED_MODEL_NAMES: tuple[str, ...] = (
    "UserTextMLP",
    "StandardGCN",
    "StandardGATv2",
    "SGCN",
    "FAGCN",
    "SBiFAGAT-Link",
)

_MODEL_TIE_PRIORITY: dict[str, int] = {
    "SGCN": 0,
    "FAGCN": 1,
    "StandardGATv2": 2,
    "StandardGCN": 3,
    "UserTextMLP": 4,
}


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

    def encode(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        del edge_index, edge_attr
        h0 = self.stem(x)
        return h0 + self.encoder(h0)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, pairs: Tensor) -> Tensor:
        return self.head(self.encode(x, edge_index, edge_attr), pairs)


class StandardGCNLink(nn.Module):
    """Unsigned undirected GCN encoder (Kipf & Welling, ICLR 2017) for link sign prediction."""

    def __init__(self, in_dim: int = 772, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.stem = UserNodeStem(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.drop = nn.Dropout(dropout)
        self.head = PairwiseSignedHead(hidden_dim, dropout=dropout)

    def encode(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        del edge_attr
        num_nodes = x.size(0)
        ei = to_undirected(edge_index, num_nodes=num_nodes) if edge_index.numel() > 0 else edge_index
        h = self.stem(x)
        for conv, norm in zip(self.convs, self.norms):
            h = h + self.drop(norm(F.relu(conv(h, ei))))
        return h

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, pairs: Tensor) -> Tensor:
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
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})")
        self.stem = UserNodeStem(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
        head_dim = hidden_dim // heads
        self.convs = nn.ModuleList(
            [GATv2Conv(hidden_dim, head_dim, heads=heads, concat=True, dropout=dropout, add_self_loops=True) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.drop = nn.Dropout(dropout)
        self.head = PairwiseSignedHead(hidden_dim, dropout=dropout)

    def encode(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        del edge_attr
        num_nodes = x.size(0)
        ei = to_undirected(edge_index, num_nodes=num_nodes) if edge_index.numel() > 0 else edge_index
        h = self.stem(x)
        for conv, norm in zip(self.convs, self.norms):
            h = h + self.drop(norm(F.elu(conv(h, ei))))
        return h

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, pairs: Tensor) -> Tensor:
        return self.head(self.encode(x, edge_index, edge_attr), pairs)


class SGCNLink(nn.Module):
    """Signed Graph Convolutional Network (Derr et al., ICDM 2018) with balanced/unbalanced paths."""

    def __init__(self, in_dim: int = 772, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_dim % 2 != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be even for SGCN")
        half = hidden_dim // 2
        self.stem = UserNodeStem(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
        convs = [SignedConv(hidden_dim, half, first_aggr=True)]
        for _ in range(max(0, num_layers - 1)):
            convs.append(SignedConv(half, half, first_aggr=False))
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(len(convs))])
        self.drop = nn.Dropout(dropout)
        self.head = PairwiseSignedHead(hidden_dim, dropout=dropout)

    def encode(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        num_nodes = x.size(0)
        if edge_index.numel() > 0 and edge_attr.numel() > 0:
            signs = edge_attr[:, 0]
            pos_ei = edge_index[:, signs >= 0.0]
            neg_ei = edge_index[:, signs < 0.0]
            pos_ei = to_undirected(pos_ei, num_nodes=num_nodes) if pos_ei.numel() > 0 else pos_ei
            neg_ei = to_undirected(neg_ei, num_nodes=num_nodes) if neg_ei.numel() > 0 else neg_ei
        else:
            pos_ei = edge_index.new_empty((2, 0))
            neg_ei = edge_index.new_empty((2, 0))

        h = self.stem(x)
        for conv, norm in zip(self.convs, self.norms):
            h = h + self.drop(norm(torch.tanh(conv(h, pos_ei, neg_ei))))
        return h

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, pairs: Tensor) -> Tensor:
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
    ) -> None:
        super().__init__()
        self.stem = UserNodeStem(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.convs = nn.ModuleList(
            [FAConv(channels=hidden_dim, eps=eps, dropout=dropout, add_self_loops=True) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.drop = nn.Dropout(dropout)
        self.head = PairwiseSignedHead(hidden_dim, dropout=dropout)

    def encode(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        del edge_attr
        num_nodes = x.size(0)
        ei = to_undirected(edge_index, num_nodes=num_nodes) if edge_index.numel() > 0 else edge_index
        h0 = self.stem(x)
        h = h0
        for conv, norm in zip(self.convs, self.norms):
            h = h + self.drop(norm(F.elu(conv(h, h0, ei))))
        return h

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, pairs: Tensor) -> Tensor:
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
    ) -> None:
        super().__init__()
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

    def encode(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        h0 = self.stem(x)
        h = h0
        rev_ei = edge_index[[1, 0]] if edge_index.numel() > 0 else edge_index
        for conv_td, conv_bu, mix in zip(self.convs_td, self.convs_bu, self.mix_projs):
            h_td, _ = conv_td(h, h0, h0, edge_index, edge_attr)
            h_bu, _ = conv_bu(h, h0, h0, rev_ei, edge_attr)
            h = h + mix(torch.cat([h_td, h_bu], dim=-1))
        return h

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, pairs: Tensor) -> Tensor:
        return self.head(self.encode(x, edge_index, edge_attr), pairs)


def build_link_model(
    model_name: str,
    *,
    in_dim: int,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
) -> nn.Module:
    """Instantiate a signed link prediction architecture by canonical name."""
    if model_name == "UserTextMLP":
        return UserTextMLP(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
    if model_name == "StandardGCN":
        return StandardGCNLink(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    if model_name == "StandardGATv2":
        return StandardGATv2Link(
            in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers, heads=min(4, max(1, hidden_dim // 16)), dropout=dropout
        )
    if model_name == "SGCN":
        return SGCNLink(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    if model_name == "FAGCN":
        return FAGCNLink(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    if model_name == "SBiFAGAT-Link":
        return SBiFAGATLink(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    raise ValueError(f"Unknown link prediction model '{model_name}'")


def _safe_binary_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> tuple[float, float, float]:
    """Compute ROC-AUC, PR-AUC, and Macro-F1 safely for finite/non-empty splits."""
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float("nan"), float("nan"), float("nan")
    roc = float(roc_auc_score(y_true, probs))
    pr = float(average_precision_score(y_true, probs))
    preds = (probs >= float(threshold)).astype(np.int64)
    mf1 = float(f1_score(y_true.astype(np.int64), preds, average="macro", zero_division=0))
    return roc, pr, mf1


def _find_best_val_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Select binary threshold strictly on Val probabilities to maximize Macro-F1."""
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.5
    best_th = 0.5
    best_f1 = -1.0
    for th in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        score = float(
            f1_score(y_true.astype(np.int64), (probs >= th).astype(np.int64), average="macro", zero_division=0)
        )
        if score > best_f1:
            best_f1 = score
            best_th = float(th)
    return best_th


@torch.no_grad()
def evaluate_link_split(
    model: nn.Module,
    x: Tensor,
    mp_edge_index: Tensor,
    mp_edge_attr: Tensor,
    split: Mapping[str, Tensor],
    *,
    threshold: float = 0.5,
    tune_threshold: bool = False,
) -> tuple[dict[str, float], float]:
    """Evaluate all temporal edges and the transductive both-users-seen subset."""
    model.eval()
    logits = model(x, mp_edge_index, mp_edge_attr, split["pairs"])
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    y_true = split["y_oppose"].detach().cpu().numpy()
    both_seen = split["both_seen"].detach().cpu().numpy().astype(bool)

    used_th = _find_best_val_threshold(y_true, probs) if tune_threshold else float(threshold)
    roc, pr, mf1 = _safe_binary_metrics(y_true, probs, used_th)
    t_roc, t_pr, t_mf1 = _safe_binary_metrics(y_true[both_seen], probs[both_seen], used_th)
    return {
        "roc_auc": roc,
        "pr_auc": pr,
        "macro_f1": mf1,
        "trans_roc_auc": t_roc,
        "trans_pr_auc": t_pr,
        "trans_macro_f1": t_mf1,
    }, used_th


def _summarize_seed_metrics(runs: Sequence[Mapping[str, float]]) -> dict[str, tuple[float, float]]:
    """Compute (mean, ddof=1 std) across seed runs."""
    keys = [k for k, _, _ in LINK_PRED_METRICS]
    summary: dict[str, tuple[float, float]] = {}
    for k in keys:
        vals = [float(r.get(k, float("nan"))) for r in runs]
        finite = [v for v in vals if math.isfinite(v)]
        if not finite:
            summary[k] = (float("nan"), float("nan"))
        elif len(finite) == 1:
            summary[k] = (finite[0], 0.0)
        else:
            arr = np.asarray(finite, dtype=np.float64)
            summary[k] = (float(arr.mean()), float(arr.std(ddof=1)))
    return summary


def _select_best_val_comparator(
    val_summary_stats: Mapping[str, Mapping[str, tuple[float, float]]],
    *,
    reference_model: str = "SBiFAGAT-Link",
) -> str:
    """Pick the strongest competing model strictly by average Val rank (zero Test peeking)."""
    candidates = [m for m in val_summary_stats.keys() if m != reference_model]
    if not candidates:
        return "SGCN"
    ranks: dict[str, list[float]] = {c: [] for c in candidates}
    for key, _, _ in LINK_PRED_METRICS:
        scored = [
            (c, float(val_summary_stats[c][key][0]))
            for c in candidates
            if key in val_summary_stats[c] and math.isfinite(float(val_summary_stats[c][key][0]))
        ]
        if len(scored) < 2:
            continue
        scored.sort(key=lambda kv: kv[1], reverse=True)
        for rank_idx, (c, _) in enumerate(scored):
            ranks[c].append(float(rank_idx))
    if not any(ranks.values()):
        return min(candidates, key=lambda c: (_MODEL_TIE_PRIORITY.get(c, 99), c))
    return min(
        candidates,
        key=lambda c: (
            float(np.mean(ranks[c])) if ranks[c] else float("inf"),
            _MODEL_TIE_PRIORITY.get(c, 99),
            c,
        ),
    )


def _train_single_seed(
    model_name: str,
    dataset: Mapping[str, Any],
    *,
    seed: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    epochs: int,
    lr: float,
    patience: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float]]:
    """Train one model for one seed with strict intra-batch unordered-pair masking."""
    seed_everything(seed)
    x = dataset["x"].to(device)
    train_split = {k: v.to(device) for k, v in dataset["train"].items()}
    val_split = {k: v.to(device) for k, v in dataset["val"].items()}
    test_split = {k: v.to(device) for k, v in dataset["test"].items()}

    train_pairs = train_split["pairs"]
    train_attr = train_split["edge_attr"]
    train_y = train_split["y_oppose"]
    num_users = int(dataset["num_users"])
    num_train = int(train_y.size(0))

    # Precompute unordered pair IDs so mini-batch supervision pairs are masked out
    # of message passing (prevents reading the target edge's own [s_uv, h_uv] attr).
    pair_lo = torch.minimum(train_pairs[0], train_pairs[1])
    pair_hi = torch.maximum(train_pairs[0], train_pairs[1])
    train_pair_keys = pair_lo * num_users + pair_hi

    n_pos = float((train_y > 0.5).sum().item())
    n_neg = float((train_y <= 0.5).sum().item())
    pos_weight_val = min(10.0, max(0.1, n_neg / max(1.0, n_pos))) if n_pos > 0 else 1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device))

    model = build_link_model(
        model_name,
        in_dim=int(dataset["in_dim"]),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_score = -1.0
    best_state: dict[str, Tensor] | None = None
    best_threshold = 0.5
    epochs_no_improve = 0
    gen = torch.Generator(device="cpu").manual_seed(seed)

    eff_batch = (
        min(max(1, batch_size), max(1, num_train // 4))
        if num_train >= 8
        else max(1, batch_size)
    )
    for _epoch in range(1, max(1, epochs) + 1):
        model.train()
        perm = torch.randperm(num_train, generator=gen).to(device)
        for start in range(0, num_train, eff_batch):
            b_idx = perm[start : start + eff_batch]
            batch_keys = torch.unique(train_pair_keys[b_idx])
            keep_mask = ~torch.isin(train_pair_keys, batch_keys)
            mp_ei = train_pairs[:, keep_mask]
            mp_ea = train_attr[keep_mask]

            optimizer.zero_grad(set_to_none=True)
            logits = model(x, mp_ei, mp_ea, train_pairs[:, b_idx])
            loss = criterion(logits, train_y[b_idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        val_metrics, val_th = evaluate_link_split(
            model, x, train_pairs, train_attr, val_split, tune_threshold=True
        )
        score = float(val_metrics.get("roc_auc", float("nan")))
        if not math.isfinite(score):
            score = float(val_metrics.get("macro_f1", -1.0))
        if best_state is None or (math.isfinite(score) and score > best_score + 1e-4):
            best_score = score
            best_threshold = val_th
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= max(1, patience):
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    final_val, best_threshold = evaluate_link_split(
        model, x, train_pairs, train_attr, val_split, tune_threshold=True
    )
    final_test, _ = evaluate_link_split(
        model, x, train_pairs, train_attr, test_split, threshold=best_threshold, tune_threshold=False
    )
    return final_val, final_test


def run_signed_link_benchmark(
    data_dir: Path,
    *,
    dataset_name: str | None = None,
    models: Sequence[str] = LINK_PRED_MODEL_NAMES,
    seeds: Sequence[int] = OFFICIAL_SEEDS,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    epochs: int = 40,
    lr: float = 1e-3,
    patience: int = 8,
    batch_size: int = 1024,
    stance_threshold: float = 0.1,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run multi-seed chronological signed link prediction benchmark."""
    run_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_temporal_signed_link_dataset(
        data_dir,
        dataset_name=dataset_name,
        stance_threshold=stance_threshold,
    )
    raw_val: dict[str, list[dict[str, float]]] = {m: [] for m in models}
    raw_test: dict[str, list[dict[str, float]]] = {m: [] for m in models}

    for model_name in models:
        for s in seeds:
            v_m, t_m = _train_single_seed(
                model_name,
                dataset,
                seed=int(s),
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                epochs=epochs,
                lr=lr,
                patience=patience,
                batch_size=batch_size,
                device=run_device,
            )
            raw_val[model_name].append(v_m)
            raw_test[model_name].append(t_m)
            print(
                f"  [{dataset['dataset']}] {model_name} seed={s}: "
                f"Val ROC-AUC={v_m['roc_auc']:.4f} | Test ROC-AUC={t_m['roc_auc']:.4f} "
                f"PR-AUC={t_m['pr_auc']:.4f} F1={t_m['macro_f1']:.4f}",
                flush=True,
            )

    val_summary = {m: _summarize_seed_metrics(raw_val[m]) for m in models}
    test_summary = {m: _summarize_seed_metrics(raw_test[m]) for m in models}

    ref_model = "SBiFAGAT-Link" if "SBiFAGAT-Link" in models else str(models[-1])
    cmp_model = _select_best_val_comparator(val_summary, reference_model=ref_model)

    p_values: dict[str, float] = {}
    boot_p_values: dict[str, float] = {}
    if ref_model in raw_test and cmp_model in raw_test and len(seeds) >= 2:
        for key, _, _ in LINK_PRED_METRICS:
            a_vals = [float(r[key]) for r in raw_test[ref_model]]
            b_vals = [float(r[key]) for r in raw_test[cmp_model]]
            p_values[key] = compute_paired_ttest(a_vals, b_vals)
            boot_p_values[key] = compute_paired_bootstrap_pvalue(a_vals, b_vals)

    return {
        "dataset": dataset["dataset"],
        "edge_label_source": dataset["edge_label_source"],
        "seeds": [int(s) for s in seeds],
        "reference_model": ref_model,
        "comparator_model": cmp_model,
        "summary_stats": test_summary,
        "val_summary_stats": val_summary,
        "p_values": p_values,
        "adjusted_p_values": holm_bonferroni_adjust(p_values),
        "bootstrap_p_values": boot_p_values,
        "adjusted_bootstrap_p_values": holm_bonferroni_adjust(boot_p_values),
        "nonparametric_test_label": "Exact Perm" if len(seeds) < 6 else "Bootstrap",
        "raw_results": raw_test,
        "val_raw_results": raw_val,
        "split_counts": {
            "num_users": int(dataset["num_users"]),
            "train_users": int(len(dataset["train_user_set"])),
            "train_edges": int(dataset["train"]["y_oppose"].numel()),
            "val_edges": int(dataset["val"]["y_oppose"].numel()),
            "test_edges": int(dataset["test"]["y_oppose"].numel()),
            "test_transductive_edges": int(dataset["test"]["both_seen"].sum().item()),
        },
    }


def _collect_macro_stats_for_table(paper_dir: Path, latest_stats: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Combine any sibling `results_macro_network_*.json` artifacts so tables stay multi-corpus."""
    by_name: dict[str, Mapping[str, Any]] = {}
    for jf in sorted(paper_dir.glob("results_macro_network_*.json")):
        try:
            loaded = json.loads(jf.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(loaded, Mapping) and "dataset" in loaded and "structural_balance" in loaded:
            by_name[str(loaded["dataset"]).lower()] = loaded
    by_name[str(latest_stats["dataset"]).lower()] = latest_stats
    order = {"vn": 0, "pheme": 1}
    return [by_name[k] for k in sorted(by_name.keys(), key=lambda s: (order.get(s, 99), s))]


def run_smoke_test() -> None:
    """Offline <5s synthetic smoke test verifying macro stats + link-pred GNNs end-to-end."""
    from datetime import datetime, timezone

    import pyarrow as pa

    from src.data.schema import EDGES_ARROW_SCHEMA, NODES_ARROW_SCHEMA, save_cascade_parquet

    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "smoke_ds"
        d.mkdir(parents=True, exist_ok=True)
        out_paper = Path(tmp) / "paper"
        out_fig = Path(tmp) / "figs"

        # 2 polarized camps of users across 6 cascades with valid schema & advancing timestamps
        nodes_rows: list[dict[str, Any]] = []
        edges_rows: list[dict[str, Any]] = []
        edge_stance_map: dict[str, float] = {}
        edge_host_map: dict[str, float] = {}
        node_idx = 0
        t_base = 1_700_000_000.0
        for c_id in range(6):
            r_user = f"u_{c_id % 10:02d}"
            t_root = datetime.fromtimestamp(t_base, tz=timezone.utc)
            root_nid = f"n_{node_idx}"
            nodes_rows.append(
                {
                    "node_id": root_nid,
                    "cascade_id": f"c_{c_id}",
                    "topic_id": f"topic_{c_id % 2}",
                    "parent_id": None,
                    "is_root": True,
                    "text": f"Root post {c_id}",
                    "created_at_utc": t_root,
                    "lang": "vi",
                    "source_dataset": "smoke",
                    "user_id_hash": r_user,
                    "rumor_label": None,
                }
            )
            node_idx += 1
            for r_i in range(14):
                t_child_sec = t_base + 60.0 * (r_i + 1)
                t_child = datetime.fromtimestamp(t_child_sec, tz=timezone.utc)
                src_u = f"u_{(c_id * 3 + r_i) % 16:02d}"
                dst_u = r_user if src_u != r_user else "u_17"
                same_camp = (int(src_u.split("_")[1]) % 2) == (int(dst_u.split("_")[1]) % 2)
                s_uv = float(0.65 if same_camp else -0.65)
                h_uv = float(0.10 if same_camp else 0.72)
                nid = f"n_{node_idx}"
                eid = f"e_{c_id}_{r_i}"
                node_idx += 1
                nodes_rows.append(
                    {
                        "node_id": nid,
                        "cascade_id": f"c_{c_id}",
                        "topic_id": f"topic_{c_id % 2}",
                        "parent_id": root_nid,
                        "is_root": False,
                        "text": f"Reply {r_i} in cascade {c_id}",
                        "created_at_utc": t_child,
                        "lang": "vi",
                        "source_dataset": "smoke",
                        "user_id_hash": src_u,
                        "rumor_label": None,
                    }
                )
                edges_rows.append(
                    {
                        "edge_id": eid,
                        "cascade_id": f"c_{c_id}",
                        "source_id": root_nid,
                        "target_id": nid,
                        "created_at_utc": t_child,
                        "time_delay_sec": float(60.0 * (r_i + 1)),
                        "edge_type": "reply",
                    }
                )
                edge_stance_map[eid] = s_uv
                edge_host_map[eid] = h_uv
            t_base += 1200.0

        nodes_tbl = pa.Table.from_pylist(nodes_rows, schema=NODES_ARROW_SCHEMA)
        edges_tbl = pa.Table.from_pylist(edges_rows, schema=EDGES_ARROW_SCHEMA)
        save_cascade_parquet(nodes_tbl, edges_tbl, d, validate=True)

        embs = torch.tensor(rng.normal(0.0, 0.3, size=(len(nodes_rows), 16)), dtype=torch.float32)
        for row_i, row in enumerate(nodes_rows):
            camp = int(row["user_id_hash"].split("_")[1]) % 2
            embs[row_i, camp * 4 : camp * 4 + 4] += 1.5
        torch.save(
            {
                "node_embeddings": embs,
                "node_id_to_idx": {r["node_id"]: i for i, r in enumerate(nodes_rows)},
                "edge_stance": edge_stance_map,
                "edge_hostility": edge_host_map,
                "teacher_edge_stance": edge_stance_map,
                "teacher_edge_hostility": edge_host_map,
            },
            d / "nlp_features.pt",
        )

        stats = compute_macro_signed_network_stats(d, num_null_perms=20, seed=42)
        assert stats["num_users"] >= 10
        assert stats["structural_balance"]["num_triangles"] >= 0
        out_paper.mkdir(parents=True, exist_ok=True)
        out_fig.mkdir(parents=True, exist_ok=True)
        (out_paper / "results_macro_network_smoke.json").write_text(
            json.dumps(stats, indent=2), encoding="utf-8"
        )
        tex_macro = format_macro_network_latex([stats])
        assert "\\begin{table" in tex_macro
        pdf_p, svg_p = plot_macro_signed_network(stats, output_dir=out_fig)
        assert pdf_p.exists() and svg_p.exists()

        bench = run_signed_link_benchmark(
            d,
            models=LINK_PRED_MODEL_NAMES,
            seeds=[42, 100],
            hidden_dim=16,
            epochs=3,
            patience=3,
            batch_size=32,
        )
        export_latex_table(
            bench,
            out_paper / "tables_signed_link_smoke.tex",
            caption="Smoke Signed Link Benchmark.",
            label="tab:smoke_link",
            metrics=LINK_PRED_METRICS,
        )
        serialize_benchmark_json(bench, out_paper / "results_signed_link_smoke.json")
        print("SMOKE TEST PASSED:", json.dumps(bench["split_counts"]))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tier 2 Macro Signed User Network & Link Prediction")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/vn"), help="Processed dataset directory")
    parser.add_argument("--dataset-name", type=str, default=None, help="Override canonical dataset tag (e.g. vn or pheme)")
    parser.add_argument("--analysis-only", action="store_true", help="Run sparse CPU network polarization analytics + fig5 only")
    parser.add_argument("--epochs", type=int, default=40, help="Max epochs per seed for signed link prediction")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience on Val ROC-AUC")
    parser.add_argument("--batch-size", type=int, default=1024, help="Training mini-batch size")
    parser.add_argument("--hidden-dim", type=int, default=64, help="GNN latent width")
    parser.add_argument("--num-layers", type=int, default=2, help="GNN layer depth")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--num-null-perms", type=int, default=100, help="Degree-preserving sign permutations for SBT z-score")
    parser.add_argument("--paper-dir", type=Path, default=Path("paper"), help="Output directory for JSON and LaTeX tables")
    parser.add_argument("--figures-dir", type=Path, default=None, help="Override figure output dir (default paper/figures/<ds>)")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast offline synthetic smoke test in tempdir")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.smoke_test:
        run_smoke_test()
        return

    data_dir = Path(args.data_dir)
    paper_dir = Path(args.paper_dir)
    paper_dir.mkdir(parents=True, exist_ok=True)

    stats = compute_macro_signed_network_stats(
        data_dir,
        dataset_name=args.dataset_name,
        num_null_perms=args.num_null_perms,
        seed=42,
    )
    ds_name = str(stats["dataset"])
    macro_json_path = paper_dir / f"results_macro_network_{ds_name}.json"
    macro_json_path.write_text(json.dumps(stats, indent=2, allow_nan=False), encoding="utf-8")

    merged_stats = _collect_macro_stats_for_table(paper_dir, stats)
    macro_tex_path = paper_dir / "tables_macro_network.tex"
    macro_tex_path.write_text(format_macro_network_latex(merged_stats), encoding="utf-8")

    fig_dir = Path(args.figures_dir) if args.figures_dir is not None else (paper_dir / "figures" / ds_name)
    pdf_p, svg_p = plot_macro_signed_network(stats, output_dir=fig_dir)
    print(
        f"[{ds_name}] Macro Signed Network: users={stats['num_users']:,} "
        f"interactions={stats['num_interactions']:,} balanced={stats['structural_balance']['balanced_ratio']:.4f} "
        f"(z={stats['structural_balance']['balance_z_score']:.2f}), "
        f"Q_signed={stats['polarization']['signed_modularity']:.4f}, "
        f"RWC={stats['polarization']['rwc']:.4f}",
        flush=True,
    )
    print(f"Exported: {macro_json_path}, {macro_tex_path}, {pdf_p}, {svg_p}", flush=True)

    if args.analysis_only:
        return

    bench = run_signed_link_benchmark(
        data_dir,
        dataset_name=args.dataset_name,
        models=LINK_PRED_MODEL_NAMES,
        seeds=OFFICIAL_SEEDS,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        batch_size=args.batch_size,
    )
    link_json = paper_dir / f"results_signed_link_{ds_name}.json"
    link_tex = paper_dir / f"tables_signed_link_{ds_name}.tex"
    ds_upper = ds_name.upper().replace("_", "\\_")
    export_latex_table(
        bench,
        link_tex,
        caption=(
            f"Macro Temporal Signed Link Prediction on {ds_upper} (opposing vs.\\ supporting reply edges, "
            f"mean $\\pm$ std over 5 seeds; unseen users have zeroed features and train batch pairs are masked)."
        ),
        label=f"tab:signed_link_{ds_name}",
        metrics=LINK_PRED_METRICS,
    )
    serialize_benchmark_json(
        bench,
        link_json,
        extra={
            "val_summary_stats": bench["val_summary_stats"],
            "split_counts": bench["split_counts"],
            "nonparametric_test_label": bench["nonparametric_test_label"],
        },
    )
    print(f"Exported signed link benchmark: {link_json}, {link_tex}", flush=True)


if __name__ == "__main__":
    main()
