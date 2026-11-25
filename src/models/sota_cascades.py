"""SOTA cascade forecasting baselines & tabular comparators for Tier 1 SBi-FAGAT evaluation.

Implements missing non-GNN tabular baselines (`CalibratedEarlyKPersistence`,
`RidgeBaseline`, `LightGBMBaseline` with sklearn `HistGradientBoosting` fallback)
and representative PyTorch cascade/opinion SOTA models (`FriedkinJohnsenGNN`,
`CasCN`, `CasFlow`) adhering strictly to the zero-leakage `CascadeData -> dict`
contract.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import scatter, to_undirected

from src.data.pyg_dataset import (
    TIER2_PRIOR_DIM,
    _compute_full_tree_polarization,
    _compute_root_aligned_stances,
)
from src.models.baselines import (
    MultiTaskHead,
    TwoStreamNodeStem,
    prepare_edge_attr,
    resolve_root_indices,
)

try:  # pragma: no cover - optional runtime package
    import lightgbm as _lgb  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - fallback tested locally
    _lgb = None


def _resolve_single_root_idx(d: Any, num_nodes: int | None = None) -> int:
    rt = getattr(d, "root_index", None)
    if isinstance(rt, Tensor) and rt.dtype != torch.bool and rt.numel() > 0:
        r_idx = int(rt.view(-1)[0].item())
    elif isinstance(rt, (int, np.integer)) and not isinstance(rt, bool):
        r_idx = int(rt)
    else:
        r_idx = 0
    if num_nodes is not None and num_nodes > 0 and not (0 <= r_idx < num_nodes):
        return 0
    return r_idx if r_idx >= 0 else 0


def _get_early_polar_scalar(d: Any) -> float:
    ep = getattr(d, "early_y_polar", None)
    if ep is not None and isinstance(ep, Tensor) and ep.numel() >= 1:
        return float(ep.view(-1)[0].item())
    early_rs = getattr(d, "early_node_root_stance", None)
    if early_rs is not None and isinstance(early_rs, Tensor) and early_rs.numel() > 1:
        flat_rs = early_rs.view(-1)
        r_idx = _resolve_single_root_idx(d, int(flat_rs.numel()))
        vals = [
            float(v)
            for idx_i, v in enumerate(flat_rs.detach().cpu().tolist())
            if idx_i != r_idx
        ]
        return float(_compute_full_tree_polarization(vals))
    ei = getattr(d, "edge_index_td", getattr(d, "edge_index", None))
    ea = getattr(d, "edge_attr", getattr(d, "edge_stance", None))
    if ei is not None and ei.numel() > 0:
        x_t = getattr(d, "x", None)
        num_nodes = int(x_t.size(0)) if isinstance(x_t, Tensor) and x_t.ndim >= 1 else None
        root_id = str(_resolve_single_root_idx(d, num_nodes))
        children: dict[str, list[str]] = {}
        smap: dict[tuple[str, str], float] = {}
        for idx in range(int(ei.size(1))):
            u_s = str(int(ei[0, idx].item()))
            v_s = str(int(ei[1, idx].item()))
            children.setdefault(u_s, []).append(v_s)
            if isinstance(ea, Tensor) and ea.numel() > 0:
                s_val = float(ea[idx, 0].item()) if ea.dim() == 2 else float(ea[idx].item())
            else:
                s_val = 0.0
            smap[(u_s, v_s)] = float(min(1.0, max(-1.0, s_val)))
        aligned = _compute_root_aligned_stances(
            root_id=root_id,
            children_adj=children,
            edge_lookup={},
            stance_map=smap,
        )
        return float(_compute_full_tree_polarization(aligned))
    return 0.0


def extract_cascade_summary_features(
    cascades: Sequence[Any],
    *,
    svd_dim: int = 32,
    fit_transform: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract leakage-free tabular features (`42` structural/NLP stats + SVD SpanPool) from early cascades.

    When ``fit_transform`` is ``None``, fits ``StandardScaler`` and ``TruncatedSVD``
    strictly on ``cascades`` (the training split). Passing the returned state dict to
    validation/test calls guarantees zero split leakage.
    """
    if not cascades:
        out_dim = int(fit_transform["total_dim"]) if fit_transform is not None else 43
        return np.zeros((0, out_dim), dtype=np.float64), (fit_transform or {})

    if fit_transform is not None:
        sem_dim = int(fit_transform["sem_dim"])
    else:
        x0 = cascades[0].x
        sem_dim = 768 if int(x0.size(1)) >= 768 else int(x0.size(1))

    base_rows: list[np.ndarray] = []
    sem_rows: list[np.ndarray] = []
    for d in cascades:
        x_t = d.x.detach().cpu().float()
        num_nodes = int(x_t.size(0))
        r_idx = _resolve_single_root_idx(d, num_nodes)

        # 1. cascade_stats [8]
        cs = getattr(d, "cascade_stats", None)
        cs_vec = np.zeros(8, dtype=np.float64)
        if isinstance(cs, Tensor) and cs.numel() > 0:
            arr = cs.detach().cpu().numpy().reshape(-1).astype(np.float64)
            take = min(8, arr.size)
            cs_vec[:take] = arr[:take]

        # 2. early_y_polar [1] + size/density [2]
        early_yp = _get_early_polar_scalar(d)
        early_rep_log = math.log1p(max(0, num_nodes - 1))
        ei = getattr(d, "edge_index_td", getattr(d, "edge_index", None))
        num_edges = int(ei.size(1)) if isinstance(ei, Tensor) and ei.dim() == 2 else 0
        edge_per_reply = float(num_edges) / float(max(1, num_nodes - 1))

        # 3. pooled edge_attr [11]
        edge_vec = np.zeros(11, dtype=np.float64)
        ea = getattr(d, "edge_attr", None)
        if isinstance(ea, Tensor) and ea.numel() > 0 and ea.dim() == 2:
            ea_np = ea.detach().cpu().numpy().astype(np.float64)
            s_arr = ea_np[:, 0]
            h_arr = ea_np[:, 1] if ea_np.shape[1] >= 2 else np.zeros_like(s_arr)
            edge_vec[0:4] = [float(np.mean(s_arr)), float(np.std(s_arr)), float(np.min(s_arr)), float(np.max(s_arr))]
            edge_vec[4:8] = [float(np.mean(h_arr)), float(np.std(h_arr)), float(np.min(h_arr)), float(np.max(h_arr))]
            edge_vec[8] = float(np.mean(s_arr > 0.1))
            edge_vec[9] = float(np.mean(s_arr < -0.1))
            edge_vec[10] = float(np.mean(h_arr > 0.5))

        # 4. pooled struct_x [20]
        struct_vec = np.zeros(20, dtype=np.float64)
        sx = getattr(d, "struct_x", None)
        if isinstance(sx, Tensor) and sx.numel() > 0 and sx.dim() == 2:
            sx_np = sx.detach().cpu().numpy().astype(np.float64)
            w_s = min(10, sx_np.shape[1])
            struct_vec[:w_s] = np.mean(sx_np[:, :w_s], axis=0)
            struct_vec[10 : 10 + w_s] = np.max(sx_np[:, :w_s], axis=0)

        base_rows.append(
            np.concatenate(
                [
                    cs_vec,
                    np.array([early_yp, early_rep_log, edge_per_reply], dtype=np.float64),
                    edge_vec,
                    struct_vec,
                ]
            )
        )

        # 5. SpanPool [root_emb, mean_reply_emb]
        x_sem = x_t[:, :sem_dim].numpy().astype(np.float64)
        if x_sem.shape[1] < sem_dim:
            x_sem = np.pad(x_sem, ((0, 0), (0, sem_dim - x_sem.shape[1])))
        root_emb = x_sem[r_idx]
        if num_nodes > 1:
            reply_mask = np.ones(num_nodes, dtype=bool)
            reply_mask[r_idx] = False
            reply_emb = np.mean(x_sem[reply_mask], axis=0)
        else:
            reply_emb = root_emb.copy()
        sem_rows.append(np.concatenate([root_emb, reply_emb]))

    base_mat = np.vstack(base_rows)
    sem_mat = np.vstack(sem_rows)
    base_mat = np.nan_to_num(base_mat, nan=0.0, posinf=0.0, neginf=0.0)
    sem_mat = np.nan_to_num(sem_mat, nan=0.0, posinf=0.0, neginf=0.0)

    if fit_transform is None:
        n_train = sem_mat.shape[0]
        k_svd = min(int(svd_dim), max(0, n_train - 1), sem_mat.shape[1])
        scaler = StandardScaler()
        sem_scaled = scaler.fit_transform(sem_mat) if n_train >= 2 else sem_mat
        if k_svd >= 1 and float(np.std(sem_scaled)) > 1e-12:
            svd: TruncatedSVD | None = TruncatedSVD(n_components=k_svd, random_state=42)
            sem_red = svd.fit_transform(sem_scaled)
        else:
            svd = None
            sem_red = np.zeros((n_train, 1), dtype=np.float64)
        full = np.hstack([base_mat, sem_red])
        state: dict[str, Any] = {
            "sem_dim": sem_dim,
            "scaler": scaler if n_train >= 2 else None,
            "svd": svd,
            "svd_out_dim": sem_red.shape[1],
            "total_dim": full.shape[1],
        }
        return np.nan_to_num(full, nan=0.0, posinf=0.0, neginf=0.0), state

    scaler_fit = fit_transform.get("scaler")
    svd_fit = fit_transform.get("svd")
    sem_scaled = scaler_fit.transform(sem_mat) if scaler_fit is not None else sem_mat
    if svd_fit is not None:
        sem_red = svd_fit.transform(sem_scaled)
    else:
        sem_red = np.zeros((sem_mat.shape[0], int(fit_transform.get("svd_out_dim", 1))), dtype=np.float64)
    full = np.hstack([base_mat, sem_red])
    return np.nan_to_num(full, nan=0.0, posinf=0.0, neginf=0.0), fit_transform


def _extract_cascade_targets(
    cascades: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_size = np.array([float(d.y_size.view(-1)[0].item()) for d in cascades], dtype=np.float64)
    y_polar = np.array([float(d.y_polar.view(-1)[0].item()) for d in cascades], dtype=np.float64)
    y_rumor = np.array(
        [
            int(getattr(d, "y_rumor", getattr(d, "y", -1)).view(-1)[0].item())
            if isinstance(getattr(d, "y_rumor", getattr(d, "y", None)), Tensor)
            else -1
            for d in cascades
        ],
        dtype=np.int64,
    )
    return y_size, y_polar, y_rumor


def _fit_empirical_prior_logits(y_rumor: np.ndarray, num_classes: int) -> np.ndarray:
    valid = y_rumor[y_rumor >= 0]
    if valid.size == 0:
        return np.zeros(num_classes, dtype=np.float64)
    counts = np.bincount(valid[valid < num_classes], minlength=num_classes).astype(np.float64)
    probs = (counts + 1e-3) / float(counts.sum() + 1e-3 * num_classes)
    return np.log(np.clip(probs, 1e-6, 1.0))


def _map_proba_to_full_logits(
    probs: np.ndarray, classes: np.ndarray, num_classes: int
) -> np.ndarray:
    full = np.full((probs.shape[0], num_classes), 1e-6, dtype=np.float64)
    for col_idx, cls_id in enumerate(classes.tolist()):
        c_int = int(cls_id)
        if 0 <= c_int < num_classes:
            full[:, c_int] = probs[:, col_idx]
    full = full / np.sum(full, axis=1, keepdims=True).clip(min=1e-12)
    return np.log(np.clip(full, 1e-6, 1.0))


class TabularCascadeBaseline:
    """Base protocol for tabular/closed-form early-cascade baselines inside `run_multi_seed_benchmark`."""

    def __init__(self, num_rumor_classes: int = 4) -> None:
        self.num_rumor_classes = int(num_rumor_classes)
        self.prior_logits: np.ndarray = np.zeros(self.num_rumor_classes, dtype=np.float64)

    def fit_dataset(
        self,
        train_cascades: Sequence[Any],
        val_cascades: Sequence[Any],
        *,
        seed: int = 42,
    ) -> "TabularCascadeBaseline":
        raise NotImplementedError

    def predict_dataset(self, cascades: Sequence[Any]) -> dict[str, np.ndarray]:
        raise NotImplementedError


class CalibratedEarlyKPersistence(TabularCascadeBaseline):
    """Calibrated Early-K Persistence: affine-calibrated `early_y_polar` + early-size Ridge."""

    def __init__(self, num_rumor_classes: int = 4) -> None:
        super().__init__(num_rumor_classes=num_rumor_classes)
        self._size_reg: Ridge | None = None
        self._polar_reg: Ridge | None = None
        self._fallback_size: float = 0.0
        self._fallback_polar: float = 0.0

    @staticmethod
    def _extract_size_features(cascades: Sequence[Any]) -> np.ndarray:
        rows: list[list[float]] = []
        for d in cascades:
            num_nodes = int(d.x.size(0))
            cs = getattr(d, "cascade_stats", None)
            cs_vals = (
                cs.view(-1)[:4].detach().cpu().tolist()
                if isinstance(cs, Tensor) and cs.numel() >= 4
                else [0.0, 0.0, 0.5, 0.0]
            )
            rows.append([math.log1p(max(0, num_nodes - 1))] + [float(v) for v in cs_vals])
        return np.asarray(rows, dtype=np.float64)

    def fit_dataset(
        self,
        train_cascades: Sequence[Any],
        val_cascades: Sequence[Any],
        *,
        seed: int = 42,
    ) -> "CalibratedEarlyKPersistence":
        del val_cascades, seed
        y_size, y_polar, y_rumor = _extract_cascade_targets(train_cascades)
        self.prior_logits = _fit_empirical_prior_logits(y_rumor, self.num_rumor_classes)
        self._fallback_size = float(np.mean(y_size)) if y_size.size > 0 else 0.0
        self._fallback_polar = float(np.mean(y_polar)) if y_polar.size > 0 else 0.0

        if len(train_cascades) >= 2:
            x_size = self._extract_size_features(train_cascades)
            self._size_reg = Ridge(alpha=1.0).fit(x_size, y_size)
            ep_tr = np.array([[_get_early_polar_scalar(d)] for d in train_cascades], dtype=np.float64)
            self._polar_reg = Ridge(alpha=1e-4).fit(ep_tr, y_polar)
        else:
            self._size_reg = None
            self._polar_reg = None
        return self

    def predict_dataset(self, cascades: Sequence[Any]) -> dict[str, np.ndarray]:
        n = len(cascades)
        if n == 0:
            return {
                "y_size_pred": np.zeros(0, dtype=np.float64),
                "y_polar_pred": np.zeros(0, dtype=np.float64),
                "rumor_logits": np.zeros((0, self.num_rumor_classes), dtype=np.float64),
            }
        if self._size_reg is not None:
            y_size_pred = np.clip(self._size_reg.predict(self._extract_size_features(cascades)), 0.0, None)
        else:
            y_size_pred = np.full(n, max(0.0, self._fallback_size), dtype=np.float64)
        if self._polar_reg is not None:
            ep = np.array([[_get_early_polar_scalar(d)] for d in cascades], dtype=np.float64)
            y_polar_pred = np.clip(self._polar_reg.predict(ep), 0.0, 1.0)
        else:
            y_polar_pred = np.full(n, min(1.0, max(0.0, self._fallback_polar)), dtype=np.float64)
        rumor_logits = np.tile(self.prior_logits[None, :], (n, 1))
        return {
            "y_size_pred": y_size_pred.astype(np.float64),
            "y_polar_pred": y_polar_pred.astype(np.float64),
            "rumor_logits": rumor_logits.astype(np.float64),
        }


class RidgeBaseline(TabularCascadeBaseline):
    """L2-regularized Ridge + LogisticRegression baseline on summary + SVD SpanPool features."""

    def __init__(self, num_rumor_classes: int = 4, svd_dim: int = 32) -> None:
        super().__init__(num_rumor_classes=num_rumor_classes)
        self.svd_dim = int(svd_dim)
        self._feat_state: dict[str, Any] | None = None
        self._scaler: StandardScaler | None = None
        self._size_reg: Ridge | None = None
        self._polar_reg: Ridge | None = None
        self._rumor_clf: LogisticRegression | None = None
        self._fallback_size: float = 0.0
        self._fallback_polar: float = 0.0

    def fit_dataset(
        self,
        train_cascades: Sequence[Any],
        val_cascades: Sequence[Any],
        *,
        seed: int = 42,
    ) -> "RidgeBaseline":
        x_tr_raw, self._feat_state = extract_cascade_summary_features(
            train_cascades, svd_dim=self.svd_dim
        )
        y_size_tr, y_polar_tr, y_rumor_tr = _extract_cascade_targets(train_cascades)
        self.prior_logits = _fit_empirical_prior_logits(y_rumor_tr, self.num_rumor_classes)
        self._fallback_size = float(np.mean(y_size_tr)) if y_size_tr.size > 0 else 0.0
        self._fallback_polar = float(np.mean(y_polar_tr)) if y_polar_tr.size > 0 else 0.0

        self._scaler = StandardScaler()
        x_tr = self._scaler.fit_transform(x_tr_raw) if x_tr_raw.shape[0] >= 2 else x_tr_raw

        has_val = len(val_cascades) > 0
        if has_val:
            x_va_raw, _ = extract_cascade_summary_features(
                val_cascades, svd_dim=self.svd_dim, fit_transform=self._feat_state
            )
            x_va = self._scaler.transform(x_va_raw) if x_tr_raw.shape[0] >= 2 else x_va_raw
            y_size_va, y_polar_va, y_rumor_va = _extract_cascade_targets(val_cascades)
        else:
            x_va = np.zeros((0, x_tr.shape[1]), dtype=np.float64)
            y_size_va = np.zeros(0, dtype=np.float64)
            y_polar_va = np.zeros(0, dtype=np.float64)
            y_rumor_va = np.zeros(0, dtype=np.int64)

        alphas = (0.1, 1.0, 10.0, 100.0, 500.0)
        best_s_reg, best_s_mse = None, float("inf")
        best_p_reg, best_p_mse = None, float("inf")
        for alpha in alphas:
            m_s = Ridge(alpha=alpha, random_state=seed).fit(x_tr, y_size_tr)
            m_p = Ridge(alpha=alpha, random_state=seed).fit(x_tr, y_polar_tr)
            if has_val:
                mse_s = float(np.mean((np.clip(m_s.predict(x_va), 0.0, None) - y_size_va) ** 2))
                mse_p = float(np.mean((np.clip(m_p.predict(x_va), 0.0, 1.0) - y_polar_va) ** 2))
            else:
                mse_s = mse_p = abs(alpha - 10.0)
            if mse_s < best_s_mse:
                best_s_mse = mse_s
                best_s_reg = m_s
            if mse_p < best_p_mse:
                best_p_mse = mse_p
                best_p_reg = m_p
        self._size_reg = best_s_reg
        self._polar_reg = best_p_reg

        valid_tr = y_rumor_tr >= 0
        self._rumor_clf = None
        if np.any(valid_tr) and np.unique(y_rumor_tr[valid_tr]).size >= 2:
            x_cls_tr = x_tr[valid_tr]
            y_cls_tr = y_rumor_tr[valid_tr]
            valid_va = y_rumor_va >= 0
            best_clf, best_loss = None, float("inf")
            for c_val in (0.05, 0.25, 1.0, 4.0):
                try:
                    clf = LogisticRegression(
                        C=c_val,
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=seed,
                    ).fit(x_cls_tr, y_cls_tr)
                except Exception:
                    continue
                if has_val and np.any(valid_va):
                    probs = clf.predict_proba(x_va[valid_va])
                    logits = _map_proba_to_full_logits(probs, clf.classes_, self.num_rumor_classes)
                    ce = -float(np.mean(logits[np.arange(logits.shape[0]), y_rumor_va[valid_va]]))
                else:
                    ce = abs(c_val - 1.0)
                if ce < best_loss:
                    best_loss = ce
                    best_clf = clf
            self._rumor_clf = best_clf
        return self

    def predict_dataset(self, cascades: Sequence[Any]) -> dict[str, np.ndarray]:
        n = len(cascades)
        if n == 0:
            return {
                "y_size_pred": np.zeros(0, dtype=np.float64),
                "y_polar_pred": np.zeros(0, dtype=np.float64),
                "rumor_logits": np.zeros((0, self.num_rumor_classes), dtype=np.float64),
            }
        x_raw, _ = extract_cascade_summary_features(
            cascades, svd_dim=self.svd_dim, fit_transform=self._feat_state
        )
        x_in = self._scaler.transform(x_raw) if (self._scaler is not None and hasattr(self._scaler, "mean_")) else x_raw
        y_size_pred = (
            np.clip(self._size_reg.predict(x_in), 0.0, None)
            if self._size_reg is not None
            else np.full(n, max(0.0, self._fallback_size), dtype=np.float64)
        )
        y_polar_pred = (
            np.clip(self._polar_reg.predict(x_in), 0.0, 1.0)
            if self._polar_reg is not None
            else np.full(n, min(1.0, max(0.0, self._fallback_polar)), dtype=np.float64)
        )
        if self._rumor_clf is not None:
            probs = self._rumor_clf.predict_proba(x_in)
            rumor_logits = _map_proba_to_full_logits(
                probs, self._rumor_clf.classes_, self.num_rumor_classes
            )
        else:
            rumor_logits = np.tile(self.prior_logits[None, :], (n, 1))
        return {
            "y_size_pred": y_size_pred.astype(np.float64),
            "y_polar_pred": y_polar_pred.astype(np.float64),
            "rumor_logits": rumor_logits.astype(np.float64),
        }


class LightGBMBaseline(TabularCascadeBaseline):
    """Gradient-boosted tree baseline (`lightgbm` if installed, else sklearn `HistGradientBoosting`)."""

    def __init__(self, num_rumor_classes: int = 4, svd_dim: int = 32) -> None:
        super().__init__(num_rumor_classes=num_rumor_classes)
        self.svd_dim = int(svd_dim)
        self.backend_name: str = "lightgbm" if _lgb is not None else "sklearn_hist_gbrt"
        self._feat_state: dict[str, Any] | None = None
        self._size_reg: Any = None
        self._polar_reg: Any = None
        self._rumor_clf: Any = None
        self._fallback_size: float = 0.0
        self._fallback_polar: float = 0.0

    def _make_regressor(self, max_depth: int, lr: float, min_leaf: int, seed: int) -> Any:
        if _lgb is not None:
            return _lgb.LGBMRegressor(
                n_estimators=150,
                max_depth=max_depth,
                learning_rate=lr,
                min_child_samples=min_leaf,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                verbose=-1,
            )
        return HistGradientBoostingRegressor(
            max_iter=150,
            max_depth=max_depth,
            learning_rate=lr,
            min_samples_leaf=min_leaf,
            random_state=seed,
        )

    def _make_classifier(self, max_depth: int, lr: float, min_leaf: int, seed: int) -> Any:
        if _lgb is not None:
            return _lgb.LGBMClassifier(
                n_estimators=150,
                max_depth=max_depth,
                learning_rate=lr,
                min_child_samples=min_leaf,
                class_weight="balanced",
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                verbose=-1,
            )
        return HistGradientBoostingClassifier(
            max_iter=150,
            max_depth=max_depth,
            learning_rate=lr,
            min_samples_leaf=min_leaf,
            class_weight="balanced",
            random_state=seed,
        )

    def fit_dataset(
        self,
        train_cascades: Sequence[Any],
        val_cascades: Sequence[Any],
        *,
        seed: int = 42,
    ) -> "LightGBMBaseline":
        x_tr, self._feat_state = extract_cascade_summary_features(
            train_cascades, svd_dim=self.svd_dim
        )
        y_size_tr, y_polar_tr, y_rumor_tr = _extract_cascade_targets(train_cascades)
        self.prior_logits = _fit_empirical_prior_logits(y_rumor_tr, self.num_rumor_classes)
        self._fallback_size = float(np.mean(y_size_tr)) if y_size_tr.size > 0 else 0.0
        self._fallback_polar = float(np.mean(y_polar_tr)) if y_polar_tr.size > 0 else 0.0

        n_tr = x_tr.shape[0]
        if n_tr < 4:
            self._size_reg = Ridge(alpha=1.0, random_state=seed).fit(x_tr, y_size_tr)
            self._polar_reg = Ridge(alpha=1.0, random_state=seed).fit(x_tr, y_polar_tr)
            self._rumor_clf = None
            return self

        has_val = len(val_cascades) > 0
        if has_val:
            x_va, _ = extract_cascade_summary_features(
                val_cascades, svd_dim=self.svd_dim, fit_transform=self._feat_state
            )
            y_size_va, y_polar_va, y_rumor_va = _extract_cascade_targets(val_cascades)
        else:
            x_va = np.zeros((0, x_tr.shape[1]), dtype=np.float64)
            y_size_va = np.zeros(0, dtype=np.float64)
            y_polar_va = np.zeros(0, dtype=np.float64)
            y_rumor_va = np.zeros(0, dtype=np.int64)

        min_leaf = min(20, max(1, n_tr // 4))
        grid = ((3, 0.05), (5, 0.05), (3, 0.1))
        best_s_reg, best_s_mse = None, float("inf")
        best_p_reg, best_p_mse = None, float("inf")
        for md, lr in grid:
            m_s = self._make_regressor(md, lr, min_leaf, seed).fit(x_tr, y_size_tr)
            m_p = self._make_regressor(md, lr, min_leaf, seed).fit(x_tr, y_polar_tr)
            if has_val:
                mse_s = float(np.mean((np.clip(m_s.predict(x_va), 0.0, None) - y_size_va) ** 2))
                mse_p = float(np.mean((np.clip(m_p.predict(x_va), 0.0, 1.0) - y_polar_va) ** 2))
            else:
                mse_s = mse_p = float(md)
            if mse_s < best_s_mse:
                best_s_mse = mse_s
                best_s_reg = m_s
            if mse_p < best_p_mse:
                best_p_mse = mse_p
                best_p_reg = m_p
        self._size_reg = best_s_reg
        self._polar_reg = best_p_reg

        valid_tr = y_rumor_tr >= 0
        self._rumor_clf = None
        if np.sum(valid_tr) >= 4 and np.unique(y_rumor_tr[valid_tr]).size >= 2:
            x_cls_tr = x_tr[valid_tr]
            y_cls_tr = y_rumor_tr[valid_tr]
            cls_leaf = min(20, max(1, x_cls_tr.shape[0] // 4))
            valid_va = y_rumor_va >= 0
            best_clf, best_ce = None, float("inf")
            for md, lr in grid:
                try:
                    clf = self._make_classifier(md, lr, cls_leaf, seed).fit(x_cls_tr, y_cls_tr)
                except Exception:
                    continue
                if has_val and np.any(valid_va):
                    probs = clf.predict_proba(x_va[valid_va])
                    logits = _map_proba_to_full_logits(probs, clf.classes_, self.num_rumor_classes)
                    ce = -float(np.mean(logits[np.arange(logits.shape[0]), y_rumor_va[valid_va]]))
                else:
                    ce = float(md)
                if ce < best_ce:
                    best_ce = ce
                    best_clf = clf
            self._rumor_clf = best_clf
        return self

    def predict_dataset(self, cascades: Sequence[Any]) -> dict[str, np.ndarray]:
        n = len(cascades)
        if n == 0:
            return {
                "y_size_pred": np.zeros(0, dtype=np.float64),
                "y_polar_pred": np.zeros(0, dtype=np.float64),
                "rumor_logits": np.zeros((0, self.num_rumor_classes), dtype=np.float64),
            }
        x_in, _ = extract_cascade_summary_features(
            cascades, svd_dim=self.svd_dim, fit_transform=self._feat_state
        )
        y_size_pred = (
            np.clip(self._size_reg.predict(x_in), 0.0, None)
            if self._size_reg is not None
            else np.full(n, max(0.0, self._fallback_size), dtype=np.float64)
        )
        y_polar_pred = (
            np.clip(self._polar_reg.predict(x_in), 0.0, 1.0)
            if self._polar_reg is not None
            else np.full(n, min(1.0, max(0.0, self._fallback_polar)), dtype=np.float64)
        )
        if self._rumor_clf is not None:
            probs = self._rumor_clf.predict_proba(x_in)
            rumor_logits = _map_proba_to_full_logits(
                probs, self._rumor_clf.classes_, self.num_rumor_classes
            )
        else:
            rumor_logits = np.tile(self.prior_logits[None, :], (n, 1))
        return {
            "y_size_pred": y_size_pred.astype(np.float64),
            "y_polar_pred": y_polar_pred.astype(np.float64),
            "rumor_logits": rumor_logits.astype(np.float64),
        }


class FriedkinJohnsenGNN(nn.Module):
    """Differentiable Friedkin-Johnsen opinion dynamics GNN on signed early cascades."""

    def __init__(
        self,
        in_dim: int = 770,
        hidden_dim: int = 64,
        edge_dim: int = 2,
        num_rumor_classes: int = 4,
        num_steps: int = 5,
        dropout: float = 0.1,
        use_cascade_stats: bool = False,
        cascade_feat_dim: int = 8,
        predict_uncertainty: bool = False,
        use_polar_persistence_anchor: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.edge_dim = int(edge_dim)
        self.num_steps = max(1, int(num_steps))
        self.use_cascade_stats = bool(use_cascade_stats)
        self.cascade_feat_dim = int(cascade_feat_dim)
        self.use_polar_persistence_anchor = bool(use_polar_persistence_anchor)

        self.stem = TwoStreamNodeStem(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.opinion_init = nn.Linear(hidden_dim, hidden_dim)
        self.stubborn_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, max(16, hidden_dim // 2)),
            nn.GELU(),
            nn.Linear(max(16, hidden_dim // 2), 1),
        )
        self.edge_weight_mlp = nn.Sequential(
            nn.Linear(max(1, self.edge_dim), 16),
            nn.GELU(),
            nn.Linear(16, 1),
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
        num_nodes = int(x.size(0))
        root_idx, batch_vec = resolve_root_indices(data, num_nodes)
        num_graphs = int(root_idx.size(0))

        h0 = self.stem(x, data, root_idx)
        z0 = torch.tanh(self.opinion_init(h0))
        lam = torch.sigmoid(self.stubborn_mlp(torch.cat([h0, z0], dim=-1)))

        edge_index_td = getattr(data, "edge_index_td", data.edge_index)
        edge_attr = prepare_edge_attr(x, edge_index_td, data, edge_dim=max(1, self.edge_dim))

        z = z0
        if edge_index_td.numel() > 0:
            s_uv = edge_attr[:, 0]
            mag = F.softplus(self.edge_weight_mlp(edge_attr)).squeeze(-1)
            a_uv = s_uv * mag
            src = torch.cat([edge_index_td[0], edge_index_td[1]], dim=0)
            dst = torch.cat([edge_index_td[1], edge_index_td[0]], dim=0)
            w = torch.cat([a_uv, a_uv], dim=0)
            denom = scatter(w.abs(), dst, dim=0, dim_size=num_nodes, reduce="sum") + 1.0
            norm_w = (w / denom[dst]).unsqueeze(-1)
            for _ in range(self.num_steps):
                agg = scatter(norm_w * z[src], dst, dim=0, dim_size=num_nodes, reduce="sum")
                z = lam * z0 + (1.0 - lam) * agg
        else:
            dummy_w = sum(p.sum() for p in self.edge_weight_mlp.parameters())
            z = lam * z0 + 0.0 * dummy_w

        node_emb = z
        graph_emb = torch.cat(
            [node_emb[root_idx], global_mean_pool(node_emb, batch_vec, size=num_graphs)],
            dim=-1,
        )
        c_stats = getattr(data, "cascade_stats", None) if self.use_cascade_stats else None
        early_yp = getattr(data, "early_y_polar", None) if self.use_polar_persistence_anchor else None
        return self.head(graph_emb, node_emb, cascade_stats=c_stats, early_y_polar=early_yp)


def _compute_snapshot_ranks(
    data: Any,
    num_nodes: int,
    root_idx: Tensor,
    batch_vec: Tensor,
    num_graphs: int,
) -> tuple[Tensor, Tensor]:
    is_root = torch.zeros(num_nodes, dtype=torch.bool, device=batch_vec.device)
    valid_roots = root_idx[(root_idx >= 0) & (root_idx < num_nodes)]
    if valid_roots.numel() > 0:
        is_root[valid_roots] = True

    sx = getattr(data, "struct_x", None)
    if isinstance(sx, Tensor) and sx.dim() == 2 and sx.size(0) == num_nodes and sx.size(1) >= 4:
        ranks = sx[:, 3].to(device=batch_vec.device, dtype=torch.float32)
        if bool((ranks > 0).any()):
            max_per_g = scatter(ranks, batch_vec, dim=0, dim_size=num_graphs, reduce="max").clamp(min=1e-6)
            return (ranks / max_per_g[batch_vec]).clamp(0.0, 1.0), is_root

    ranks = torch.zeros(num_nodes, dtype=torch.float32, device=batch_vec.device)
    for g in range(num_graphs):
        idxs = (batch_vec == g).nonzero(as_tuple=False).view(-1)
        if idxs.numel() > 1:
            ranks[idxs] = torch.linspace(0.0, 1.0, steps=int(idxs.numel()), device=batch_vec.device)
    return ranks, is_root


class CasCN(nn.Module):
    """Cascade Convolutional Network (Chen et al., ICDE 2019): chronological GCN snapshots + GRU."""

    def __init__(
        self,
        in_dim: int = 770,
        hidden_dim: int = 64,
        num_snapshots: int = 4,
        num_rumor_classes: int = 4,
        dropout: float = 0.1,
        use_cascade_stats: bool = False,
        cascade_feat_dim: int = 8,
        predict_uncertainty: bool = False,
        use_polar_persistence_anchor: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_snapshots = max(1, int(num_snapshots))
        self.use_cascade_stats = bool(use_cascade_stats)
        self.cascade_feat_dim = int(cascade_feat_dim)
        self.use_polar_persistence_anchor = bool(use_polar_persistence_anchor)

        self.stem = TwoStreamNodeStem(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
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
        num_nodes = int(x.size(0))
        root_idx, batch_vec = resolve_root_indices(data, num_nodes)
        num_graphs = int(root_idx.size(0))

        h0 = self.stem(x, data, root_idx)
        raw_edge_index = getattr(data, "edge_index_td", data.edge_index)
        rel_ranks, is_root = _compute_snapshot_ranks(data, num_nodes, root_idx, batch_vec, num_graphs)
        thresholds = torch.linspace(
            1.0 / float(self.num_snapshots), 1.0, self.num_snapshots, device=x.device
        )

        snap_pools: list[Tensor] = []
        last_node_emb = h0
        for thr in thresholds:
            active = (rel_ranks <= thr + 1e-6) | is_root
            if raw_edge_index.numel() > 0:
                keep_e = active[raw_edge_index[0]] & active[raw_edge_index[1]]
                sub_ei = raw_edge_index[:, keep_e]
                snap_ei = to_undirected(sub_ei, num_nodes=num_nodes) if sub_ei.numel() > 0 else sub_ei
            else:
                snap_ei = raw_edge_index

            h = self.dropout(self.norm1(F.relu(self.conv1(h0, snap_ei))))
            h = self.dropout(self.norm2(F.relu(self.conv2(h, snap_ei))))
            last_node_emb = h

            mask_f = active.to(dtype=h.dtype).unsqueeze(-1)
            pool_num = scatter(h * mask_f, batch_vec, dim=0, dim_size=num_graphs, reduce="sum")
            pool_den = scatter(mask_f, batch_vec, dim=0, dim_size=num_graphs, reduce="sum").clamp(min=1.0)
            snap_pools.append(pool_num / pool_den)

        seq = torch.stack(snap_pools, dim=1)
        gru_out, _ = self.gru(seq)
        graph_emb = torch.cat([last_node_emb[root_idx], gru_out[:, -1, :]], dim=-1)
        c_stats = getattr(data, "cascade_stats", None) if self.use_cascade_stats else None
        early_yp = getattr(data, "early_y_polar", None) if self.use_polar_persistence_anchor else None
        return self.head(graph_emb, last_node_emb, cascade_stats=c_stats, early_y_polar=early_yp)


class CasFlow(nn.Module):
    """CasFlow (Xu et al., TKDE 2021) adapted with local snapshot flow + causal Tier 2 user prior."""

    def __init__(
        self,
        in_dim: int = 770,
        hidden_dim: int = 64,
        edge_dim: int = 2,
        num_snapshots: int = 4,
        num_rumor_classes: int = 4,
        dropout: float = 0.1,
        use_cascade_stats: bool = False,
        cascade_feat_dim: int = 8,
        predict_uncertainty: bool = False,
        use_polar_persistence_anchor: bool = True,
        tier2_prior_dim: int = TIER2_PRIOR_DIM,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.edge_dim = int(edge_dim)
        self.num_snapshots = max(1, int(num_snapshots))
        self.use_cascade_stats = bool(use_cascade_stats)
        self.cascade_feat_dim = int(cascade_feat_dim)
        self.use_polar_persistence_anchor = bool(use_polar_persistence_anchor)
        self.tier2_prior_dim = int(tier2_prior_dim)

        self.stem = TwoStreamNodeStem(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        self.prior_node_proj = nn.Sequential(
            nn.Linear(self.tier2_prior_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.prior_pool_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.prior_gate = nn.Parameter(torch.tensor(0.25))

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
        num_nodes = int(x.size(0))
        root_idx, batch_vec = resolve_root_indices(data, num_nodes)
        num_graphs = int(root_idx.size(0))

        h0 = self.stem(x, data, root_idx)
        raw_edge_index = getattr(data, "edge_index_td", data.edge_index)
        rel_ranks, is_root = _compute_snapshot_ranks(data, num_nodes, root_idx, batch_vec, num_graphs)
        thresholds = torch.linspace(
            1.0 / float(self.num_snapshots), 1.0, self.num_snapshots, device=x.device
        )

        snap_pools: list[Tensor] = []
        last_node_emb = h0
        for thr in thresholds:
            active = (rel_ranks <= thr + 1e-6) | is_root
            if raw_edge_index.numel() > 0:
                keep_e = active[raw_edge_index[0]] & active[raw_edge_index[1]]
                sub_ei = raw_edge_index[:, keep_e]
                snap_ei = to_undirected(sub_ei, num_nodes=num_nodes) if sub_ei.numel() > 0 else sub_ei
            else:
                snap_ei = raw_edge_index

            h = self.dropout(self.norm1(F.relu(self.conv1(h0, snap_ei))))
            h = self.dropout(self.norm2(F.relu(self.conv2(h, snap_ei))))
            last_node_emb = h

            mask_f = active.to(dtype=h.dtype).unsqueeze(-1)
            pool_num = scatter(h * mask_f, batch_vec, dim=0, dim_size=num_graphs, reduce="sum")
            pool_den = scatter(mask_f, batch_vec, dim=0, dim_size=num_graphs, reduce="sum").clamp(min=1.0)
            snap_pools.append(pool_num / pool_den)

        seq = torch.stack(snap_pools, dim=1)
        gru_out, _ = self.gru(seq)
        z_local = gru_out[:, -1, :]

        raw_prior = getattr(data, "tier2_user_prior", None)
        if (
            isinstance(raw_prior, Tensor)
            and raw_prior.dim() == 2
            and raw_prior.size(0) == num_nodes
        ):
            p_t = raw_prior.to(device=x.device, dtype=h0.dtype)
            if p_t.size(1) < self.tier2_prior_dim:
                p_t = F.pad(p_t, (0, self.tier2_prior_dim - p_t.size(1)))
            elif p_t.size(1) > self.tier2_prior_dim:
                p_t = p_t[:, : self.tier2_prior_dim]
        else:
            p_t = h0.new_zeros((num_nodes, self.tier2_prior_dim))
        h_prior = self.prior_node_proj(p_t)
        z_global = self.prior_pool_proj(
            torch.cat(
                [h_prior[root_idx], global_mean_pool(h_prior, batch_vec, size=num_graphs)],
                dim=-1,
            )
        )

        fused_flow = z_local + torch.tanh(self.prior_gate) * z_global
        graph_emb = torch.cat([last_node_emb[root_idx], fused_flow], dim=-1)
        c_stats = getattr(data, "cascade_stats", None) if self.use_cascade_stats else None
        early_yp = getattr(data, "early_y_polar", None) if self.use_polar_persistence_anchor else None
        return self.head(graph_emb, last_node_emb, cascade_stats=c_stats, early_y_polar=early_yp)
