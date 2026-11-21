"""Training loop, Early Stopping, multi-seed runner, and paired t-test utilities."""

from __future__ import annotations

import copy
import math
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr, ttest_rel
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.loader import DataLoader

from src.training.metrics import compute_multitask_loss, evaluate_predictions
from src.utils.seed import OFFICIAL_SEEDS, seed_everything


class EarlyStopping:
    """Stop training when validation metric stops improving and keep best weights."""

    def __init__(
        self,
        patience: int = 10,
        mode: Literal["min", "max"] = "min",
        min_delta: float = 1e-6,
    ) -> None:
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")

        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score: float | None = None
        self.best_state: dict[str, torch.Tensor] | None = None
        self.counter = 0
        self.should_stop = False

    def _is_improved(self, score: float) -> bool:
        if not math.isfinite(float(score)):
            return False
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta

    def step(self, score: float, model: nn.Module) -> bool:
        if self._is_improved(score):
            self.best_score = float(score)
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class _ModelEMA:
    """Exponential Moving Average tracker with step-debiasing for stable validation."""

    def __init__(self, model: nn.Module, decay: float = 0.985) -> None:
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow_params: dict[str, torch.Tensor] = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        self.buffers: dict[str, torch.Tensor] = {
            name: b.detach().clone() for name, b in model.named_buffers()
        }

    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        eff_decay = min(
            self.decay,
            (1.0 + float(self.num_updates)) / (10.0 + float(self.num_updates)),
        )
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad and name in self.shadow_params:
                    self.shadow_params[name].mul_(eff_decay).add_(
                        p.detach(), alpha=1.0 - eff_decay
                    )
            for name, b in model.named_buffers():
                if name in self.buffers:
                    self.buffers[name].copy_(b.detach())

    def copy_to(self, target_model: nn.Module) -> None:
        with torch.no_grad():
            for name, p in target_model.named_parameters():
                if name in self.shadow_params:
                    p.copy_(self.shadow_params[name])
            for name, b in target_model.named_buffers():
                if name in self.buffers:
                    b.copy_(self.buffers[name])


def _estimate_train_target_stats(loader: DataLoader) -> dict[str, Any]:
    """Compute target means and inverse-std loss scales strictly from `train_loader`."""
    size_vals: list[np.ndarray] = []
    polar_vals: list[np.ndarray] = []
    for batch in loader:
        ys = getattr(batch, "y_size", None)
        yp = getattr(batch, "y_polar", None)
        if ys is not None:
            size_vals.append(ys.view(-1).detach().cpu().numpy())
        if yp is not None:
            polar_vals.append(yp.view(-1).detach().cpu().numpy())

    size_arr = np.concatenate(size_vals) if size_vals else np.array([], dtype=np.float64)
    polar_arr = np.concatenate(polar_vals) if polar_vals else np.array([], dtype=np.float64)
    size_arr = size_arr[np.isfinite(size_arr)] if size_arr.size > 0 else size_arr
    polar_arr = polar_arr[np.isfinite(polar_arr)] if polar_arr.size > 0 else polar_arr

    mean_size = float(np.mean(size_arr)) if size_arr.size > 0 else None
    std_size = float(np.std(size_arr)) if size_arr.size >= 2 else 0.0
    mean_polar = float(np.mean(polar_arr)) if polar_arr.size > 0 else None
    std_polar = float(np.std(polar_arr)) if polar_arr.size >= 2 else 0.0

    size_scale = (
        float(np.clip(1.0 / max(std_size, 0.25), 0.5, 4.0))
        if size_arr.size >= 2 and std_size > 1e-6
        else 1.0
    )
    polar_scale = (
        float(np.clip(1.0 / max(std_polar, 0.12), 0.5, 4.0))
        if polar_arr.size >= 2 and std_polar > 1e-6
        else 1.0
    )
    return {
        "count": int(min(size_arr.size, polar_arr.size)) if size_vals and polar_vals else int(size_arr.size),
        "mean_size": mean_size,
        "std_size": std_size,
        "mean_polar": mean_polar,
        "std_polar": std_polar,
        "size_loss_scale": size_scale,
        "polar_loss_scale": polar_scale,
    }


def _init_multitask_head_biases(model: nn.Module, stats: Mapping[str, Any]) -> None:
    """Initialize output biases of MultiTaskHead from train-set target means."""
    head = getattr(model, "head", None)
    if head is None:
        return
    with torch.no_grad():
        mean_size = stats.get("mean_size")
        if mean_size is not None and hasattr(head, "size_head") and len(head.size_head) > 0:
            last_size = head.size_head[-1]
            if getattr(last_size, "bias", None) is not None:
                size_bias = math.log(math.expm1(max(0.1, float(mean_size))))
                last_size.bias.fill_(float(size_bias))
        mean_polar = stats.get("mean_polar")
        if mean_polar is not None and hasattr(head, "polar_head") and len(head.polar_head) > 0:
            last_polar = head.polar_head[-1]
            if getattr(last_polar, "bias", None) is not None:
                p_clamped = min(0.95, max(0.05, float(mean_polar)))
                polar_bias = math.log(p_clamped / (1.0 - p_clamped))
                last_polar.bias.fill_(float(polar_bias))


def _estimate_balanced_class_weights(
    loader: DataLoader, num_classes: int
) -> Tensor | None:
    """Compute normalized inverse-frequency class weights strictly from `train_loader`."""
    if num_classes < 2:
        return None
    counts = np.zeros(num_classes, dtype=np.float64)
    for batch in loader:
        y_cls = getattr(batch, "y_rumor", getattr(batch, "y", None))
        if y_cls is None:
            continue
        arr = y_cls.view(-1).cpu().numpy()
        for c in arr:
            if 0 <= int(c) < num_classes:
                counts[int(c)] += 1.0
    present = counts > 0
    if not np.any(present):
        return None
    total = float(counts.sum())
    weights = np.ones(num_classes, dtype=np.float32)
    num_present = float(present.sum())
    # Dampened square-root inverse-frequency weighting prevents rare veracity classes
    # from dominating validation Early Stopping while preserving minority-class recall.
    weights[present] = np.sqrt(total / (num_present * counts[present])).astype(np.float32)
    weights[present] = weights[present] / float(np.mean(weights[present]))
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    alpha_size: float = 1.0,
    alpha_polar: float = 1.0,
    alpha_rumor: float = 1.0,
    size_loss_scale: float = 1.0,
    polar_loss_scale: float = 1.0,
    lambda_uncertainty: float = 0.0,
    class_weights: Tensor | None = None,
    classification_only: bool = False,
    return_predictions: bool = False,
) -> dict[str, Any]:
    """Evaluate a model over a PyG DataLoader and return loss, task_loss, and metrics."""
    model.eval()
    total_loss = 0.0
    total_task_loss = 0.0
    total_graphs = 0

    size_true_list: list[np.ndarray] = []
    size_pred_list: list[np.ndarray] = []
    polar_true_list: list[np.ndarray] = []
    polar_pred_list: list[np.ndarray] = []
    bimodality_true_list: list[np.ndarray] = []
    assortativity_true_list: list[np.ndarray] = []
    rumor_true_list: list[np.ndarray] = []
    rumor_logits_list: list[np.ndarray] = []
    topic_ids_list: list[str] = []

    for batch in loader:
        batch = batch.to(device)
        outputs = model(batch)
        loss = compute_multitask_loss(
            outputs,
            batch,
            alpha_size=alpha_size,
            alpha_polar=alpha_polar,
            alpha_rumor=alpha_rumor,
            size_loss_scale=size_loss_scale,
            polar_loss_scale=polar_loss_scale,
            lambda_uncertainty=lambda_uncertainty,
            class_weights=class_weights,
            classification_only=classification_only,
        )
        if lambda_uncertainty > 0.0 and "size_logvar" in outputs:
            t_loss = compute_multitask_loss(
                outputs,
                batch,
                alpha_size=alpha_size,
                alpha_polar=alpha_polar,
                alpha_rumor=alpha_rumor,
                size_loss_scale=size_loss_scale,
                polar_loss_scale=polar_loss_scale,
                lambda_uncertainty=0.0,
                class_weights=class_weights,
                classification_only=classification_only,
            )
        else:
            t_loss = loss
        num_graphs = int(batch.num_graphs)
        total_loss += float(loss.item()) * num_graphs
        total_task_loss += float(t_loss.item()) * num_graphs
        total_graphs += num_graphs

        if not classification_only:
            if hasattr(batch, "y_size") and batch.y_size is not None:
                size_true_list.append(batch.y_size.view(-1).detach().cpu().numpy())
                size_pred_list.append(outputs["y_size_pred"].view(-1).detach().cpu().numpy())
            if hasattr(batch, "y_polar") and batch.y_polar is not None:
                polar_true_list.append(batch.y_polar.view(-1).detach().cpu().numpy())
                polar_pred_list.append(outputs["y_polar_pred"].view(-1).detach().cpu().numpy())
            if hasattr(batch, "y_bimodality") and batch.y_bimodality is not None:
                bimodality_true_list.append(batch.y_bimodality.view(-1).detach().cpu().numpy())
            if hasattr(batch, "y_assortativity") and batch.y_assortativity is not None:
                assortativity_true_list.append(batch.y_assortativity.view(-1).detach().cpu().numpy())

        raw_tids = getattr(batch, "topic_id", None)
        if raw_tids is not None:
            if isinstance(raw_tids, (list, tuple)):
                topic_ids_list.extend(str(t) for t in raw_tids)
            else:
                topic_ids_list.append(str(raw_tids))

        y_rumor = getattr(batch, "y_rumor", getattr(batch, "y", None))
        if y_rumor is not None:
            rumor_true_list.append(y_rumor.view(-1).detach().cpu().numpy())
            rumor_logits_list.append(outputs["rumor_logits"].detach().cpu().numpy())

    y_size_true = np.concatenate(size_true_list) if size_true_list else np.array([])
    y_size_pred = np.concatenate(size_pred_list) if size_pred_list else np.array([])
    y_polar_true = np.concatenate(polar_true_list) if polar_true_list else np.array([])
    y_polar_pred = np.concatenate(polar_pred_list) if polar_pred_list else np.array([])
    y_bimodality_true = np.concatenate(bimodality_true_list) if bimodality_true_list else None
    y_assortativity_true = (
        np.concatenate(assortativity_true_list) if assortativity_true_list else None
    )
    y_rumor_true = np.concatenate(rumor_true_list) if rumor_true_list else None
    rumor_logits = np.concatenate(rumor_logits_list, axis=0) if rumor_logits_list else None

    metrics: dict[str, Any] = evaluate_predictions(
        y_size_true=y_size_true,
        y_size_pred=y_size_pred,
        y_polar_true=y_polar_true,
        y_polar_pred=y_polar_pred,
        y_rumor_true=y_rumor_true,
        rumor_logits=rumor_logits,
        y_bimodality_true=y_bimodality_true,
        y_assortativity_true=y_assortativity_true,
    )
    metrics["loss"] = total_loss / max(1, total_graphs)
    metrics["task_loss"] = total_task_loss / max(1, total_graphs)
    if return_predictions:
        expected_n = (
            int(y_size_true.size)
            if y_size_true.size > 0
            else (int(y_rumor_true.size) if y_rumor_true is not None else 0)
        )
        metrics["_predictions"] = {
            "y_size_true": y_size_true,
            "y_size_pred": y_size_pred,
            "y_polar_true": y_polar_true,
            "y_polar_pred": y_polar_pred,
            "y_rumor_true": y_rumor_true,
            "rumor_logits": rumor_logits,
            "topic_ids": topic_ids_list if len(topic_ids_list) == expected_n else None,
        }
    return metrics


def train_and_evaluate_single_seed(
    model_fn: Callable[[], nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    *,
    seed: int,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    alpha_size: float = 1.0,
    alpha_polar: float = 1.0,
    alpha_rumor: float = 1.0,
    lambda_uncertainty: float = 0.0,
    use_class_weights: bool = True,
    use_target_scaling: bool = True,
    use_bias_init: bool = True,
    use_ema: bool = True,
    ema_decay: float = 0.985,
    classification_only: bool = False,
    device: torch.device | str | None = None,
    return_predictions: bool = False,
) -> tuple[nn.Module, dict[str, Any]]:
    """Train one model for a single random seed with warmup, ReduceLROnPlateau, EMA & Early Stopping."""
    seed_everything(seed)
    target_device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = model_fn().to(target_device)
    class_weights: Tensor | None = None
    if use_class_weights:
        num_cls = None
        if hasattr(model, "head") and hasattr(model.head, "rumor_head"):
            last_layer = model.head.rumor_head[-1]
            if hasattr(last_layer, "out_features"):
                num_cls = int(last_layer.out_features)
        if num_cls is not None:
            class_weights = _estimate_balanced_class_weights(train_loader, num_cls)

    s_scale = 1.0
    p_scale = 1.0
    if not classification_only:
        train_stats = _estimate_train_target_stats(train_loader)
        if use_bias_init and train_stats["count"] > 0:
            _init_multitask_head_biases(model, train_stats)
        if use_target_scaling:
            s_scale = float(train_stats["size_loss_scale"])
            p_scale = float(train_stats["polar_loss_scale"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup_epochs = min(5, max(1, epochs // 10)) if epochs >= 10 else 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(3, patience // 2),
        min_lr=lr * 0.02,
    )
    ema_active = bool(use_ema and epochs >= 10)
    ema = _ModelEMA(model, decay=ema_decay) if ema_active else None
    ema_eval_model = copy.deepcopy(model) if ema_active else None
    early_stopper = EarlyStopping(patience=patience, mode="min")

    # Reseed deterministically after model construction and attach a dedicated per-seed
    # torch.Generator to shuffled train_loaders so every architecture receives the exact
    # same minibatch order across ALL epochs regardless of internal Dropout/RNG draws.
    seed_everything(int(seed) + 10000)
    active_train_loader = train_loader
    if isinstance(
        getattr(train_loader, "sampler", None), torch.utils.data.RandomSampler
    ) and hasattr(train_loader, "dataset"):
        shuffle_gen = torch.Generator()
        shuffle_gen.manual_seed(int(seed) + 10000)
        try:
            active_train_loader = DataLoader(
                train_loader.dataset,
                batch_size=train_loader.batch_size or 1,
                shuffle=True,
                drop_last=bool(getattr(train_loader, "drop_last", False)),
                num_workers=int(getattr(train_loader, "num_workers", 0)),
                pin_memory=bool(getattr(train_loader, "pin_memory", False)),
                generator=shuffle_gen,
            )
        except Exception:
            active_train_loader = train_loader

    for epoch in range(epochs):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            warmup_lr = lr * float(epoch + 1) / float(warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        model.train()
        for batch in active_train_loader:
            batch = batch.to(target_device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = compute_multitask_loss(
                outputs,
                batch,
                alpha_size=alpha_size,
                alpha_polar=alpha_polar,
                alpha_rumor=alpha_rumor,
                size_loss_scale=s_scale,
                polar_loss_scale=p_scale,
                lambda_uncertainty=lambda_uncertainty,
                class_weights=class_weights,
                classification_only=classification_only,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            if ema is not None:
                ema.update(model)

        raw_val_metrics = evaluate_loader(
            model,
            val_loader,
            target_device,
            alpha_size=alpha_size,
            alpha_polar=alpha_polar,
            alpha_rumor=alpha_rumor,
            size_loss_scale=s_scale,
            polar_loss_scale=p_scale,
            lambda_uncertainty=lambda_uncertainty,
            class_weights=class_weights,
            classification_only=classification_only,
        )
        raw_val_score = float(raw_val_metrics.get("task_loss", raw_val_metrics["loss"]))
        chosen_model = model
        chosen_score = raw_val_score

        if ema is not None and ema_eval_model is not None:
            ema.copy_to(ema_eval_model)
            ema_val_metrics = evaluate_loader(
                ema_eval_model,
                val_loader,
                target_device,
                alpha_size=alpha_size,
                alpha_polar=alpha_polar,
                alpha_rumor=alpha_rumor,
                size_loss_scale=s_scale,
                polar_loss_scale=p_scale,
                lambda_uncertainty=lambda_uncertainty,
                class_weights=class_weights,
                classification_only=classification_only,
            )
            ema_val_score = float(ema_val_metrics.get("task_loss", ema_val_metrics["loss"]))
            if math.isfinite(ema_val_score) and ema_val_score <= raw_val_score:
                chosen_model = ema_eval_model
                chosen_score = ema_val_score

        if epoch >= warmup_epochs and math.isfinite(chosen_score):
            scheduler.step(chosen_score)

        if early_stopper.step(chosen_score, chosen_model):
            break

    early_stopper.restore_best(model)
    test_metrics = evaluate_loader(
        model,
        test_loader,
        target_device,
        alpha_size=alpha_size,
        alpha_polar=alpha_polar,
        alpha_rumor=alpha_rumor,
        size_loss_scale=s_scale,
        polar_loss_scale=p_scale,
        lambda_uncertainty=lambda_uncertainty,
        class_weights=class_weights,
        classification_only=classification_only,
        return_predictions=return_predictions,
    )
    test_metrics["val_task_loss"] = (
        float(early_stopper.best_score)
        if early_stopper.best_score is not None
        else float("nan")
    )
    return model, test_metrics


def compute_paired_ttest(
    model_a_scores: Sequence[float],
    model_b_scores: Sequence[float],
) -> float:
    """Run two-sided paired Student's t-test (`scipy.stats.ttest_rel`) across seeds."""
    arr_a = np.asarray(model_a_scores, dtype=np.float64)
    arr_b = np.asarray(model_b_scores, dtype=np.float64)
    valid = ~(np.isnan(arr_a) | np.isnan(arr_b))
    arr_a, arr_b = arr_a[valid], arr_b[valid]
    if arr_a.size < 2:
        return float("nan")
    if np.allclose(arr_a - arr_b, 0.0):
        return 1.0
    res = ttest_rel(arr_a, arr_b)
    p_val = float(res.pvalue)
    return 1.0 if np.isnan(p_val) else p_val


def compute_paired_bootstrap_pvalue(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    n_resamples: int = 2000,
    seed: int = 42,
) -> float:
    """Two-sided paired test: exact sign-flip permutation for n < 6, +1-smoothed bootstrap for n >= 6."""
    arr_a = np.asarray(scores_a, dtype=np.float64)
    arr_b = np.asarray(scores_b, dtype=np.float64)
    valid = ~(np.isnan(arr_a) | np.isnan(arr_b))
    arr_a, arr_b = arr_a[valid], arr_b[valid]
    n_obs = int(arr_a.size)
    if n_obs < 2:
        return float("nan")
    diffs = arr_a - arr_b
    if np.allclose(diffs, 0.0):
        return 1.0
    if n_obs < 6:
        # Exact two-sided paired sign-flip permutation test over all 2^n flips
        # (prevents small-sample bootstrap collapse such as fake p=0.0000 at F=4 LOO folds).
        obs_mean = float(np.mean(diffs))
        signs = np.array(
            [
                [1.0 if ((mask >> bit) & 1) else -1.0 for bit in range(n_obs)]
                for mask in range(1 << n_obs)
            ],
            dtype=np.float64,
        )
        perm_means = (signs * diffs[None, :]).mean(axis=1)
        p_right = float(np.mean(perm_means >= obs_mean - 1e-12))
        p_left = float(np.mean(perm_means <= obs_mean + 1e-12))
        return float(min(1.0, 2.0 * min(p_left, p_right)))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_obs, size=(n_resamples, n_obs))
    boot_means = diffs[idx].mean(axis=1)
    p_left = float((np.sum(boot_means <= 0.0) + 1.0) / (n_resamples + 1.0))
    p_right = float((np.sum(boot_means >= 0.0) + 1.0) / (n_resamples + 1.0))
    return float(min(1.0, 2.0 * min(p_left, p_right)))


_LOWER_IS_BETTER: frozenset[str] = frozenset(
    {"msle", "mae", "rmse_size", "rmse", "mae_polar", "loss"}
)
_COMPARATOR_RANK_METRICS: tuple[str, ...] = (
    "msle",
    "mae",
    "r2_size",
    "rmse",
    "mae_polar",
    "r2_polar",
    "pearson_r",
    "spearman_polar",
    "macro_f1",
    "accuracy",
    "auc_roc",
)
_COMPARATOR_TIE_PRIORITY: dict[str, int] = {
    "BiGCN": 0,
    "FAGCN": 1,
    "EdgeGATv2": 2,
    "TemporalStanceMLP": 3,
    "StandardGATv2": 4,
    "StandardGCN": 5,
    "TextMLP": 6,
}


def holm_bonferroni_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Apply Holm-Bonferroni step-down family-wise error adjustment to p-values."""
    adjusted: dict[str, float] = {k: float(v) for k, v in p_values.items()}
    valid = [
        (k, float(v))
        for k, v in p_values.items()
        if v is not None and not math.isnan(float(v)) and math.isfinite(float(v))
    ]
    if not valid:
        return adjusted
    m = len(valid)
    valid.sort(key=lambda kv: kv[1])
    running_max = 0.0
    for rank, (k, raw_p) in enumerate(valid):
        scaled = min(1.0, max(0.0, float(m - rank) * raw_p))
        running_max = max(running_max, scaled)
        adjusted[k] = running_max
    return adjusted


_NON_LEARNED_BASELINES: frozenset[str] = frozenset({"Early-K Persistence"})


def select_best_comparator(
    summary_stats: Mapping[str, Mapping[str, tuple[float, float]]],
    *,
    reference_model: str = "SBiFAGAT",
    metric_keys: Sequence[str] | None = None,
) -> str:
    """Choose the strongest competing learned baseline by average rank across non-NaN metrics."""
    candidates = [
        m
        for m in summary_stats.keys()
        if m != reference_model and m not in _NON_LEARNED_BASELINES
    ]
    if not candidates:
        candidates = [m for m in summary_stats.keys() if m != reference_model]
    if not candidates:
        return "BiGCN"

    active_rank_metrics = (
        tuple(m for m in metric_keys if m in _COMPARATOR_RANK_METRICS)
        if metric_keys is not None
        else _COMPARATOR_RANK_METRICS
    ) or _COMPARATOR_RANK_METRICS

    ranks_by_model: dict[str, list[float]] = {c: [] for c in candidates}
    for metric in active_rank_metrics:
        scored: list[tuple[str, float]] = []
        for c in candidates:
            stat = summary_stats.get(c, {}).get(metric)
            if stat is not None and not math.isnan(float(stat[0])):
                scored.append((c, float(stat[0])))
        if len(scored) < 2:
            continue
        reverse = metric not in _LOWER_IS_BETTER
        scored.sort(key=lambda kv: kv[1], reverse=reverse)
        for rank_idx, (c, _) in enumerate(scored):
            ranks_by_model[c].append(float(rank_idx))

    if not any(ranks_by_model.values()):
        return min(candidates, key=lambda c: (_COMPARATOR_TIE_PRIORITY.get(c, 99), c))

    return min(
        candidates,
        key=lambda c: (
            float(np.mean(ranks_by_model[c])) if ranks_by_model[c] else float("inf"),
            -len(ranks_by_model[c]),
            _COMPARATOR_TIE_PRIORITY.get(c, 99),
            c,
        ),
    )


def _boot_pvalue_from_diffs(diffs: np.ndarray) -> float:
    arr = np.asarray(diffs, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return float("nan")
    if np.allclose(arr, 0.0):
        return 1.0
    p_left = float((np.sum(arr <= 0.0) + 1.0) / (arr.size + 1.0))
    p_right = float((np.sum(arr >= 0.0) + 1.0) / (arr.size + 1.0))
    return float(min(1.0, 2.0 * min(p_left, p_right)))


def _vec_r2(y_true: np.ndarray, y_pred: np.ndarray, idx_mat: np.ndarray) -> np.ndarray:
    yt_b = y_true[idx_mat]
    yp_b = y_pred[idx_mat]
    ss_res = np.sum((yt_b - yp_b) ** 2, axis=1)
    ss_tot = np.sum((yt_b - np.mean(yt_b, axis=1, keepdims=True)) ** 2, axis=1)
    out = np.zeros(idx_mat.shape[0], dtype=np.float64)
    mask = ss_tot > 1e-12
    out[mask] = 1.0 - ss_res[mask] / ss_tot[mask]
    return out


def _vec_pearson(y_true: np.ndarray, y_pred: np.ndarray, idx_mat: np.ndarray) -> np.ndarray:
    yt_b = y_true[idx_mat]
    yp_b = y_pred[idx_mat]
    yt_c = yt_b - np.mean(yt_b, axis=1, keepdims=True)
    yp_c = yp_b - np.mean(yp_b, axis=1, keepdims=True)
    denom = np.sqrt(np.sum(yt_c**2, axis=1) * np.sum(yp_c**2, axis=1))
    out = np.zeros(idx_mat.shape[0], dtype=np.float64)
    mask = denom > 1e-12
    out[mask] = np.sum(yt_c[mask] * yp_c[mask], axis=1) / denom[mask]
    return out


def _safe_auc_from_probs(truth: np.ndarray, probs: np.ndarray) -> float:
    unique_cls = np.unique(truth)
    if unique_cls.size < 2 or probs.ndim != 2:
        return float("nan")
    try:
        if probs.shape[1] == 2:
            return float(roc_auc_score(truth, probs[:, 1]))
        if unique_cls.size == probs.shape[1]:
            return float(roc_auc_score(truth, probs, multi_class="ovr", average="macro"))
        cls_aucs = [
            float(roc_auc_score((truth == c).astype(np.int64), probs[:, int(c)]))
            for c in unique_cls
            if int(c) < probs.shape[1]
        ]
        return float(np.mean(cls_aucs)) if cls_aucs else float("nan")
    except ValueError:
        return float("nan")


def _compute_sample_level_bootstrap_pvalues(
    preds_ref: Sequence[dict[str, Any]],
    preds_cmp: Sequence[dict[str, Any]],
    *,
    metric_keys: Sequence[str],
    seed: int = 42,
    n_resamples: int = 5000,
    n_stat_resamples: int = 2500,
) -> dict[str, float]:
    """Paired bootstrap across held-out test cascades averaging per-seed metrics per draw.

    Defaults (`n_resamples=5000`, `n_stat_resamples=2500`) ensure that even after `+1`
    permutation smoothing and Holm-Bonferroni adjustment across up to 8–9 metrics, the
    minimum achievable adjusted p-value (`<= 9 * 2 / 2501 = 0.0072`) stays below `0.01`.
    """
    if not preds_ref or not preds_cmp:
        return {k: float("nan") for k in metric_keys}
    y_size_true = np.asarray(preds_ref[0]["y_size_true"], dtype=np.float64).reshape(-1)
    y_polar_true = np.asarray(preds_ref[0]["y_polar_true"], dtype=np.float64).reshape(-1)
    y_rumor_raw = preds_ref[0].get("y_rumor_true")
    y_rumor_len = (
        int(np.asarray(y_rumor_raw, dtype=np.int64).reshape(-1).size)
        if y_rumor_raw is not None
        else 0
    )
    n_reg = int(y_size_true.size)
    n_total = n_reg if n_reg >= 2 else y_rumor_len
    if n_total < 2:
        return {k: float("nan") for k in metric_keys}

    rng = np.random.default_rng(seed)
    raw_tids = preds_ref[0].get("topic_ids")
    topic_groups: list[np.ndarray] | None = None
    if isinstance(raw_tids, (list, tuple)) and len(raw_tids) == n_total:
        group_map: dict[str, list[int]] = {}
        for idx_i, tid in enumerate(raw_tids):
            group_map.setdefault(str(tid), []).append(idx_i)
        if len(group_map) >= 6:
            topic_groups = [np.asarray(idxs, dtype=np.int64) for idxs in group_map.values()]

    cluster_draws: list[np.ndarray] | None = None
    n_draws = max(2, int(n_resamples))
    n_stat_draws = min(n_draws, max(2, int(n_stat_resamples)))
    if topic_groups is not None:
        # Topic-cluster bootstrap prevents intra-topic correlation inflation when >= 6 test topics exist
        n_topics = len(topic_groups)
        cluster_draws = [
            np.concatenate([topic_groups[g] for g in rng.integers(0, n_topics, size=n_topics)])
            for _ in range(n_draws)
        ]

    out: dict[str, float] = {}

    if n_reg >= 2:
        ref_size_seeds = [np.asarray(p["y_size_pred"], dtype=np.float64).reshape(-1) for p in preds_ref]
        cmp_size_seeds = [np.asarray(p["y_size_pred"], dtype=np.float64).reshape(-1) for p in preds_cmp]
        ref_polar_seeds = [np.asarray(p["y_polar_pred"], dtype=np.float64).reshape(-1) for p in preds_ref]
        cmp_polar_seeds = [np.asarray(p["y_polar_pred"], dtype=np.float64).reshape(-1) for p in preds_cmp]

        se_ref_size_s = [(yp - y_size_true) ** 2 for yp in ref_size_seeds]
        se_cmp_size_s = [(yp - y_size_true) ** 2 for yp in cmp_size_seeds]
        se_ref_polar_s = [(yp - y_polar_true) ** 2 for yp in ref_polar_seeds]
        se_cmp_polar_s = [(yp - y_polar_true) ** 2 for yp in cmp_polar_seeds]
        ae_ref_size_s = [np.abs(yp - y_size_true) for yp in ref_size_seeds]
        ae_cmp_size_s = [np.abs(yp - y_size_true) for yp in cmp_size_seeds]
        ae_ref_polar_s = [np.abs(yp - y_polar_true) for yp in ref_polar_seeds]
        ae_cmp_polar_s = [np.abs(yp - y_polar_true) for yp in cmp_polar_seeds]

        def _safe_sp(yt_arr: np.ndarray, yp_arr: np.ndarray) -> float:
            if yt_arr.size < 2 or np.std(yt_arr) <= 1e-12 or np.std(yp_arr) <= 1e-12:
                return 0.0
            val = float(spearmanr(yt_arr, yp_arr).statistic)
            return 0.0 if math.isnan(val) else val

        def _safe_r2_1d(yt_arr: np.ndarray, yp_arr: np.ndarray) -> float:
            if yt_arr.size < 2:
                return 0.0
            ss_tot = float(np.sum((yt_arr - np.mean(yt_arr)) ** 2))
            if ss_tot <= 1e-12:
                return 0.0
            return float(1.0 - np.sum((yt_arr - yp_arr) ** 2) / ss_tot)

        def _safe_pr_1d(yt_arr: np.ndarray, yp_arr: np.ndarray) -> float:
            if yt_arr.size < 2:
                return 0.0
            yt_c = yt_arr - np.mean(yt_arr)
            yp_c = yp_arr - np.mean(yp_arr)
            denom = float(np.sqrt(np.sum(yt_c**2) * np.sum(yp_c**2)))
            if denom <= 1e-12:
                return 0.0
            return float(np.sum(yt_c * yp_c) / denom)

        if cluster_draws is not None:
            msle_d, rmses_d, mae_d, rmsep_d, maep_d = [], [], [], [], []
            r2s_d, r2p_d, pr_d, sps_d, spp_d = [], [], [], [], []
            for b_pos, b_idx in enumerate(cluster_draws):
                msle_d.append(
                    float(np.mean([se[b_idx].mean() for se in se_ref_size_s]))
                    - float(np.mean([se[b_idx].mean() for se in se_cmp_size_s]))
                )
                rmses_d.append(
                    float(np.mean([math.sqrt(float(se[b_idx].mean())) for se in se_ref_size_s]))
                    - float(np.mean([math.sqrt(float(se[b_idx].mean())) for se in se_cmp_size_s]))
                )
                mae_d.append(
                    float(np.mean([ae[b_idx].mean() for ae in ae_ref_size_s]))
                    - float(np.mean([ae[b_idx].mean() for ae in ae_cmp_size_s]))
                )
                rmsep_d.append(
                    float(np.mean([math.sqrt(float(se[b_idx].mean())) for se in se_ref_polar_s]))
                    - float(np.mean([math.sqrt(float(se[b_idx].mean())) for se in se_cmp_polar_s]))
                )
                maep_d.append(
                    float(np.mean([ae[b_idx].mean() for ae in ae_ref_polar_s]))
                    - float(np.mean([ae[b_idx].mean() for ae in ae_cmp_polar_s]))
                )
                if b_pos < n_stat_draws:
                    yt_s = y_size_true[b_idx]
                    yt_p = y_polar_true[b_idx]
                    r2s_d.append(
                        float(np.mean([_safe_r2_1d(yt_s, yp[b_idx]) for yp in ref_size_seeds]))
                        - float(np.mean([_safe_r2_1d(yt_s, yp[b_idx]) for yp in cmp_size_seeds]))
                    )
                    r2p_d.append(
                        float(np.mean([_safe_r2_1d(yt_p, yp[b_idx]) for yp in ref_polar_seeds]))
                        - float(np.mean([_safe_r2_1d(yt_p, yp[b_idx]) for yp in cmp_polar_seeds]))
                    )
                    pr_d.append(
                        float(np.mean([_safe_pr_1d(yt_p, yp[b_idx]) for yp in ref_polar_seeds]))
                        - float(np.mean([_safe_pr_1d(yt_p, yp[b_idx]) for yp in cmp_polar_seeds]))
                    )
                    if "spearman_size" in metric_keys:
                        sps_d.append(
                            float(np.mean([_safe_sp(yt_s, yp[b_idx]) for yp in ref_size_seeds]))
                            - float(np.mean([_safe_sp(yt_s, yp[b_idx]) for yp in cmp_size_seeds]))
                        )
                    if "spearman_polar" in metric_keys:
                        spp_d.append(
                            float(np.mean([_safe_sp(yt_p, yp[b_idx]) for yp in ref_polar_seeds]))
                            - float(np.mean([_safe_sp(yt_p, yp[b_idx]) for yp in cmp_polar_seeds]))
                        )
            out["msle"] = _boot_pvalue_from_diffs(np.asarray(msle_d, dtype=np.float64))
            out["rmse_size"] = _boot_pvalue_from_diffs(np.asarray(rmses_d, dtype=np.float64))
            out["mae"] = _boot_pvalue_from_diffs(np.asarray(mae_d, dtype=np.float64))
            out["rmse"] = _boot_pvalue_from_diffs(np.asarray(rmsep_d, dtype=np.float64))
            out["mae_polar"] = _boot_pvalue_from_diffs(np.asarray(maep_d, dtype=np.float64))
            out["r2_size"] = _boot_pvalue_from_diffs(np.asarray(r2s_d, dtype=np.float64))
            out["r2_polar"] = _boot_pvalue_from_diffs(np.asarray(r2p_d, dtype=np.float64))
            out["pearson_r"] = _boot_pvalue_from_diffs(np.asarray(pr_d, dtype=np.float64))
            if sps_d:
                out["spearman_size"] = _boot_pvalue_from_diffs(np.asarray(sps_d, dtype=np.float64))
            if spp_d:
                out["spearman_polar"] = _boot_pvalue_from_diffs(np.asarray(spp_d, dtype=np.float64))
        else:
            idx_fast = rng.integers(0, n_reg, size=(n_draws, n_reg))
            idx_stat = idx_fast[:n_stat_draws]

            ref_msle_b = np.mean([se[idx_fast].mean(axis=1) for se in se_ref_size_s], axis=0)
            cmp_msle_b = np.mean([se[idx_fast].mean(axis=1) for se in se_cmp_size_s], axis=0)
            out["msle"] = _boot_pvalue_from_diffs(ref_msle_b - cmp_msle_b)

            ref_rmses_b = np.mean([np.sqrt(se[idx_fast].mean(axis=1)) for se in se_ref_size_s], axis=0)
            cmp_rmses_b = np.mean([np.sqrt(se[idx_fast].mean(axis=1)) for se in se_cmp_size_s], axis=0)
            out["rmse_size"] = _boot_pvalue_from_diffs(ref_rmses_b - cmp_rmses_b)

            ae_ref_size = np.mean(ae_ref_size_s, axis=0)
            ae_cmp_size = np.mean(ae_cmp_size_s, axis=0)
            out["mae"] = compute_paired_bootstrap_pvalue(ae_ref_size, ae_cmp_size, n_resamples=n_draws, seed=seed + 1)

            ref_rmsep_b = np.mean([np.sqrt(se[idx_fast].mean(axis=1)) for se in se_ref_polar_s], axis=0)
            cmp_rmsep_b = np.mean([np.sqrt(se[idx_fast].mean(axis=1)) for se in se_cmp_polar_s], axis=0)
            out["rmse"] = _boot_pvalue_from_diffs(ref_rmsep_b - cmp_rmsep_b)

            ae_ref_polar = np.mean(ae_ref_polar_s, axis=0)
            ae_cmp_polar = np.mean(ae_cmp_polar_s, axis=0)
            out["mae_polar"] = compute_paired_bootstrap_pvalue(ae_ref_polar, ae_cmp_polar, n_resamples=n_draws, seed=seed + 3)

            out["r2_size"] = _boot_pvalue_from_diffs(
                np.mean([_vec_r2(y_size_true, yp, idx_stat) for yp in ref_size_seeds], axis=0)
                - np.mean([_vec_r2(y_size_true, yp, idx_stat) for yp in cmp_size_seeds], axis=0)
            )
            out["r2_polar"] = _boot_pvalue_from_diffs(
                np.mean([_vec_r2(y_polar_true, yp, idx_stat) for yp in ref_polar_seeds], axis=0)
                - np.mean([_vec_r2(y_polar_true, yp, idx_stat) for yp in cmp_polar_seeds], axis=0)
            )
            out["pearson_r"] = _boot_pvalue_from_diffs(
                np.mean([_vec_pearson(y_polar_true, yp, idx_stat) for yp in ref_polar_seeds], axis=0)
                - np.mean([_vec_pearson(y_polar_true, yp, idx_stat) for yp in cmp_polar_seeds], axis=0)
            )

            if "spearman_size" in metric_keys:
                sp_size_diffs: list[float] = []
                for b_idx in idx_stat:
                    yt = y_size_true[b_idx]
                    sr = float(np.mean([_safe_sp(yt, yp[b_idx]) for yp in ref_size_seeds]))
                    sc = float(np.mean([_safe_sp(yt, yp[b_idx]) for yp in cmp_size_seeds]))
                    sp_size_diffs.append(sr - sc)
                out["spearman_size"] = _boot_pvalue_from_diffs(np.asarray(sp_size_diffs, dtype=np.float64))

            if "spearman_polar" in metric_keys:
                sp_polar_diffs: list[float] = []
                for b_idx in idx_stat:
                    yt = y_polar_true[b_idx]
                    sr = float(np.mean([_safe_sp(yt, yp[b_idx]) for yp in ref_polar_seeds]))
                    sc = float(np.mean([_safe_sp(yt, yp[b_idx]) for yp in cmp_polar_seeds]))
                    sp_polar_diffs.append(sr - sc)
                out["spearman_polar"] = _boot_pvalue_from_diffs(np.asarray(sp_polar_diffs, dtype=np.float64))

    ref_logits_list = [p["rumor_logits"] for p in preds_ref if p.get("rumor_logits") is not None]
    cmp_logits_list = [p["rumor_logits"] for p in preds_cmp if p.get("rumor_logits") is not None]
    if ref_logits_list and cmp_logits_list and y_rumor_raw is not None:
        y_rumor_arr = np.asarray(y_rumor_raw, dtype=np.int64).reshape(-1)
        valid = y_rumor_arr >= 0
        n_cls_samples = int(np.sum(valid))
        if n_cls_samples >= 2:
            if cluster_draws is not None and y_rumor_arr.size == n_total:
                # Apply the same topic-cluster draws to classification metrics (filtered by valid labels)
                ref_probs_full = [
                    torch.softmax(torch.as_tensor(lg), dim=-1).numpy() for lg in ref_logits_list
                ]
                cmp_probs_full = [
                    torch.softmax(torch.as_tensor(lg), dim=-1).numpy() for lg in cmp_logits_list
                ]
                ref_cls_full = [np.argmax(pr, axis=-1) for pr in ref_probs_full]
                cmp_cls_full = [np.argmax(pr, axis=-1) for pr in cmp_probs_full]
                acc_diffs_c: list[float] = []
                f1_diffs_c: list[float] = []
                auc_diffs_c: list[float] = []
                for b_pos, b_idx in enumerate(cluster_draws):
                    v_idx = b_idx[valid[b_idx]]
                    if v_idx.size < 2:
                        continue
                    tb = y_rumor_arr[v_idx]
                    ar = float(np.mean([(pc[v_idx] == tb).mean() for pc in ref_cls_full]))
                    ac = float(np.mean([(pc[v_idx] == tb).mean() for pc in cmp_cls_full]))
                    acc_diffs_c.append(ar - ac)
                    if b_pos < n_stat_draws:
                        if "macro_f1" in metric_keys:
                            f1_r = float(np.mean([f1_score(tb, pc[v_idx], average="macro", zero_division=0) for pc in ref_cls_full]))
                            f1_c = float(np.mean([f1_score(tb, pc[v_idx], average="macro", zero_division=0) for pc in cmp_cls_full]))
                            f1_diffs_c.append(f1_r - f1_c)
                        if "auc_roc" in metric_keys:
                            ars = [_safe_auc_from_probs(tb, pr[v_idx]) for pr in ref_probs_full]
                            acs = [_safe_auc_from_probs(tb, pr[v_idx]) for pr in cmp_probs_full]
                            ars_v = [a for a in ars if not math.isnan(a)]
                            acs_v = [a for a in acs if not math.isnan(a)]
                            if ars_v and acs_v:
                                auc_diffs_c.append(float(np.mean(ars_v)) - float(np.mean(acs_v)))
                out["accuracy"] = (
                    _boot_pvalue_from_diffs(np.asarray(acc_diffs_c, dtype=np.float64))
                    if len(acc_diffs_c) >= 2
                    else float("nan")
                )
                if "macro_f1" in metric_keys:
                    out["macro_f1"] = (
                        _boot_pvalue_from_diffs(np.asarray(f1_diffs_c, dtype=np.float64))
                        if len(f1_diffs_c) >= 2
                        else float("nan")
                    )
                if "auc_roc" in metric_keys:
                    out["auc_roc"] = (
                        _boot_pvalue_from_diffs(np.asarray(auc_diffs_c, dtype=np.float64))
                        if len(auc_diffs_c) >= 2
                        else float("nan")
                    )
            else:
                truth_v = y_rumor_arr[valid]
                ref_probs_seeds = [
                    torch.softmax(torch.as_tensor(lg), dim=-1).numpy()[valid] for lg in ref_logits_list
                ]
                cmp_probs_seeds = [
                    torch.softmax(torch.as_tensor(lg), dim=-1).numpy()[valid] for lg in cmp_logits_list
                ]
                ref_cls_seeds = [np.argmax(pr, axis=-1) for pr in ref_probs_seeds]
                cmp_cls_seeds = [np.argmax(pr, axis=-1) for pr in cmp_probs_seeds]

                acc_ref_mean = np.mean([(pc == truth_v).astype(np.float64) for pc in ref_cls_seeds], axis=0)
                acc_cmp_mean = np.mean([(pc == truth_v).astype(np.float64) for pc in cmp_cls_seeds], axis=0)
                out["accuracy"] = compute_paired_bootstrap_pvalue(
                    acc_ref_mean,
                    acc_cmp_mean,
                    n_resamples=n_draws,
                    seed=seed + 4,
                )
                idx_cls = rng.integers(0, n_cls_samples, size=(n_stat_draws, n_cls_samples))
                if "macro_f1" in metric_keys:
                    f1_diffs: list[float] = []
                    for b_idx in idx_cls:
                        tb = truth_v[b_idx]
                        f1_r = float(np.mean([f1_score(tb, pc[b_idx], average="macro", zero_division=0) for pc in ref_cls_seeds]))
                        f1_c = float(np.mean([f1_score(tb, pc[b_idx], average="macro", zero_division=0) for pc in cmp_cls_seeds]))
                        f1_diffs.append(f1_r - f1_c)
                    out["macro_f1"] = _boot_pvalue_from_diffs(np.asarray(f1_diffs, dtype=np.float64))
                if "auc_roc" in metric_keys:
                    auc_diffs: list[float] = []
                    for b_idx in idx_cls:
                        tb = truth_v[b_idx]
                        ars = [_safe_auc_from_probs(tb, pr[b_idx]) for pr in ref_probs_seeds]
                        acs = [_safe_auc_from_probs(tb, pr[b_idx]) for pr in cmp_probs_seeds]
                        ars_v = [a for a in ars if not math.isnan(a)]
                        acs_v = [a for a in acs if not math.isnan(a)]
                        if ars_v and acs_v:
                            auc_diffs.append(float(np.mean(ars_v)) - float(np.mean(acs_v)))
                    out["auc_roc"] = (
                        _boot_pvalue_from_diffs(np.asarray(auc_diffs, dtype=np.float64))
                        if len(auc_diffs) >= 2
                        else float("nan")
                    )

    for k in metric_keys:
        out.setdefault(k, float("nan"))
    return out


def run_multi_seed_benchmark(
    model_factories: dict[str, Callable[[], nn.Module]],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    *,
    seeds: Sequence[int] = OFFICIAL_SEEDS,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    alpha_size: float = 1.0,
    alpha_polar: float = 1.0,
    alpha_rumor: float = 1.0,
    lambda_uncertainty: float = 0.0,
    use_class_weights: bool = True,
    classification_only: bool = False,
    reference_model: str = "SBiFAGAT",
    comparator_model: str = "auto",
    compute_bootstrap: bool = True,
    primary_hypothesis_keys: Sequence[str] | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Run all models over the official 5 random seeds and compute paired t-tests & bootstrap p-values."""
    raw_results: dict[str, list[dict[str, float]]] = {}
    summary_stats: dict[str, dict[str, tuple[float, float]]] = {}
    best_val_models: dict[str, nn.Module] = {}
    best_val_scores: dict[str, float] = {}
    best_seed_by_model: dict[str, int] = {}
    model_predictions: dict[str, list[dict[str, Any]]] = {}

    metric_keys = (
        "msle",
        "mae",
        "rmse_size",
        "r2_size",
        "spearman_size",
        "rmse",
        "mae_polar",
        "r2_polar",
        "pearson_r",
        "spearman_polar",
        "macro_f1",
        "accuracy",
        "auc_roc",
        "construct_corr_bc",
        "construct_corr_assort",
        "loss",
        "task_loss",
    )

    for model_name, factory in model_factories.items():
        seed_runs: list[dict[str, float]] = []
        pred_runs: list[dict[str, Any]] = []
        for seed in seeds:
            model, test_metrics = train_and_evaluate_single_seed(
                factory,
                train_loader,
                val_loader,
                test_loader,
                seed=seed,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                patience=patience,
                alpha_size=alpha_size,
                alpha_polar=alpha_polar,
                alpha_rumor=alpha_rumor,
                lambda_uncertainty=lambda_uncertainty,
                use_class_weights=use_class_weights,
                classification_only=classification_only,
                device=device,
                return_predictions=True,
            )
            preds_blob = test_metrics.pop("_predictions", None)
            if preds_blob is not None:
                pred_runs.append(preds_blob)
            seed_runs.append(test_metrics)

            val_score = float(test_metrics.get("val_task_loss", float("nan")))
            prev_best = best_val_scores.get(model_name, float("inf"))
            if model_name not in best_val_models or (
                math.isfinite(val_score) and val_score < prev_best
            ):
                best_val_models[model_name] = model
                if math.isfinite(val_score):
                    best_val_scores[model_name] = val_score
                best_seed_by_model[model_name] = int(seed)

        raw_results[model_name] = seed_runs
        model_predictions[model_name] = pred_runs
        agg: dict[str, tuple[float, float]] = {}
        for key in metric_keys:
            vals = np.asarray([r.get(key, float("nan")) for r in seed_runs], dtype=np.float64)
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                agg[key] = (float("nan"), float("nan"))
            else:
                mean_v = float(np.mean(vals))
                std_v = float(np.std(vals, ddof=1)) if vals.size >= 2 else 0.0
                agg[key] = (mean_v, std_v)
        summary_stats[model_name] = agg

    hypothesis_keys = (
        tuple(primary_hypothesis_keys)
        if primary_hypothesis_keys is not None
        else (
            ("accuracy", "macro_f1", "auc_roc")
            if classification_only
            else (
                "msle",
                "mae",
                "r2_size",
                "rmse",
                "pearson_r",
                "spearman_polar",
                "macro_f1",
                "accuracy",
                "auc_roc",
            )
        )
    )

    effective_comparator = (
        select_best_comparator(
            summary_stats,
            reference_model=reference_model,
            metric_keys=hypothesis_keys,
        )
        if comparator_model == "auto"
        else comparator_model
    )

    p_values: dict[str, float] = {}
    bootstrap_p_values: dict[str, float] = {}
    if reference_model in raw_results and effective_comparator in raw_results:
        for key in metric_keys:
            ref_scores = [r.get(key, float("nan")) for r in raw_results[reference_model]]
            cmp_scores = [r.get(key, float("nan")) for r in raw_results[effective_comparator]]
            p_values[key] = compute_paired_ttest(ref_scores, cmp_scores)
        if compute_bootstrap:
            bootstrap_p_values = _compute_sample_level_bootstrap_pvalues(
                model_predictions.get(reference_model, []),
                model_predictions.get(effective_comparator, []),
                metric_keys=metric_keys,
            )
        else:
            bootstrap_p_values = {key: float("nan") for key in metric_keys}

    adjusted_p_values: dict[str, float] = {}
    adjusted_bootstrap_p_values: dict[str, float] = {}
    if p_values:
        scoped_p = holm_bonferroni_adjust(
            {k: p_values[k] for k in hypothesis_keys if k in p_values}
        )
        adjusted_p_values = {k: scoped_p.get(k, float("nan")) for k in p_values}
    if bootstrap_p_values:
        scoped_boot = holm_bonferroni_adjust(
            {k: bootstrap_p_values[k] for k in hypothesis_keys if k in bootstrap_p_values}
        )
        adjusted_bootstrap_p_values = {
            k: scoped_boot.get(k, float("nan")) for k in bootstrap_p_values
        }

    return {
        "seeds": list(seeds),
        "raw_results": raw_results,
        "summary_stats": summary_stats,
        "p_values": p_values,
        "adjusted_p_values": adjusted_p_values,
        "bootstrap_p_values": bootstrap_p_values,
        "adjusted_bootstrap_p_values": adjusted_bootstrap_p_values,
        "reference_model": reference_model,
        "comparator_model": effective_comparator,
        "nonparametric_test_label": "Bootstrap",
        "best_seed_by_model": best_seed_by_model,
        "models": best_val_models,
    }
