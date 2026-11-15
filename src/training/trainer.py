"""Training loop, Early Stopping, multi-seed runner, and paired t-test utilities."""

from __future__ import annotations

import copy
from typing import Any, Callable, Literal, Sequence

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import ttest_rel
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
    weights[present] = (total / (num_present * counts[present])).astype(np.float32)
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
    class_weights: Tensor | None = None,
    classification_only: bool = False,
    return_predictions: bool = False,
) -> dict[str, Any]:
    """Evaluate a model over a PyG DataLoader and return loss + metrics."""
    model.eval()
    total_loss = 0.0
    total_graphs = 0

    size_true_list: list[np.ndarray] = []
    size_pred_list: list[np.ndarray] = []
    polar_true_list: list[np.ndarray] = []
    polar_pred_list: list[np.ndarray] = []
    rumor_true_list: list[np.ndarray] = []
    rumor_logits_list: list[np.ndarray] = []

    for batch in loader:
        batch = batch.to(device)
        outputs = model(batch)
        loss = compute_multitask_loss(
            outputs,
            batch,
            alpha_size=alpha_size,
            alpha_polar=alpha_polar,
            alpha_rumor=alpha_rumor,
            class_weights=class_weights,
            classification_only=classification_only,
        )
        num_graphs = int(batch.num_graphs)
        total_loss += float(loss.item()) * num_graphs
        total_graphs += num_graphs

        if hasattr(batch, "y_size") and batch.y_size is not None:
            size_true_list.append(batch.y_size.view(-1).detach().cpu().numpy())
            size_pred_list.append(outputs["y_size_pred"].view(-1).detach().cpu().numpy())
        if hasattr(batch, "y_polar") and batch.y_polar is not None:
            polar_true_list.append(batch.y_polar.view(-1).detach().cpu().numpy())
            polar_pred_list.append(outputs["y_polar_pred"].view(-1).detach().cpu().numpy())

        y_rumor = getattr(batch, "y_rumor", getattr(batch, "y", None))
        if y_rumor is not None:
            rumor_true_list.append(y_rumor.view(-1).detach().cpu().numpy())
            rumor_logits_list.append(outputs["rumor_logits"].detach().cpu().numpy())

    y_size_true = np.concatenate(size_true_list) if size_true_list else np.array([])
    y_size_pred = np.concatenate(size_pred_list) if size_pred_list else np.array([])
    y_polar_true = np.concatenate(polar_true_list) if polar_true_list else np.array([])
    y_polar_pred = np.concatenate(polar_pred_list) if polar_pred_list else np.array([])
    y_rumor_true = np.concatenate(rumor_true_list) if rumor_true_list else None
    rumor_logits = np.concatenate(rumor_logits_list, axis=0) if rumor_logits_list else None

    metrics: dict[str, Any] = evaluate_predictions(
        y_size_true=y_size_true,
        y_size_pred=y_size_pred,
        y_polar_true=y_polar_true,
        y_polar_pred=y_polar_pred,
        y_rumor_true=y_rumor_true,
        rumor_logits=rumor_logits,
    )
    metrics["loss"] = total_loss / max(1, total_graphs)
    if return_predictions:
        metrics["_predictions"] = {
            "y_size_true": y_size_true,
            "y_size_pred": y_size_pred,
            "y_polar_true": y_polar_true,
            "y_polar_pred": y_polar_pred,
            "y_rumor_true": y_rumor_true,
            "rumor_logits": rumor_logits,
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
    use_class_weights: bool = True,
    classification_only: bool = False,
    device: torch.device | str | None = None,
    return_predictions: bool = False,
) -> tuple[nn.Module, dict[str, Any]]:
    """Train one model for a single random seed with Early Stopping and return test metrics."""
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopper = EarlyStopping(patience=patience, mode="min")

    for _epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(target_device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = compute_multitask_loss(
                outputs,
                batch,
                alpha_size=alpha_size,
                alpha_polar=alpha_polar,
                alpha_rumor=alpha_rumor,
                class_weights=class_weights,
                classification_only=classification_only,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        val_metrics = evaluate_loader(
            model,
            val_loader,
            target_device,
            alpha_size=alpha_size,
            alpha_polar=alpha_polar,
            alpha_rumor=alpha_rumor,
            class_weights=class_weights,
            classification_only=classification_only,
        )
        if early_stopper.step(val_metrics["loss"], model):
            break

    early_stopper.restore_best(model)
    test_metrics = evaluate_loader(
        model,
        test_loader,
        target_device,
        alpha_size=alpha_size,
        alpha_polar=alpha_polar,
        alpha_rumor=alpha_rumor,
        class_weights=class_weights,
        classification_only=classification_only,
        return_predictions=return_predictions,
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
    """Two-sided paired bootstrap test over sample-level or fold-level paired observations."""
    arr_a = np.asarray(scores_a, dtype=np.float64)
    arr_b = np.asarray(scores_b, dtype=np.float64)
    valid = ~(np.isnan(arr_a) | np.isnan(arr_b))
    arr_a, arr_b = arr_a[valid], arr_b[valid]
    if arr_a.size < 2:
        return float("nan")
    diffs = arr_a - arr_b
    if np.allclose(diffs, 0.0):
        return 1.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_resamples, diffs.size))
    boot_means = diffs[idx].mean(axis=1)
    p_left = float(np.mean(boot_means <= 0.0))
    p_right = float(np.mean(boot_means >= 0.0))
    return float(min(1.0, 2.0 * min(p_left, p_right)))


def _compute_sample_level_bootstrap_pvalues(
    preds_ref: Sequence[dict[str, Any]],
    preds_cmp: Sequence[dict[str, Any]],
    *,
    metric_keys: Sequence[str],
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap across held-out test cascades (using seed-averaged predictions)."""
    if not preds_ref or not preds_cmp:
        return {}
    y_size_true = np.asarray(preds_ref[0]["y_size_true"], dtype=np.float64)
    y_polar_true = np.asarray(preds_ref[0]["y_polar_true"], dtype=np.float64)
    if y_size_true.size < 2:
        return {}

    ref_size = np.mean([p["y_size_pred"] for p in preds_ref], axis=0)
    cmp_size = np.mean([p["y_size_pred"] for p in preds_cmp], axis=0)
    ref_polar = np.mean([p["y_polar_pred"] for p in preds_ref], axis=0)
    cmp_polar = np.mean([p["y_polar_pred"] for p in preds_cmp], axis=0)

    out: dict[str, float] = {}
    out["msle"] = compute_paired_bootstrap_pvalue(
        (ref_size - y_size_true) ** 2, (cmp_size - y_size_true) ** 2, seed=seed
    )
    out["rmse_size"] = out["msle"]
    out["mae"] = compute_paired_bootstrap_pvalue(
        np.abs(ref_size - y_size_true), np.abs(cmp_size - y_size_true), seed=seed + 1
    )
    out["rmse"] = compute_paired_bootstrap_pvalue(
        (ref_polar - y_polar_true) ** 2, (cmp_polar - y_polar_true) ** 2, seed=seed + 2
    )
    out["mae_polar"] = compute_paired_bootstrap_pvalue(
        np.abs(ref_polar - y_polar_true), np.abs(cmp_polar - y_polar_true), seed=seed + 3
    )

    ref_logits_list = [p["rumor_logits"] for p in preds_ref if p.get("rumor_logits") is not None]
    cmp_logits_list = [p["rumor_logits"] for p in preds_cmp if p.get("rumor_logits") is not None]
    y_rumor_raw = preds_ref[0].get("y_rumor_true")
    if ref_logits_list and cmp_logits_list and y_rumor_raw is not None:
        y_rumor_arr = np.asarray(y_rumor_raw, dtype=np.int64).reshape(-1)
        valid = y_rumor_arr >= 0
        if np.sum(valid) >= 2:
            ref_probs = np.mean(
                [torch.softmax(torch.as_tensor(lg), dim=-1).numpy() for lg in ref_logits_list],
                axis=0,
            )[valid]
            cmp_probs = np.mean(
                [torch.softmax(torch.as_tensor(lg), dim=-1).numpy() for lg in cmp_logits_list],
                axis=0,
            )[valid]
            ref_correct = (np.argmax(ref_probs, axis=-1) == y_rumor_arr[valid]).astype(np.float64)
            cmp_correct = (np.argmax(cmp_probs, axis=-1) == y_rumor_arr[valid]).astype(np.float64)
            out["accuracy"] = compute_paired_bootstrap_pvalue(ref_correct, cmp_correct, seed=seed + 4)

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
    use_class_weights: bool = True,
    classification_only: bool = False,
    reference_model: str = "SBiFAGAT",
    comparator_model: str = "BiGCN",
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Run all models over the official 5 random seeds and compute paired t-tests & bootstrap p-values."""
    raw_results: dict[str, list[dict[str, float]]] = {}
    summary_stats: dict[str, dict[str, tuple[float, float]]] = {}
    trained_last_models: dict[str, nn.Module] = {}
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
        "loss",
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
                use_class_weights=use_class_weights,
                classification_only=classification_only,
                device=device,
                return_predictions=True,
            )
            preds_blob = test_metrics.pop("_predictions", None)
            if preds_blob is not None:
                pred_runs.append(preds_blob)
            seed_runs.append(test_metrics)
            trained_last_models[model_name] = model

        raw_results[model_name] = seed_runs
        model_predictions[model_name] = pred_runs
        agg: dict[str, tuple[float, float]] = {}
        for key in metric_keys:
            vals = np.asarray([r[key] for r in seed_runs], dtype=np.float64)
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                agg[key] = (float("nan"), float("nan"))
            else:
                mean_v = float(np.mean(vals))
                std_v = float(np.std(vals, ddof=1)) if vals.size >= 2 else 0.0
                agg[key] = (mean_v, std_v)
        summary_stats[model_name] = agg

    p_values: dict[str, float] = {}
    bootstrap_p_values: dict[str, float] = {}
    if reference_model in raw_results and comparator_model in raw_results:
        for key in metric_keys:
            ref_scores = [r[key] for r in raw_results[reference_model]]
            cmp_scores = [r[key] for r in raw_results[comparator_model]]
            p_values[key] = compute_paired_ttest(ref_scores, cmp_scores)
        bootstrap_p_values = _compute_sample_level_bootstrap_pvalues(
            model_predictions.get(reference_model, []),
            model_predictions.get(comparator_model, []),
            metric_keys=metric_keys,
        )

    return {
        "seeds": list(seeds),
        "raw_results": raw_results,
        "summary_stats": summary_stats,
        "p_values": p_values,
        "bootstrap_p_values": bootstrap_p_values,
        "reference_model": reference_model,
        "comparator_model": comparator_model,
        "models": trained_last_models,
    }
