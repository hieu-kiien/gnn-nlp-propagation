"""Unit tests for `scripts.eval_large_post_topic` pure aggregation and metric helpers."""

from __future__ import annotations

import numpy as np

from scripts.eval_large_post_topic import (
    PHEME_EVENT_TOPICS,
    _aggregate_scale_trajectories,
    _binary_early_warning_metrics,
    _classification_metrics,
)


def test_pheme_event_topics_have_no_slug_leakage() -> None:
    assert len(PHEME_EVENT_TOPICS) == 9
    for item in PHEME_EVENT_TOPICS:
        tid = item["id"].lower()
        combined = f"{item['label']} {item['description']}".lower()
        assert tid not in combined
        assert "germanwings" not in combined
        assert "charlie" not in combined


def test_classification_and_binary_metrics_and_trajectory_aggregation() -> None:
    cand_ids = ["a", "b", "c"]
    probs = np.array(
        [
            [0.8, 0.15, 0.05],
            [0.2, 0.7, 0.1],
            [0.1, 0.35, 0.55],
            [0.45, 0.5, 0.05],
        ],
        dtype=np.float64,
    )
    m = _classification_metrics(["a", "b", "c", "a"], probs, cand_ids)
    assert m["accuracy"] == 0.75
    assert m["top2_accuracy"] == 1.0
    assert 0.0 <= m["macro_f1"] <= 1.0

    y_bin = np.array([0, 0, 1, 1], dtype=np.int64)
    scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    bm = _binary_early_warning_metrics(y_bin, scores)
    assert bm["roc_auc"] == 1.0
    assert bm["pr_auc"] == 1.0
    assert bm["best_f1"] == 1.0

    per_cascade = {
        f"c{i}": {
            "num_replies": 60 if i < 4 else 30,
            "scale_tier": "large" if i < 4 else "medium",
            "evolution_trend": "escalating" if i < 3 else "stable",
            "full_y_polar": 0.14 if i < 4 else 0.08,
            "polar_delta": 0.04 if i < 3 else 0.0,
            "trajectory": [
                {"k": 5, "y_polar": 0.05},
                {"k": 20, "y_polar": 0.10},
                {"k": 50, "y_polar": 0.14} if i < 4 else {"k": 30, "y_polar": 0.08},
            ],
        }
        for i in range(8)
    }
    agg = _aggregate_scale_trajectories(
        per_cascade,
        scale_tiers=("medium", "large_ge_50"),
        milestones=(5, 20, 50),
    )
    assert agg["medium"]["count"] == 4
    assert agg["large_ge_50"]["count"] == 4
    assert agg["large_ge_50"]["escalating_rate"] == 0.75
    assert any(pt["k"] == 50 and pt["n"] == 4 for pt in agg["large_ge_50"]["milestones"])
