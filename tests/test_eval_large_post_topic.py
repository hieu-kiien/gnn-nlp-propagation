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


def test_length_sorted_mdeberta_nli_preserves_alignment(monkeypatch) -> None:
    import sys
    import types
    import torch
    from scripts.eval_large_post_topic import _score_batch_mdeberta_nli

    class _FakeTok:
        def __call__(self, premises, hyps, **kwargs):
            # Different contexts prefer different candidates so row swaps immediately fail assertions.
            scores = []
            for p, h in zip(premises, hyps):
                s = 2.0 if (("sports" in p.lower() and "Sports" in h) or ("finance" in p.lower() and "Finance" in h) or ("tech" in p.lower() and "Tech" in h)) else -2.0
                scores.append(s)
            return {"input_ids": torch.tensor(scores, dtype=torch.float32).unsqueeze(1)}

    class _FakeCfg:
        label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}

    class _FakeModel:
        config = _FakeCfg()

        def to(self, dev):
            return self

        def eval(self):
            return self

        def __call__(self, **enc):
            sig = enc["input_ids"][:, 0]
            logits = torch.stack([sig, torch.zeros_like(sig), -sig], dim=-1)
            return types.SimpleNamespace(logits=logits)

    fake_tf = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: _FakeTok()),
        AutoModelForSequenceClassification=types.SimpleNamespace(from_pretrained=lambda *a, **k: _FakeModel()),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)

    contexts = [
        "short sports clip",
        "much longer finance market macroeconomics discussion thread with many tokens",
        "medium length tech software engineering post",
    ]
    cands = [
        {"id": "sp", "label": "Sports"},
        {"id": "fn", "label": "Finance"},
        {"id": "tc", "label": "Tech"},
    ]
    probs, _ = _score_batch_mdeberta_nli(
        contexts, cands, lang="en", device=torch.device("cpu"), batch_size=2, max_length=160, allow_download=False
    )
    assert probs.shape == (3, 3)
    assert list(np.argmax(probs, axis=1)) == [0, 1, 2]
