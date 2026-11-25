"""Unit tests verifying scientific safeguards of the live social thread predictor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from scripts.predict_live_thread import (
    VN_POLAR_Q50,
    VN_POLAR_Q75,
    VN_POLAR_Q90,
    _classify_calibrated_regime,
    _format_report_vi,
    _make_demo_offline_thread,
    predict_live_cascade,
)


def test_predict_live_blocks_untrained_on_real_thread() -> None:
    """Real cascades without a trained checkpoint must raise RuntimeError unless explicitly permitted."""
    post, comments = _make_demo_offline_thread()
    post = dict(post, id="real_live_999")
    comments = [
        dict(
            c,
            link_id="t3_real_live_999",
            parent_id="t3_real_live_999" if c["parent_id"] == "t3_demo_vn_live01" else c["parent_id"],
        )
        for c in comments
    ]
    with pytest.raises(RuntimeError, match="Không tìm thấy checkpoint mô hình GNN"):
        predict_live_cascade(post, comments, is_demo=False, allow_untrained=False)


def test_predict_live_demo_offline_allowed_without_checkpoint() -> None:
    """The self-contained offline demo cascade is allowed to run for structural verification."""
    post, comments = _make_demo_offline_thread()
    res = predict_live_cascade(post, comments, is_demo=True, allow_untrained=False)
    assert res["checkpoint_loaded"] is False
    assert "early_burst_metrics" in res
    assert "top_trigger_edges" in res


def test_predict_live_allow_untrained_flag_bypasses_block() -> None:
    """Passing allow_untrained=True permits structural checking while marking checkpoint_loaded=False."""
    post, comments = _make_demo_offline_thread()
    res = predict_live_cascade(post, comments, is_demo=False, allow_untrained=True)
    assert res["checkpoint_loaded"] is False


def test_predict_live_corrupt_checkpoint_raises_value_error(tmp_path: Path) -> None:
    """A corrupt or non-state-dict checkpoint must raise ValueError with a clear diagnostic message."""
    bad_ckpt = tmp_path / "corrupt_gnn.pt"
    torch.save({"corrupt_key": 123}, bad_ckpt)
    post, comments = _make_demo_offline_thread()
    with pytest.raises(ValueError, match="thiếu trường 'model_state_dict'"):
        predict_live_cascade(post, comments, gnn_ckpt=bad_ckpt, is_demo=False, allow_untrained=False)


def test_predict_live_empty_comments_rejected() -> None:
    """An empty or invalid thread with 0 valid comments must raise ValueError."""
    post, _ = _make_demo_offline_thread()
    with pytest.raises(ValueError, match="không có đủ bình luận hợp lệ|insufficient valid comments"):
        predict_live_cascade(post, [], is_demo=True, allow_untrained=True)


def test_predict_live_honest_confidence_none_and_clean_2d_report() -> None:
    """When predict_uncertainty=False, confidence_score is None, report prints N/A, and 4D legacy strings are absent."""
    post, comments = _make_demo_offline_thread()
    res = predict_live_cascade(post, comments, is_demo=True, allow_untrained=False)
    assert res["confidence_score"] is None
    assert res["uncertainty_available"] is False
    assert isinstance(res["early_y_polar"], float)
    assert 0.0 <= res["early_y_polar"] <= 1.0
    assert "irrational_fallacy" not in res["behavior_counts"]
    assert set(res["behavior_counts"].keys()) == {
        "support",
        "neutral",
        "constructive_oppose",
        "hostile_conflict",
    }
    assert "flagged_hostile" in res
    assert "flagged_irrational_or_hostile" not in res
    assert res["actual_post_hoc_comparison"]["has_future_replies"] is True
    assert res["actual_post_hoc_comparison"]["actual_full_polarization_mode"] == "future_only"
    assert res["actual_post_hoc_comparison"]["actual_full_polarization"] is not None

    report_txt = _format_report_vi(res)
    assert "N/A (uncertainty disabled)" in report_txt
    assert "5 NHOM HANH VI" not in report_txt
    assert "vo_ly=" not in report_txt


def test_predict_live_no_future_replies_sets_actual_polar_none() -> None:
    """When total replies <= K, actual_full_polarization is None with mode 'no_future_replies'."""
    post, comments = _make_demo_offline_thread()
    res = predict_live_cascade(post, comments[:5], k_horizon=20, is_demo=True, allow_untrained=False)
    assert res["k_observed"] == 5
    act = res["actual_post_hoc_comparison"]
    assert act["has_future_replies"] is False
    assert act["actual_full_polarization"] is None
    assert act["actual_full_polarization_mode"] == "no_future_replies"
    report_txt = _format_report_vi(res)
    assert "N/A (chua co binh luan t > K)" in report_txt


def test_calibrated_regime_never_overridden_by_counts() -> None:
    """Strict quantile regime classification ignores heavy conflict/oppose counts."""
    counts = {"support": 0, "neutral": 0, "constructive_oppose": 10, "hostile_conflict": 10}
    assert "< Q50" in _classify_calibrated_regime(VN_POLAR_Q50 - 0.01, counts)
    assert "Q50-Q75" in _classify_calibrated_regime((VN_POLAR_Q50 + VN_POLAR_Q75) * 0.5, counts)
    assert "Q75-Q90" in _classify_calibrated_regime((VN_POLAR_Q75 + VN_POLAR_Q90) * 0.5, counts)
    assert ">= Q90" in _classify_calibrated_regime(VN_POLAR_Q90 + 0.01, counts)
