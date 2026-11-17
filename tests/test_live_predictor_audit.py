"""Unit tests verifying scientific safeguards of the live social thread predictor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from scripts.predict_live_thread import (
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
