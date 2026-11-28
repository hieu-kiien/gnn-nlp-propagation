"""Unit tests for `scripts.eval_hostility_router_audit`."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from scripts.eval_hostility_router_audit import evaluate_hostility_router


class _StubNLIScorer:
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None

    def _score_premise_hypothesis(self, premises, hypotheses, *, effective_max_len: int):
        # Always score 3rd-party hypothesis higher than direct-attack when phrase matches
        out = []
        for h in hypotheses:
            if "bên thứ ba" in h or "third party" in h:
                out.append([0.85, 0.10, 0.05])
            else:
                out.append([0.10, 0.15, 0.75])
        return torch.tensor(out, dtype=torch.float32)


def _write_pair_csvs(base_dir: Path, name: str, rows: list[dict]) -> tuple[Path, Path]:
    audit_p = base_dir / f"{name}.csv"
    key_p = base_dir / f"{name}_machine_key.csv"
    with audit_p.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "sample_id",
                "edge_id",
                "parent_text",
                "reply_text",
                "gold_adjudicated_stance",
                "gold_adjudicated_hostility",
                "sampling_weight",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "sample_id": r["sample_id"],
                    "edge_id": r["edge_id"],
                    "parent_text": r["parent_text"],
                    "reply_text": r["reply_text"],
                    "gold_adjudicated_stance": r["gold_stance"],
                    "gold_adjudicated_hostility": r["gold_hostility"],
                    "sampling_weight": r["sampling_weight"],
                }
            )
    with key_p.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["sample_id", "edge_id", "machine_stance_label", "s_uv", "h_uv", "sampling_weight"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "sample_id": r["sample_id"],
                    "edge_id": r["edge_id"],
                    "machine_stance_label": r["machine_label"],
                    "s_uv": r["s_uv"],
                    "h_uv": r["h_uv"],
                    "sampling_weight": r["sampling_weight"],
                }
            )
    return audit_p, key_p


def test_hostility_router_audit_zero_pii_and_flips(tmp_path: Path) -> None:
    pilot_rows = [
        {
            "sample_id": "p1",
            "edge_id": "e1",
            "parent_text": "Chủ đầu tư làm ăn tệ quá",
            "reply_text": "SECRET_RAW_COMMENT bọn nó lừa đảo thật",
            "gold_stance": "support",
            "gold_hostility": "uncivil",
            "sampling_weight": 1.0,
            "machine_label": "oppose",
            "s_uv": -0.55,
            "h_uv": 0.92,
        }
    ]
    test_rows = [
        {
            "sample_id": "t1",
            "edge_id": "e2",
            "parent_text": "Sếp công ty đó vô lý quá",
            "reply_text": "bọn nó quá đáng thật",
            "gold_stance": "support",
            "gold_hostility": "uncivil",
            "sampling_weight": 2.0,
            "machine_label": "oppose",
            "s_uv": -0.45,
            "h_uv": 0.88,
        },
        {
            "sample_id": "t2",
            "edge_id": "e3",
            "parent_text": "Bài viết rất hay",
            "reply_text": "Cảm ơn bạn đã chia sẻ",
            "gold_stance": "support",
            "gold_hostility": "clean",
            "sampling_weight": 5.0,
            "machine_label": "support",
            "s_uv": 0.65,
            "h_uv": 0.05,
        },
    ]
    p_csv, pk_csv = _write_pair_csvs(tmp_path, "pilot", pilot_rows)
    t_csv, tk_csv = _write_pair_csvs(tmp_path, "test", test_rows)
    out_json = tmp_path / "router.json"
    out_tex = tmp_path / "router.tex"

    res = evaluate_hostility_router(
        audit_120_csv=t_csv,
        key_120_csv=tk_csv,
        pilot_20_csv=p_csv,
        key_20_csv=pk_csv,
        nli_scorer=_StubNLIScorer(),
        output_json=out_json,
        output_tex=out_tex,
    )
    assert res["is_real_data"] is True
    assert res["vn_140_hostile_quadrants"]["before"]["support_uncivil"]["oppose_count"] == 2
    assert res["vn_140_hostile_quadrants"]["after"]["support_uncivil"]["correct_count"] == 2
    assert res["vn_120_test_overall"]["after"]["accuracy"] > res["vn_120_test_overall"]["before"]["accuracy"]
    raw_txt = out_json.read_text(encoding="utf-8")
    assert "SECRET_RAW_COMMENT" not in raw_txt
    assert out_tex.is_file()
    json.loads(raw_txt)


def test_hostility_router_preserves_reply_under_long_parent() -> None:
    from src.nlp.dual_edge_scorer import HostilityGatedTargetRouter

    captured_premises: list[str] = []

    class _CaptureStubScorer:
        model = None
        tokenizer = None

        def _score_premise_hypothesis(self, premises, hypotheses, *, effective_max_len: int):
            captured_premises.extend(str(p) for p in premises)
            return torch.tensor([[0.10, 0.80, 0.10] for _ in hypotheses], dtype=torch.float32)

    router = HostilityGatedTargetRouter(hostility_gate_threshold=0.5, lang="vi", nli_scorer=_CaptureStubScorer())
    long_parent = "Nội dung bài gốc dài " * 200  # > 3000 chars
    reply_marker = "ĐỒNG Ý VỚI CHỦ THỚT VỤ NÀY"
    routed, stats = router.route_pairs(
        [{"parent_text": long_parent, "reply_text": reply_marker, "s_uv": -0.3, "h_uv": 0.9}]
    )
    assert stats["gated_count"] == 1
    assert len(routed) == 1
    assert captured_premises
    assert all(reply_marker in prem for prem in captured_premises)
    assert all(len(prem) < 1300 for prem in captured_premises)
