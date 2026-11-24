"""Unit tests for SemEval-2019 Task 7 (RumourEval SDQC) download/extraction, parsing, and paper merge."""

from __future__ import annotations

import json
import tarfile
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from scripts.assemble_paper_artifacts import assemble_paper_artifacts
from scripts.eval_rumoureval_sdqc import (
    _safe_extract_tar,
    _safe_extract_zip,
    evaluate_nli_on_rumoureval,
    load_rumoureval_sdqc_pairs,
)


class _StubNLIScorer:
    def score_pairs(
        self,
        parent_texts,
        reply_texts,
        *,
        batch_size: int = 32,
        lang: str = "en",
        use_stance_verbalizer: bool = True,
    ):
        assert len(parent_texts) == len(reply_texts)
        scores = []
        labels = []
        for rep in reply_texts:
            low = str(rep).lower()
            if "confirm" in low or "agree" in low:
                scores.append(0.82)
                labels.append("support")
            elif "fake" in low or "hoax" in low:
                scores.append(-0.85)
                labels.append("oppose")
            else:
                scores.append(0.02)
                labels.append("neutral")
        return {
            "s_uv": np.asarray(scores, dtype=np.float32),
            "stance_probs": np.zeros((len(scores), 3), dtype=np.float32),
            "pred_label": labels,
        }


def test_rumoureval_safe_extract_rejects_traversal(tmp_path: Path) -> None:
    bad_zip = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("../../escape.txt", b"oops")
    with pytest.raises(ValueError, match="Unsafe archive member path"):
        _safe_extract_zip(bad_zip, tmp_path / "out_zip")

    bad_tar = tmp_path / "bad.tar.bz2"
    with tarfile.open(bad_tar, "w:bz2") as tf:
        info = tarfile.TarInfo(name="../escape.txt")
        data = b"oops"
        info.size = len(data)
        tf.addfile(info, BytesIO(data))
    with pytest.raises(ValueError, match="Unsafe archive member path"):
        _safe_extract_tar(bad_tar, tmp_path / "out_tar")


def test_rumoureval_tree_parsing_and_end_to_end_assembly(tmp_path: Path) -> None:
    ext = tmp_path / "extracted" / "rumoureval2019"
    tw_thread = ext / "rumoureval-2019-test-data" / "twitter-en-test-data" / "topic1" / "r1"
    (tw_thread / "source-tweet").mkdir(parents=True)
    (tw_thread / "replies").mkdir(parents=True)
    (tw_thread / "structure.json").write_text(
        json.dumps({"r1": {"tw_c1": {"tw_c2": []}, "tw_c3": []}}),
        encoding="utf-8",
    )
    (tw_thread / "source-tweet" / "r1.json").write_text(
        json.dumps({"id_str": "r1", "text": "Breaking claim on campus"}),
        encoding="utf-8",
    )
    (tw_thread / "replies" / "tw_c1.json").write_text(
        json.dumps({"id_str": "tw_c1", "text": "I agree and confirm this report."}),
        encoding="utf-8",
    )
    (tw_thread / "replies" / "tw_c2.json").write_text(
        json.dumps({"id_str": "tw_c2", "full_text": "No, that is totally fake hoax."}),
        encoding="utf-8",
    )
    (tw_thread / "replies" / "tw_c3.json").write_text(
        json.dumps({"id_str": "tw_c3", "text": "Do we have any source link yet?"}),
        encoding="utf-8",
    )

    rd_thread = ext / "rumoureval-2019-test-data" / "reddit-test-data" / "rd1"
    (rd_thread / "source-tweet").mkdir(parents=True)
    (rd_thread / "replies").mkdir(parents=True)
    (rd_thread / "structure.json").write_text(
        json.dumps({"rd1": {"rd_c1": []}}),
        encoding="utf-8",
    )
    (rd_thread / "source-tweet" / "rd1.json").write_text(
        json.dumps(
            {
                "kind": "Listing",
                "data": {"children": [{"kind": "t3", "data": {"title": "Reddit rumour post", "selftext": "Details"}}]},
            }
        ),
        encoding="utf-8",
    )
    (rd_thread / "replies" / "rd_c1.json").write_text(
        json.dumps({"kind": "t1", "data": {"id": "rd_c1", "body": "Just a general comment here."}}),
        encoding="utf-8",
    )

    # Include a source tweet entry in subtaskaenglish to verify only reply edges are scored
    (ext / "final-eval-key.json").write_text(
        json.dumps(
            {
                "subtaskaenglish": {
                    "r1": "support",
                    "tw_c1": "support",
                    "tw_c2": "deny",
                    "tw_c3": "query",
                    "rd_c1": "comment",
                }
            }
        ),
        encoding="utf-8",
    )

    pairs = load_rumoureval_sdqc_pairs(tmp_path / "extracted", split="test")
    by_id = {p["id"]: p for p in pairs}
    assert set(by_id.keys()) == {"tw_c1", "tw_c2", "tw_c3", "rd_c1"}
    assert by_id["tw_c2"]["parent_id"] == "tw_c1"
    assert "confirm" in by_id["tw_c2"]["parent_text"]
    assert by_id["tw_c2"]["platform"] == "twitter"
    assert by_id["rd_c1"]["platform"] == "reddit"
    assert "Reddit rumour post" in by_id["rd_c1"]["parent_text"]

    paper_dir = tmp_path / "paper"
    out_json = paper_dir / "results_rumoureval_sdqc.json"
    pred_json = tmp_path / "preds.json"
    res = evaluate_nli_on_rumoureval(
        pairs,
        scorer=_StubNLIScorer(),
        model_name="stub-nli",
        output_json=out_json,
        predictions_json=pred_json,
    )
    assert out_json.is_file()
    assert pred_json.is_file()
    assert res["rumoureval_sdqc"]["num_samples"] == 4
    assert res["rumoureval_sdqc"]["accuracy"] == pytest.approx(1.0)
    assert res["meta"]["platform_counts"] == {"twitter": 3, "reddit": 1}

    manifest = assemble_paper_artifacts(paper_dir=paper_dir)
    assert manifest["latex_underscore_issues"] == {}
    audit_tex = (paper_dir / "tables_human_audit.tex").read_text(encoding="utf-8")
    assert "RumourEval SDQC & 4 &" in audit_tex
