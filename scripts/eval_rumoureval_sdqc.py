"""Download and evaluate NLI Stance Teacher on official SemEval-2019 Task 7 (RumourEval SDQC).

Reuses `src.nlp.dual_edge_scorer.NLIStanceTeacherScorer` (Symmetric Stance Verbalizer)
and `scripts.compute_kappa.evaluate_rumoureval_sdqc` to measure zero-shot English stance
alignment (`Support / Deny / Query / Comment` -> `support / oppose / neutral`), emitting
`paper/results_rumoureval_sdqc.json` compatible with `scripts/assemble_paper_artifacts.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import requests

from scripts.compute_kappa import evaluate_rumoureval_sdqc
from src.nlp.dual_edge_scorer import MDEBERTA_NLI_MODEL_NAME, NLIStanceTeacherScorer

RUMOUREVAL_FIGSHARE_URL = "https://ndownloader.figshare.com/files/16188500"
RUMOUREVAL_ARCHIVE_NAME = "rumoureval2019.tar.bz2"
RUMOUREVAL_EXPECTED_SHA256 = "2b9bde22c38d6fe6b2f4fe52de7544444e0c6436201f79600854a38522dfd76c"

_SPLIT_KEY_FILENAMES: dict[str, tuple[str, ...]] = {
    "test": ("final-eval-key.json",),
    "dev": ("dev-key.json",),
    "train": ("train-key.json",),
    "all": ("final-eval-key.json", "dev-key.json", "train-key.json"),
}


def _validate_archive_member_path(member_name: str, dest_dir: Path) -> Path:
    """Reject path traversal (`..`), Windows drive paths, or symlink escape."""
    normalized = member_name.replace("\\", "/")
    posix = PurePosixPath(normalized)
    if posix.is_absolute() or ".." in posix.parts:
        raise ValueError(f"Unsafe archive member path rejected: {member_name!r}")
    if normalized.startswith("/") or (len(normalized) >= 2 and normalized[1] == ":"):
        raise ValueError(f"Absolute archive member path rejected: {member_name!r}")
    dest_root = Path(dest_dir).resolve()
    target = (dest_root / Path(*posix.parts)).resolve()
    if target != dest_root and dest_root not in target.parents:
        raise ValueError(f"Archive member escapes target directory: {member_name!r}")
    return target


def _safe_extract_tar(archive_path: Path, dest_dir: Path) -> None:
    dest_root = Path(dest_dir).resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:bz2") as tf:
        members = tf.getmembers()
        for member in members:
            _validate_archive_member_path(member.name, dest_root)
            if member.issym() or member.islnk():
                raise ValueError(f"Unsafe link archive member rejected: {member.name!r}")
        for member in members:
            target = _validate_archive_member_path(member.name, dest_root)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                src_fh = tf.extractfile(member)
                if src_fh is None:
                    continue
                with src_fh, target.open("wb") as out_fh:
                    out_fh.write(src_fh.read())


def _safe_extract_zip(zip_path: Path, dest_dir: Path) -> None:
    dest_root = Path(dest_dir).resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            target = _validate_archive_member_path(info.filename, dest_root)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as src_fh, target.open("wb") as out_fh:
                    out_fh.write(src_fh.read())


def _download_archive(url: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest_path.with_suffix(dest_path.suffix + ".part")
    part_path.unlink(missing_ok=True)

    # Figshare AWS WAF rejects spoofed Chrome User-Agents paired with non-browser TLS fingerprints
    # (HTTP 202 WAF challenge); use native curl User-Agent first.
    try:
        res = subprocess.run(
            [
                "curl",
                "-L",
                "--fail",
                "-s",
                "-S",
                "-o",
                str(part_path),
                url,
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if res.returncode == 0 and part_path.is_file() and part_path.stat().st_size > 1_000_000:
            part_path.replace(dest_path)
            return
    except (OSError, subprocess.SubprocessError):
        pass

    part_path.unlink(missing_ok=True)
    with requests.Session() as sess:
        sess.headers.update({"User-Agent": "curl/8.7.1", "Accept": "*/*"})
        with sess.get(url, stream=True, timeout=180) as resp:
            resp.raise_for_status()
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Unexpected HTTP {resp.status_code} (likely WAF challenge) downloading {url}"
                )
            with part_path.open("wb") as out_fh:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        out_fh.write(chunk)
    if not part_path.is_file() or part_path.stat().st_size <= 1_000_000:
        part_path.unlink(missing_ok=True)
        raise RuntimeError(f"Incomplete or empty archive downloaded from {url}")
    part_path.replace(dest_path)


def _verify_archive_sha256(archive_path: Path, expected_sha256: str) -> str:
    h = hashlib.sha256()
    with Path(archive_path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    actual = h.hexdigest().lower()
    if actual != expected_sha256.strip().lower():
        raise ValueError(
            f"SHA-256 mismatch for '{archive_path}': expected {expected_sha256}, got {actual}"
        )
    return actual


def download_and_extract_rumoureval(
    data_dir: Path = Path("data/raw/rumoureval2019"),
    *,
    url: str = RUMOUREVAL_FIGSHARE_URL,
    split: str = "all",
    verify_checksum: bool = True,
    expected_sha256: str | None = RUMOUREVAL_EXPECTED_SHA256,
) -> Path:
    """Ensure `rumoureval2019.tar.bz2` and requested inner split zips are safely extracted."""
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / RUMOUREVAL_ARCHIVE_NAME
    extracted_dir = root / "extracted"

    if not archive_path.is_file() or archive_path.stat().st_size < 1_000_000:
        _download_archive(url, archive_path)
    if verify_checksum and expected_sha256:
        try:
            _verify_archive_sha256(archive_path, expected_sha256)
        except ValueError:
            archive_path.unlink(missing_ok=True)
            _download_archive(url, archive_path)
            _verify_archive_sha256(archive_path, expected_sha256)

    has_keys = any(extracted_dir.rglob("final-eval-key.json")) if extracted_dir.exists() else False
    if not has_keys:
        _safe_extract_tar(archive_path, extracted_dir)

    split_norm = str(split).strip().lower()
    for zpath in sorted(extracted_dir.rglob("*.zip")):
        zname = zpath.name.lower()
        if split_norm == "test" and "test" not in zname:
            continue
        if split_norm in {"train", "dev"} and "train" not in zname:
            continue
        target_marker = zpath.with_suffix("")
        if not target_marker.exists():
            _safe_extract_zip(zpath, zpath.parent)
    return extracted_dir


def _extract_post_text(payload: Any) -> str:
    """Extract clean tweet/Reddit post or reply text from RumourEval JSON payloads."""
    if isinstance(payload, list):
        for item in payload:
            txt = _extract_post_text(item)
            if txt:
                return txt
        return ""
    if not isinstance(payload, Mapping):
        return ""

    # Reddit Listing wrapper (used in `source-tweet/*.json`)
    if payload.get("kind") == "Listing":
        children = (payload.get("data") or {}).get("children") or []
        return _extract_post_text(children)

    # Reddit object wrapper (`kind: t1|t3`) or raw dict
    if "data" in payload and isinstance(payload["data"], Mapping):
        inner = _extract_post_text(payload["data"])
        if inner:
            return inner

    if isinstance(payload.get("full_text"), str) and payload["full_text"].strip():
        return payload["full_text"].strip()
    if isinstance(payload.get("text"), str) and payload["text"].strip():
        return payload["text"].strip()
    if isinstance(payload.get("body"), str) and payload["body"].strip():
        return payload["body"].strip()

    title = str(payload.get("title") or "").strip()
    selftext = str(payload.get("selftext") or "").strip()
    if title and selftext and selftext not in {"[deleted]", "[removed]"}:
        return f"{title}\n{selftext}".strip()
    return title or (selftext if selftext not in {"[deleted]", "[removed]"} else "")


def _collect_parent_map(struct_node: Any, parent_id: str | None, out: dict[str, str]) -> None:
    if isinstance(struct_node, Mapping):
        for k, children in struct_node.items():
            cid = str(k).strip()
            if parent_id is not None and cid:
                out[cid] = parent_id
            _collect_parent_map(children, cid or parent_id, out)
    elif isinstance(struct_node, Sequence) and not isinstance(struct_node, (str, bytes)):
        for item in struct_node:
            _collect_parent_map(item, parent_id, out)


def load_rumoureval_sdqc_pairs(
    extracted_root: Path,
    *,
    split: str = "test",
) -> list[dict[str, Any]]:
    """Parse RumourEval Twitter & Reddit trees with gold Subtask A (`subtaskaenglish`) labels."""
    root = Path(extracted_root)
    if not root.exists():
        raise FileNotFoundError(f"RumourEval directory not found: {root}")
    key_names = _SPLIT_KEY_FILENAMES.get(split)
    if key_names is None:
        raise ValueError(f"Unsupported split {split!r}; expected one of {sorted(_SPLIT_KEY_FILENAMES)}")

    # Auto-extract nested zip archives if called right on `extracted/` before extraction.
    for zpath in sorted(root.rglob("*.zip")):
        if not zpath.with_suffix("").exists():
            _safe_extract_zip(zpath, zpath.parent)

    gold_labels: dict[str, tuple[str, str]] = {}
    for kname in key_names:
        for kfile in sorted(root.rglob(kname)):
            blob = json.loads(kfile.read_text(encoding="utf-8"))
            sub = blob.get("subtaskaenglish") if isinstance(blob, Mapping) else None
            if not isinstance(sub, Mapping):
                continue
            for item_id, lbl in sub.items():
                gold_labels[str(item_id).strip()] = (str(lbl).strip().lower(), kname.replace("-key.json", ""))

    if not gold_labels:
        raise ValueError(f"No subtaskaenglish labels found under {root} for split={split!r}")

    pairs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    # A thread directory is any directory containing `source-tweet` or `replies`
    thread_dirs: set[Path] = set()
    for tag_dir in ("source-tweet", "replies"):
        for found in root.rglob(tag_dir):
            if found.is_dir():
                thread_dirs.add(found.parent)

    for tdir in sorted(thread_dirs):
        src_dir = tdir / "source-tweet"
        rep_dir = tdir / "replies"
        node_texts: dict[str, str] = {}
        root_id = tdir.name
        root_text = ""

        if src_dir.is_dir():
            for jf in sorted(src_dir.glob("*.json")):
                try:
                    txt = _extract_post_text(json.loads(jf.read_text(encoding="utf-8", errors="replace")))
                except (OSError, ValueError):
                    txt = ""
                node_texts[jf.stem] = txt
                if txt and not root_text:
                    root_id = jf.stem
                    root_text = txt

        if not rep_dir.is_dir():
            continue

        for jf in sorted(rep_dir.glob("*.json")):
            try:
                txt = _extract_post_text(json.loads(jf.read_text(encoding="utf-8", errors="replace")))
            except (OSError, ValueError):
                txt = ""
            if txt:
                node_texts[jf.stem] = txt

        parent_of: dict[str, str] = {}
        struct_path = tdir / "structure.json"
        if struct_path.is_file():
            try:
                struct_obj = json.loads(struct_path.read_text(encoding="utf-8", errors="replace"))
                _collect_parent_map(struct_obj, None, parent_of)
            except (OSError, ValueError):
                parent_of = {}

        path_low = tdir.as_posix().lower()
        platform = "reddit" if "reddit" in path_low else "twitter"

        for jf in sorted(rep_dir.glob("*.json")):
            item_id = jf.stem
            if item_id in seen_ids or item_id not in gold_labels:
                continue
            reply_text = node_texts.get(item_id, "")
            if not reply_text.strip():
                continue
            parent_id = parent_of.get(item_id, root_id)
            parent_text = node_texts.get(parent_id, "").strip() or root_text.strip()
            if not parent_text:
                continue
            gold_lbl, key_src = gold_labels[item_id]
            seen_ids.add(item_id)
            pairs.append(
                {
                    "id": item_id,
                    "edge_id": item_id,
                    "thread_id": root_id,
                    "parent_id": parent_id,
                    "root_text": root_text,
                    "parent_text": parent_text,
                    "reply_text": reply_text,
                    "sdqc_label": gold_lbl,
                    "platform": platform,
                    "split": key_src,
                }
            )

    return pairs


def evaluate_nli_on_rumoureval(
    pairs: Sequence[Mapping[str, Any]],
    *,
    scorer: Any | None = None,
    model_name: str = MDEBERTA_NLI_MODEL_NAME,
    batch_size: int = 32,
    output_json: Path = Path("paper/results_rumoureval_sdqc.json"),
    predictions_json: Path | None = None,
    context_mode: str = "root",
) -> dict[str, Any]:
    """Score `(parent_text, reply_text)` pairs with NLI verbalizer and evaluate SDQC metrics."""
    if not pairs:
        raise ValueError("Cannot evaluate RumourEval SDQC on an empty pair list")
    if context_mode not in {"parent", "root"}:
        raise ValueError(f"Unsupported context_mode={context_mode!r}; expected 'parent' or 'root'")

    active_scorer = (
        scorer
        if scorer is not None
        else NLIStanceTeacherScorer(
            model_name_or_path=model_name,
            lang="en",
            use_stance_verbalizer=True,
        )
    )

    u_texts = [
        str(p.get("root_text") if context_mode == "root" and p.get("root_text") else p.get("parent_text") or "")
        for p in pairs
    ]
    v_texts = [str(p.get("reply_text") or "") for p in pairs]
    out = active_scorer.score_pairs(
        u_texts,
        v_texts,
        batch_size=int(batch_size),
        lang="en",
        use_stance_verbalizer=True,
    )
    s_arr = out["s_uv"]
    labels_arr = out.get("pred_label") or []

    scored_items: list[dict[str, Any]] = []
    for idx, p in enumerate(pairs):
        sv = float(s_arr[idx])
        plbl = str(labels_arr[idx]) if idx < len(labels_arr) else (
            "support" if sv >= 0.1 else ("oppose" if sv <= -0.1 else "neutral")
        )
        scored_items.append(
            {
                "id": str(p.get("id") or p.get("edge_id") or idx),
                "edge_id": str(p.get("edge_id") or p.get("id") or idx),
                "thread_id": str(p.get("thread_id") or ""),
                "platform": str(p.get("platform") or ""),
                "sdqc_label": str(p["sdqc_label"]),
                "s_uv": sv,
                "pred_label": plbl,
            }
        )

    if predictions_json is not None:
        pred_path = Path(predictions_json)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_path.write_text(json.dumps(scored_items, indent=2, ensure_ascii=False), encoding="utf-8")
        metrics = evaluate_rumoureval_sdqc(pred_path)
    else:
        with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as tfh:
            tmp_path = Path(tfh.name)
            json.dump(scored_items, tfh, ensure_ascii=False)
        try:
            metrics = evaluate_rumoureval_sdqc(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    payload = {
        "rumoureval_sdqc": metrics,
        "meta": {
            "model_name": str(model_name),
            "context_mode": context_mode,
            "num_scored_replies": len(scored_items),
            "platform_counts": {
                "twitter": sum(1 for x in scored_items if x["platform"] == "twitter"),
                "reddit": sum(1 for x in scored_items if x["platform"] == "reddit"),
            },
        },
    }
    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Evaluate NLI Stance Teacher on RumourEval-2019 SDQC.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/rumoureval2019"))
    parser.add_argument("--split", choices=sorted(_SPLIT_KEY_FILENAMES.keys()), default="test")
    parser.add_argument("--model-name", type=str, default=MDEBERTA_NLI_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--context-mode", choices=("root", "parent"), default="root")
    parser.add_argument("--no-verify-checksum", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path("paper/results_rumoureval_sdqc.json"))
    parser.add_argument("--predictions-json", type=Path, default=None)
    args = parser.parse_args()

    extracted = download_and_extract_rumoureval(
        args.data_dir,
        split=args.split,
        verify_checksum=not bool(args.no_verify_checksum),
    )
    pairs = load_rumoureval_sdqc_pairs(extracted, split=args.split)
    if args.max_samples is not None:
        pairs = pairs[: max(1, int(args.max_samples))]
    result = evaluate_nli_on_rumoureval(
        pairs,
        model_name=args.model_name,
        batch_size=args.batch_size,
        output_json=args.output_json,
        predictions_json=args.predictions_json,
        context_mode=args.context_mode,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
