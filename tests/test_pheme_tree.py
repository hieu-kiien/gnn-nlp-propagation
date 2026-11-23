"""Unit tests for PHEME Figshare integrity verification and directed tree reconstruction."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from collections import deque
from pathlib import Path

import pytest

from src.data.pheme_loader import (
    _discover_local_pheme_source,
    _find_event_directories,
    compute_hashes,
    convert_veracity_label,
    extract_pheme_archive,
    flatten_structure_parent_map,
    process_pheme_dataset,
    verify_archive_integrity,
)
from src.data.schema import load_cascade_parquet, validate_cascade_contract


def _write_tweet_json(
    path: Path,
    *,
    tid: str,
    text: str,
    created_at: str,
    user_id: str,
    reply_to: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "id_str": tid,
        "id": int(tid),
        "text": text,
        "created_at": created_at,
        "user": {"id_str": user_id, "id": int(user_id)},
        "in_reply_to_status_id_str": reply_to,
        "in_reply_to_status_id": int(reply_to) if reply_to is not None else None,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_mock_pheme_corpus(root_dir: Path) -> Path:
    base = root_dir / "all-rnr-annotated-threads"

    # Cascade 1: charliehebdo rumour (true), leaf [], missing reply field recovered via structure.json,
    # 1-second clock skew on child, orphan deleted parent subtree, and macOS AppleDouble junk.
    t1 = base / "charliehebdo-all-rnr-threads" / "rumours" / "552783238415265792"
    _write_tweet_json(
        t1 / "source-tweets" / "552783238415265792.json",
        tid="552783238415265792",
        text="Breaking news: shooting reported at Charlie Hebdo HQ in Paris.",
        created_at="Wed Jan 07 11:11:33 +0000 2015",
        user_id="9001",
    )
    _write_tweet_json(
        t1 / "reactions" / "552783667052167168.json",
        tid="552783667052167168",
        text="Confirming sirens near the building right now.",
        created_at="Wed Jan 07 11:12:30 +0000 2015",
        user_id="9002",
        reply_to="552783238415265792",
    )
    # Missing in_reply_to field -> recovered via structure.json
    _write_tweet_json(
        t1 / "reactions" / "552784168849190912.json",
        tid="552784168849190912",
        text="Hold on, do we have an official police statement yet?",
        created_at="Wed Jan 07 11:12:29 +0000 2015",  # 1s clock skew before parent -> clamped
        user_id="9003",
        reply_to=None,
    )
    # Deep level-3 child
    _write_tweet_json(
        t1 / "reactions" / "552785000000000003.json",
        tid="552785000000000003",
        text="Yes, French interior ministry spokesperson just briefed reporters.",
        created_at="Wed Jan 07 11:15:10 +0000 2015",
        user_id="9004",
        reply_to="552784168849190912",
    )
    # Orphan whose parent tweet 9999999999 is absent on disk -> must be pruned cleanly
    _write_tweet_json(
        t1 / "reactions" / "552785999999999999.json",
        tid="552785999999999999",
        text="Reply to a deleted intermediate tweet.",
        created_at="Wed Jan 07 11:16:00 +0000 2015",
        user_id="9005",
        reply_to="9999999999",
    )
    (t1 / "annotation.json").write_text(
        json.dumps({"is_rumour": "rumour", "misinformation": "0", "true": "1"}),
        encoding="utf-8",
    )
    (t1 / "structure.json").write_text(
        json.dumps(
            {
                "552783238415265792": {
                    "552783667052167168": {
                        "552784168849190912": {"552785000000000003": []}
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    # AppleDouble macOS junk file that must be ignored
    (t1 / "._structure.json").write_bytes(b"\x00\x05\x16\x07Mac OS X")
    (t1 / "reactions" / "._552783667052167168.json").write_bytes(b"\xff\xfeinvalid")

    # Cascade 2: ferguson rumour (false)
    t2 = base / "ferguson-all-rnr-threads" / "rumours" / "500281131057811456"
    _write_tweet_json(
        t2 / "source-tweets" / "500281131057811456.json",
        tid="500281131057811456",
        text="Rumor claims the National Guard has already shut down every highway.",
        created_at="Fri Aug 15 12:00:00 +0000 2014",
        user_id="8001",
    )
    _write_tweet_json(
        t2 / "reactions" / "500281999999000001.json",
        tid="500281999999000001",
        text="Not true, traffic is moving normally on I-70 right now.",
        created_at="Fri Aug 15 12:01:05 +0000 2014",
        user_id="8002",
        reply_to="500281131057811456",
    )
    (t2 / "annotation.json").write_text(
        json.dumps({"is_rumour": "rumour", "misinformation": 1, "true": 0}),
        encoding="utf-8",
    )
    (t2 / "structure.json").write_text(
        json.dumps({"500281131057811456": {"500281999999000001": []}}),
        encoding="utf-8",
    )

    # Cascade 3: sydneysiege non-rumour
    t3 = base / "sydneysiege-all-rnr-threads" / "non-rumours" / "544271640379392000"
    _write_tweet_json(
        t3 / "source-tweets" / "544271640379392000.json",
        tid="544271640379392000",
        text="Commuters advised to avoid Martin Place station due to police operation.",
        created_at="Mon Dec 15 00:10:00 +0000 2014",
        user_id="7001",
    )
    _write_tweet_json(
        t3 / "reactions" / "544272000000000001.json",
        tid="544272000000000001",
        text="Thanks for the heads up, taking the bus instead.",
        created_at="Mon Dec 15 00:12:00 +0000 2014",
        user_id="7002",
        reply_to="544271640379392000",
    )

    return root_dir


def test_archive_integrity_and_safe_extraction(tmp_path: Path) -> None:
    src_corpus = _build_mock_pheme_corpus(tmp_path / "src_corpus")
    archive_path = tmp_path / "PHEME_veracity.tar.bz2"

    with tarfile.open(archive_path, "w:bz2") as tar:
        for file_path in src_corpus.rglob("*"):
            rel = file_path.relative_to(src_corpus).as_posix()
            tar.add(file_path, arcname=rel)

    md5_hex, sha256_hex = compute_hashes(archive_path)
    assert verify_archive_integrity(archive_path, expected_md5=md5_hex, expected_sha256=sha256_hex) == (
        md5_hex,
        sha256_hex,
    )

    # Corrupt checksum check must raise ValueError
    with pytest.raises(ValueError, match="MD5 checksum mismatch"):
        verify_archive_integrity(archive_path, expected_md5="0" * 32)

    extracted = extract_pheme_archive(archive_path, tmp_path / "extracted")
    assert (extracted / "all-rnr-annotated-threads").is_dir()
    # AppleDouble files must not be extracted
    assert not any(p.name.startswith("._") for p in extracted.rglob("*"))

    # Path traversal archive must be rejected
    evil_tar = tmp_path / "evil.tar.bz2"
    with tarfile.open(evil_tar, "w:bz2") as tar:
        info = tarfile.TarInfo(name="../escape.txt")
        data = b"pwn"
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    with pytest.raises(ValueError, match="unsafe tar member"):
        extract_pheme_archive(evil_tar, tmp_path / "evil_out")


def test_veracity_and_structure_helpers() -> None:
    assert convert_veracity_label({"misinformation": 0, "true": 1}, is_rumour=True) == "true"
    assert convert_veracity_label({"misinformation": 1, "true": 0}, is_rumour=True) == "false"
    assert convert_veracity_label({"misinformation": 0, "true": 0}, is_rumour=True) == "unverified"
    assert convert_veracity_label({"misinformation": 0}, is_rumour=True) == "unverified"
    assert convert_veracity_label({"misinformation": 1, "true": 0}, is_rumour=False) == "non-rumor"

    parent_map = flatten_structure_parent_map({"r": {"c1": {"c2": []}, "c3": []}})
    assert parent_map == {"c1": "r", "c2": "c1", "c3": "r"}


def test_pheme_trees_valid_connected_acyclic_and_utc(tmp_path: Path) -> None:
    raw_dir = _build_mock_pheme_corpus(tmp_path / "raw_pheme")
    out_dir = tmp_path / "interim_pheme"
    nodes_path, edges_path = process_pheme_dataset(raw_dir, output_dir=out_dir)

    assert nodes_path.is_file() and edges_path.is_file()
    nodes_table, edges_table = load_cascade_parquet(out_dir, validate=True)
    assert validate_cascade_contract(nodes_table, edges_table, raise_on_error=False) == []

    nodes = nodes_table.to_pylist()
    edges = edges_table.to_pylist()

    by_cascade_nodes: dict[str, list[dict]] = {}
    for row in nodes:
        by_cascade_nodes.setdefault(row["cascade_id"], []).append(row)
    by_cascade_edges: dict[str, list[dict]] = {}
    for row in edges:
        by_cascade_edges.setdefault(row["cascade_id"], []).append(row)

    assert len(by_cascade_nodes) == 3

    for cid, c_nodes in by_cascade_nodes.items():
        c_edges = by_cascade_edges.get(cid, [])
        roots = [n for n in c_nodes if n["is_root"]]
        assert len(roots) == 1
        root_id = roots[0]["node_id"]
        assert roots[0]["parent_id"] is None

        # Tree invariant: |E| == |V| - 1 and connected directed tree reachable from root
        assert len(c_edges) == len(c_nodes) - 1
        in_degree = {n["node_id"]: 0 for n in c_nodes}
        children: dict[str, list[str]] = {n["node_id"]: [] for n in c_nodes}
        for e in c_edges:
            in_degree[e["target_id"]] += 1
            children[e["source_id"]].append(e["target_id"])
            assert e["time_delay_sec"] >= 0.0

        assert in_degree[root_id] == 0
        assert all(deg == 1 for nid, deg in in_degree.items() if nid != root_id)

        visited = {root_id}
        q = deque([root_id])
        while q:
            curr = q.popleft()
            for nxt in children[curr]:
                assert nxt not in visited  # no cycle
                visited.add(nxt)
                q.append(nxt)
        assert visited == {n["node_id"] for n in c_nodes}

    # Verify hashed user IDs (never raw numeric IDs) and recovered parent + pruned orphan
    c1_nodes = next(v for k, v in by_cascade_nodes.items() if "552783238415265792" in k)
    c1_ids = {n["node_id"] for n in c1_nodes}
    assert "pheme_552783238415265792_552784168849190912" in c1_ids
    assert "pheme_552783238415265792_552785000000000003" in c1_ids
    assert "pheme_552783238415265792_552785999999999999" not in c1_ids  # orphan pruned
    expected_root_hash = hashlib.sha256(b"9001").hexdigest()
    root_c1 = next(n for n in c1_nodes if n["is_root"])
    assert root_c1["user_id_hash"] == expected_root_hash
    assert root_c1["rumor_label"] == "true"


def test_discover_local_pheme_prefixed_dir_ignores_appledouble_and_parses_singular_source_tweet(
    tmp_path: Path,
) -> None:
    """Shallow BFS locates `all-rnr-annotated-threads_1`, ignores `._*`, and reads singular `source-tweet`."""
    k_root = tmp_path / "kaggle_input"
    ds_root = k_root / "pheme-dataset"
    prefixed_root = ds_root / "all-rnr-annotated-threads_1"
    ev_dir = prefixed_root / "charliehebdo-all-rnr-threads"
    tdir = ev_dir / "rumours" / "552783238415265792"

    # AppleDouble macOS metadata file at top and inside tweet folders must never break discovery/parsing
    prefixed_root.mkdir(parents=True)
    (ds_root / "._all-rnr-annotated-threads_1").write_bytes(b"macos_meta")

    _write_tweet_json(
        tdir / "source-tweet" / "552783238415265792.json",
        tid="552783238415265792",
        text="Singular source-tweet root",
        created_at="Wed Jan 07 11:00:00 +0000 2015",
        user_id="9001",
    )
    (tdir / "source-tweet" / "._552783238415265792.json").write_bytes(b"macos_meta")
    _write_tweet_json(
        tdir / "reactions" / "552784168849190912.json",
        tid="552784168849190912",
        text="Valid child reply",
        created_at="Wed Jan 07 11:02:00 +0000 2015",
        user_id="9002",
        reply_to="552783238415265792",
    )
    (tdir / "reactions" / "._552784168849190912.json").write_bytes(b"macos_meta")
    (tdir / "annotation.json").write_text(json.dumps({"is_rumour": "rumour", "true": "1"}), encoding="utf-8")
    (tdir / "structure.json").write_text(
        json.dumps({"552783238415265792": {"552784168849190912": []}}),
        encoding="utf-8",
    )

    found_arc, found_dir = _discover_local_pheme_source(tmp_path / "empty_raw", extra_roots=(k_root,))
    assert found_arc is None
    assert found_dir == prefixed_root

    events = _find_event_directories(ds_root)
    assert events == [ev_dir]

    out_dir = tmp_path / "out_singular"
    nodes_path, edges_path = process_pheme_dataset(raw_root=found_dir, output_dir=out_dir)
    assert nodes_path.is_file() and edges_path.is_file()
    n_tbl, e_tbl = load_cascade_parquet(out_dir, validate=True)
    assert n_tbl.num_rows == 2 and e_tbl.num_rows == 1

    # Modern 2026 Kaggle container depth-4 mount: /kaggle/input/datasets/<owner>/<slug>/all-rnr-annotated-threads_1
    k_v2 = tmp_path / "kaggle_input_v2"
    v2_prefixed = k_v2 / "datasets" / "manuelcecerepalazzo" / "pheme-dataset" / "all-rnr-annotated-threads_1"
    (v2_prefixed / "charliehebdo-all-rnr-threads" / "rumours").mkdir(parents=True)
    v2_arc, v2_dir = _discover_local_pheme_source(tmp_path / "empty_raw_v2", extra_roots=(k_v2,))
    assert v2_arc is None
    assert v2_dir == v2_prefixed


def test_process_pheme_dataset_multithreaded_deterministic_parity(tmp_path: Path) -> None:
    """ThreadPoolExecutor + os.scandir parsing must be row-for-row identical to sequential max_workers=1."""
    raw_root = tmp_path / "pheme_raw" / "all-rnr-annotated-threads_1"
    events_spec = [
        ("charliehebdo-all-rnr-threads", "rumours", "source-tweets", "1001", "true"),
        ("charliehebdo-all-rnr-threads", "non-rumours", "source-tweet", "1002", None),
        ("ferguson-all-rnr-threads", "rumours", "source-tweet", "2001", "false"),
        ("ferguson-all-rnr-threads", "non-rumours", "source-tweets", "2002", None),
    ]
    for ev_name, sub_name, src_folder, root_tid, veracity in events_spec:
        tdir = raw_root / ev_name / sub_name / root_tid
        is_r = sub_name == "rumours"
        _write_tweet_json(
            tdir / src_folder / f"{root_tid}.json",
            tid=root_tid,
            text=f"Source post {root_tid}",
            created_at="Wed Jan 07 11:00:00 +0000 2015",
            user_id="8001",
        )
        (tdir / src_folder / f"._{root_tid}.json").write_bytes(b"appledouble")
        child_tid = f"{root_tid}1"
        _write_tweet_json(
            tdir / "reactions" / f"{child_tid}.json",
            tid=child_tid,
            text=f"Reply to {root_tid}",
            created_at="Wed Jan 07 11:05:00 +0000 2015",
            user_id="8002",
            reply_to=root_tid,
        )
        (tdir / "reactions" / f"._{child_tid}.json").write_bytes(b"appledouble")
        # AppleDouble sibling directory entry at thread level must be skipped cleanly
        (tdir.parent / f"._{root_tid}").write_bytes(b"appledouble")
        ann = {"is_rumour": "rumour" if is_r else "nonrumour"}
        if veracity is not None:
            ann["misinformation"] = "1" if veracity == "false" else "0"
            ann["true"] = "1" if veracity == "true" else "0"
        (tdir / "annotation.json").write_text(json.dumps(ann), encoding="utf-8")
        (tdir / "structure.json").write_text(
            json.dumps({root_tid: {child_tid: []}}),
            encoding="utf-8",
        )

    out_seq = tmp_path / "out_seq"
    out_par = tmp_path / "out_par"
    process_pheme_dataset(raw_root=raw_root, output_dir=out_seq, max_workers=1)
    process_pheme_dataset(raw_root=raw_root, output_dir=out_par, max_workers=8)
    n_seq, e_seq = load_cascade_parquet(out_seq, validate=True)
    n_par, e_par = load_cascade_parquet(out_par, validate=True)
    assert n_seq.to_pylist() == n_par.to_pylist()
    assert e_seq.to_pylist() == e_par.to_pylist()
    assert n_par.num_rows == 8 and e_par.num_rows == 4

    # Verify max_threads bound matches sequentially and across 8 workers
    out_cap_seq = tmp_path / "out_cap_seq"
    out_cap_par = tmp_path / "out_cap_par"
    process_pheme_dataset(raw_root=raw_root, output_dir=out_cap_seq, max_threads=3, max_workers=1)
    process_pheme_dataset(raw_root=raw_root, output_dir=out_cap_par, max_threads=3, max_workers=8)
    nc_seq, ec_seq = load_cascade_parquet(out_cap_seq, validate=True)
    nc_par, ec_par = load_cascade_parquet(out_cap_par, validate=True)
    assert nc_seq.to_pylist() == nc_par.to_pylist()
    assert ec_seq.to_pylist() == ec_par.to_pylist()
    assert nc_par.num_rows == 6


