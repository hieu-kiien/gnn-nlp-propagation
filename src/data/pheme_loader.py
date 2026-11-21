"""PHEME dataset downloader and parser from Figshare (article 6392078).

Downloads the public `PHEME_veracity.tar.bz2` archive directly from Figshare
without requiring Google Drive or X/Twitter API credentials, verifies archive
integrity via MD5/SHA-256 checksums, reconstructs directed propagation trees
(`source-tweets`, `reactions`, `structure.json`, `annotation.json`), normalizes
timestamps strictly to UTC, and persists validated `nodes.parquet` and
`edges.parquet` tables under `data/interim/pheme/`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import tempfile
import time
import unicodedata
from collections import deque
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import requests

from src.data.arctic_shift_vn import hash_author_sha256
from src.data.schema import (
    build_edges_table,
    build_nodes_table,
    save_cascade_parquet,
)
from src.utils.text import normalize_nfc

FIGSHARE_ARTICLE_API = "https://api.figshare.com/v2/articles/6392078"
PHEME_ARCHIVE_FILENAME = "PHEME_veracity.tar.bz2"
PHEME_FALLBACK_DOWNLOAD_URL = "https://ndownloader.figshare.com/files/11767817"
PHEME_EXPECTED_MD5 = "11530d4c0c7127fc78bbc1e46f2498f8"


def _sha256_hex(value: str) -> str:
    """Pseudonymize PHEME user ID using salted SHA-256 (aligned with ADR-01)."""
    hashed = hash_author_sha256(value)
    return hashed if hashed is not None else hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_text_nfc(text: str) -> str:
    return normalize_nfc(text)


def compute_hashes(file_path: Path) -> tuple[str, str]:
    """Compute `(md5_hex, sha256_hex)` digests over `file_path` in 1 MiB chunks."""
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with Path(file_path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def verify_archive_integrity(
    file_path: Path,
    expected_md5: str | None = PHEME_EXPECTED_MD5,
    expected_sha256: str | None = None,
) -> tuple[str, str]:
    """Verify file integrity against expected MD5 and/or SHA-256 checksums."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Archive file not found: {path}")

    actual_md5, actual_sha256 = compute_hashes(path)
    if expected_md5 is not None and actual_md5.lower() != expected_md5.strip().lower():
        raise ValueError(
            f"MD5 checksum mismatch for '{path}': expected {expected_md5}, got {actual_md5}"
        )
    if expected_sha256 is not None and actual_sha256.lower() != expected_sha256.strip().lower():
        raise ValueError(
            f"SHA-256 checksum mismatch for '{path}': expected {expected_sha256}, got {actual_sha256}"
        )
    return actual_md5, actual_sha256


FIGSHARE_HTTP_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 SBiFAGAT-Research/1.0"
    ),
    "Accept": "*/*",
}


def _resolve_figshare_file_metadata(timeout: int = 30) -> tuple[str, str]:
    try:
        resp = requests.get(
            FIGSHARE_ARTICLE_API,
            headers=FIGSHARE_HTTP_HEADERS,
            timeout=min(timeout, 30),
        )
        resp.raise_for_status()
        payload = resp.json()
        for entry in payload.get("files", []):
            if entry.get("name") == PHEME_ARCHIVE_FILENAME and entry.get("download_url"):
                url = str(entry["download_url"])
                md5 = str(entry.get("computed_md5") or entry.get("supplied_md5") or PHEME_EXPECTED_MD5)
                return url, md5
    except (requests.RequestException, ValueError, KeyError):
        pass
    return PHEME_FALLBACK_DOWNLOAD_URL, PHEME_EXPECTED_MD5


def _discover_local_pheme_source(
    raw_dir: Path,
    extra_roots: Sequence[Path] = (Path("/kaggle/input"),),
) -> tuple[Path | None, Path | None]:
    """Locate a user-provided local/Kaggle-Input PHEME archive or extracted event directory."""
    roots = [Path(raw_dir), *[Path(r) for r in extra_roots if Path(r).exists()]]
    for root in roots:
        if not root.exists():
            continue
        direct = root / PHEME_ARCHIVE_FILENAME
        if direct.is_file():
            return direct, None
        for cand in sorted(root.rglob(PHEME_ARCHIVE_FILENAME)):
            if cand.is_file():
                return cand, None
        for cand_dir in sorted(root.rglob("all-rnr-annotated-threads")):
            if cand_dir.is_dir() and _find_event_directories(cand_dir):
                return None, cand_dir
    return None, None


def download_pheme_from_figshare(
    raw_dir: Path = Path("data/raw/pheme"),
    timeout: int = 120,
) -> Path:
    """Download `PHEME_veracity.tar.bz2` from Figshare (or use verified local/Kaggle-Input copy)."""
    dest_dir = Path(raw_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dest_dir / PHEME_ARCHIVE_FILENAME
    sha_sidecar = dest_dir / f"{PHEME_ARCHIVE_FILENAME}.sha256"

    download_url, expected_md5 = _resolve_figshare_file_metadata(timeout=timeout)
    if "drive.google.com" in download_url or "api.twitter.com" in download_url:
        raise RuntimeError(f"Disallowed archive URL outside Figshare: {download_url}")

    local_archive, _ = _discover_local_pheme_source(dest_dir)
    if local_archive is not None and local_archive.is_file():
        try:
            _, sha256_hex = verify_archive_integrity(local_archive, expected_md5=expected_md5)
            if local_archive == archive_path and not sha_sidecar.exists():
                sha_sidecar.write_text(f"{sha256_hex}  {archive_path.name}\n", encoding="utf-8")
            return local_archive
        except ValueError:
            if local_archive == archive_path:
                archive_path.unlink(missing_ok=True)

    part_path = dest_dir / f"{PHEME_ARCHIVE_FILENAME}.part"
    last_exc: Exception | None = None
    with requests.Session() as session:
        session.headers.update(FIGSHARE_HTTP_HEADERS)
        for attempt in range(3):
            part_path.unlink(missing_ok=True)
            try:
                with session.get(download_url, stream=True, timeout=timeout) as resp:
                    resp.raise_for_status()
                    if resp.status_code != 200:
                        raise RuntimeError(
                            f"Figshare returned HTTP {resp.status_code} (likely AWS WAF / anti-bot "
                            f"challenge on {download_url}) instead of HTTP 200 archive bytes. "
                            "Please download `PHEME_veracity.tar.bz2` manually in your browser from "
                            "https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078 "
                            "and attach it as a Kaggle Input Dataset (auto-detected under /kaggle/input) "
                            "or place it at data/raw/pheme/PHEME_veracity.tar.bz2."
                        )
                    with part_path.open("wb") as fh:
                        for chunk in resp.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                fh.write(chunk)
                last_exc = None
                break
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < 2:
                    time.sleep(2.0 * (attempt + 1))
    if last_exc is not None:
        part_path.unlink(missing_ok=True)
        raise last_exc

    try:
        _, sha256_hex = verify_archive_integrity(part_path, expected_md5=expected_md5)
    except ValueError:
        part_path.unlink(missing_ok=True)
        raise

    os.replace(part_path, archive_path)
    sha_sidecar.write_text(f"{sha256_hex}  {archive_path.name}\n", encoding="utf-8")
    return archive_path


def _is_appledouble_or_junk(member_name: str) -> bool:
    posix = PurePosixPath(member_name)
    return any(part == "__MACOSX" or part.startswith("._") for part in posix.parts)


def extract_pheme_archive(archive_path: Path, extract_dir: Path) -> Path:
    """Safely extract `PHEME_veracity.tar.bz2`, blocking path traversal and macOS junk."""
    out_root = Path(extract_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    with tarfile.open(Path(archive_path), "r:bz2") as tar:
        for member in tar.getmembers():
            if not member.name or _is_appledouble_or_junk(member.name):
                continue

            posix = PurePosixPath(member.name)
            if posix.is_absolute() or ".." in posix.parts:
                raise ValueError(f"Blocked unsafe tar member path: {member.name!r}")

            target_path = (out_root / posix).resolve()
            if target_path != out_root and not target_path.is_relative_to(out_root):
                raise ValueError(f"Blocked tar path traversal outside '{out_root}': {member.name!r}")

            if member.issym() or member.islnk():
                raise ValueError(f"Blocked link entry in archive: {member.name!r}")

            if member.isdir():
                target_path.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                src_fh = tar.extractfile(member)
                if src_fh is None:
                    continue
                with src_fh, target_path.open("wb") as dst_fh:
                    while True:
                        buf = src_fh.read(1024 * 1024)
                        if not buf:
                            break
                        dst_fh.write(buf)

    return out_root


def convert_veracity_label(annotation: Mapping[str, Any] | None, is_rumour: bool) -> str:
    """Map PHEME `annotation.json` veracity fields using Figshare rules."""
    if not is_rumour:
        return "non-rumor"
    if not annotation:
        return "rumor"

    has_misinfo = "misinformation" in annotation and annotation["misinformation"] is not None
    has_true = "true" in annotation and annotation["true"] is not None

    if has_misinfo and has_true:
        misinfo = int(annotation["misinformation"])
        is_true = int(annotation["true"])
        if misinfo == 0 and is_true == 1:
            return "true"
        if misinfo == 1 and is_true == 0:
            return "false"
        if misinfo == 0 and is_true == 0:
            return "unverified"
        return "rumor"

    if has_misinfo:
        misinfo = int(annotation["misinformation"])
        return "unverified" if misinfo == 0 else "false"

    return "rumor"


def parse_twitter_timestamp_utc(created_at: str) -> datetime:
    """Parse Twitter timestamp strings strictly into timezone-aware UTC `datetime`."""
    raw = created_at.strip()
    try:
        dt = datetime.strptime(raw, "%a %b %d %H:%M:%S %z %Y")
        return dt.astimezone(timezone.utc)
    except ValueError:
        pass

    normalized = raw[:-1] + "+00:00" if raw.endswith(("Z", "z")) else raw
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError(f"Twitter/ISO timestamp lacks timezone offset: {created_at!r}")
    return parsed.astimezone(timezone.utc)


def flatten_structure_parent_map(structure: Any) -> dict[str, str]:
    """Flatten nested PHEME `structure.json` into `{child_id: parent_id}`.

    Handles PHEME leaf quirks where empty children are encoded as `[]` instead of `{}`.
    """
    parent_map: dict[str, str] = {}

    def _walk(parent_id: str, subtree: Any) -> None:
        if not isinstance(subtree, Mapping):
            return
        for raw_child_id, grandchildren in subtree.items():
            child_id = str(raw_child_id)
            if child_id and child_id != parent_id and child_id not in parent_map:
                parent_map[child_id] = parent_id
            _walk(child_id, grandchildren)

    if isinstance(structure, Mapping):
        for raw_node_id, children in structure.items():
            _walk(str(raw_node_id), children)

    return parent_map


def _read_json_safe(path: Path) -> Any | None:
    try:
        with Path(path).open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _extract_tweet_fields(tweet: Mapping[str, Any]) -> tuple[str, str, datetime, str | None] | None:
    raw_id = tweet.get("id_str") or tweet.get("id")
    if raw_id is None:
        return None
    tid = str(raw_id).strip()
    if not tid:
        return None

    raw_text = tweet.get("full_text") if tweet.get("full_text") is not None else tweet.get("text")
    if not isinstance(raw_text, str):
        return None
    text = _normalize_text_nfc(raw_text)
    if not text:
        return None

    created_at = tweet.get("created_at")
    if not isinstance(created_at, str) or not created_at.strip():
        return None
    try:
        dt_utc = parse_twitter_timestamp_utc(created_at)
    except ValueError:
        return None

    user_hash: str | None = None
    user_obj = tweet.get("user")
    if isinstance(user_obj, Mapping):
        uid = user_obj.get("id_str") or user_obj.get("id")
        if uid is not None and str(uid).strip():
            user_hash = _sha256_hex(str(uid).strip())

    return tid, text, dt_utc, user_hash


def parse_pheme_thread(
    thread_dir: Path,
    topic_id: str,
    is_rumour: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    """Parse one PHEME thread directory into `(node_records, edge_records)`."""
    tdir = Path(thread_dir)
    source_dir = tdir / "source-tweets"
    if not source_dir.is_dir():
        return None

    source_files = sorted(
        p for p in source_dir.glob("*.json") if p.is_file() and not p.name.startswith(".")
    )
    if not source_files:
        return None

    preferred_src = next((p for p in source_files if p.stem == tdir.name), source_files[0])
    source_json = _read_json_safe(preferred_src)
    if not isinstance(source_json, Mapping):
        return None

    root_parsed = _extract_tweet_fields(source_json)
    if root_parsed is None:
        return None
    root_id, root_text, root_dt, root_user_hash = root_parsed

    ann_json = _read_json_safe(tdir / "annotation.json")
    rumor_label = convert_veracity_label(
        ann_json if isinstance(ann_json, Mapping) else None,
        is_rumour=is_rumour,
    )

    struct_json = _read_json_safe(tdir / "structure.json")
    structure_parent_map = flatten_structure_parent_map(struct_json)

    tweets: dict[str, tuple[str, datetime, str | None, str | None]] = {
        root_id: (root_text, root_dt, root_user_hash, None)
    }

    reactions_dir = tdir / "reactions"
    if reactions_dir.is_dir():
        for rfile in sorted(reactions_dir.glob("*.json")):
            if not rfile.is_file() or rfile.name.startswith("."):
                continue
            rjson = _read_json_safe(rfile)
            if not isinstance(rjson, Mapping):
                continue
            parsed = _extract_tweet_fields(rjson)
            if parsed is None:
                continue
            tid, text, dt_utc, user_hash = parsed
            if tid == root_id or tid in tweets:
                continue
            raw_reply = rjson.get("in_reply_to_status_id_str")
            if raw_reply is None and rjson.get("in_reply_to_status_id") is not None:
                raw_reply = str(rjson["in_reply_to_status_id"])
            reply_to = str(raw_reply).strip() if raw_reply is not None else None
            tweets[tid] = (text, dt_utc, user_hash, reply_to or None)

    # Resolve parent pointers (prefer direct tweet field if parent exists, fallback to structure.json)
    children_adj: dict[str, list[str]] = {root_id: []}
    parent_of: dict[str, str] = {}
    for tid, (_, _, _, reply_to) in tweets.items():
        if tid == root_id:
            continue
        chosen_parent: str | None = None
        if reply_to and reply_to != tid and reply_to in tweets:
            chosen_parent = reply_to
        else:
            struct_parent = structure_parent_map.get(tid)
            if struct_parent and struct_parent != tid and struct_parent in tweets:
                chosen_parent = struct_parent

        if chosen_parent is not None:
            parent_of[tid] = chosen_parent
            children_adj.setdefault(chosen_parent, []).append(tid)

    # Sort children chronologically and restrict to BFS-connected component from root
    for plist in children_adj.values():
        plist.sort(key=lambda nid: (tweets[nid][1], nid))

    cascade_id = f"pheme_{topic_id}_{root_id}"
    node_records: list[dict[str, Any]] = []
    edge_records: list[dict[str, Any]] = []
    effective_dt: dict[str, datetime] = {root_id: root_dt}

    visited: set[str] = {root_id}
    queue: deque[str] = deque([root_id])

    while queue:
        curr_id = queue.popleft()
        curr_text, _, curr_user_hash, _ = tweets[curr_id]
        is_root = curr_id == root_id
        curr_parent_tid = None if is_root else parent_of[curr_id]

        node_records.append(
            {
                "node_id": f"pheme_{root_id}_{curr_id}",
                "cascade_id": cascade_id,
                "topic_id": topic_id,
                "parent_id": None if is_root else f"pheme_{root_id}_{curr_parent_tid}",
                "is_root": is_root,
                "text": curr_text,
                "created_at_utc": effective_dt[curr_id],
                "lang": "en",
                "source_dataset": "pheme",
                "user_id_hash": curr_user_hash,
                "rumor_label": rumor_label,
            }
        )

        if not is_root and curr_parent_tid is not None:
            delay_sec = max(
                0.0,
                (effective_dt[curr_id] - effective_dt[curr_parent_tid]).total_seconds(),
            )
            edge_records.append(
                {
                    "edge_id": f"pheme_{root_id}_{curr_parent_tid}->{curr_id}",
                    "cascade_id": cascade_id,
                    "source_id": f"pheme_{root_id}_{curr_parent_tid}",
                    "target_id": f"pheme_{root_id}_{curr_id}",
                    "created_at_utc": effective_dt[curr_id],
                    "time_delay_sec": float(delay_sec),
                    "edge_type": "reply",
                }
            )

        for child_id in children_adj.get(curr_id, []):
            if child_id in visited:
                continue
            visited.add(child_id)
            child_raw_dt = tweets[child_id][1]
            # Clamp minor Twitter clock skew strictly preserving temporal causality
            effective_dt[child_id] = max(child_raw_dt, effective_dt[curr_id])
            queue.append(child_id)

    return node_records, edge_records


def _find_event_directories(raw_root: Path) -> list[Path]:
    root = Path(raw_root)
    nested_root = root / "all-rnr-annotated-threads"
    search_root = nested_root if nested_root.is_dir() else root

    event_dirs = sorted(
        p
        for p in search_root.iterdir()
        if p.is_dir()
        and not p.name.startswith(".")
        and p.name != "__MACOSX"
        and ((p / "rumours").is_dir() or (p / "non-rumours").is_dir())
    )
    if event_dirs:
        return event_dirs

    # Fallback: recursive discovery of *-all-rnr-threads dirs
    return sorted(
        p
        for p in root.rglob("*all-rnr-threads")
        if p.is_dir() and "__MACOSX" not in p.parts and not p.name.startswith(".")
    )


def _topic_id_from_event_dir(event_dir: Path) -> str:
    name = event_dir.name
    if name.endswith("-all-rnr-threads"):
        return name[: -len("-all-rnr-threads")]
    return name


def process_pheme_dataset(
    raw_root: Path,
    output_dir: Path = Path("data/interim/pheme"),
    max_threads: int | None = None,
) -> tuple[Path, Path]:
    """Parse all extracted PHEME event trees and save validated Parquet tables."""
    event_dirs = _find_event_directories(Path(raw_root))
    if not event_dirs:
        raise FileNotFoundError(
            f"No PHEME event directories found under '{raw_root}'. Run download/extraction first."
        )

    all_nodes: list[dict[str, Any]] = []
    all_edges: list[dict[str, Any]] = []
    seen_roots: set[str] = set()
    thread_count = 0

    for event_dir in event_dirs:
        topic_id = _topic_id_from_event_dir(event_dir)
        for sub_name, is_rumour in (("rumours", True), ("non-rumours", False)):
            sub_dir = event_dir / sub_name
            if not sub_dir.is_dir():
                continue
            for thread_dir in sorted(sub_dir.iterdir()):
                if not thread_dir.is_dir() or thread_dir.name.startswith("."):
                    continue
                if thread_dir.name in seen_roots:
                    continue
                parsed = parse_pheme_thread(thread_dir, topic_id=topic_id, is_rumour=is_rumour)
                if parsed is None:
                    continue
                nodes, edges = parsed
                root_id = str(nodes[0]["node_id"]).rsplit("_", 1)[-1]
                if root_id in seen_roots:
                    continue
                seen_roots.add(root_id)
                seen_roots.add(thread_dir.name)
                all_nodes.extend(nodes)
                all_edges.extend(edges)
                thread_count += 1
                if max_threads is not None and thread_count >= max_threads:
                    break
            if max_threads is not None and thread_count >= max_threads:
                break
        if max_threads is not None and thread_count >= max_threads:
            break

    nodes_table = build_nodes_table(all_nodes)
    edges_table = build_edges_table(all_edges)
    return save_cascade_parquet(nodes_table, edges_table, output_dir=output_dir, validate=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and parse PHEME dataset from Figshare (6392078).")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/pheme"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/pheme"))
    parser.add_argument("--extract-dir", type=Path, default=None)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--max-threads", type=int, default=None)
    args = parser.parse_args()

    default_extract_dir = args.raw_dir / "extracted"
    discovered_archive, discovered_extracted = _discover_local_pheme_source(args.raw_dir)
    if args.extract_dir is not None:
        extract_dir = args.extract_dir
        if not args.skip_download and not _find_event_directories(extract_dir):
            archive_path = download_pheme_from_figshare(raw_dir=args.raw_dir)
            if not (extract_dir / "all-rnr-annotated-threads").exists():
                extract_pheme_archive(archive_path, extract_dir)
        parse_root = extract_dir if _find_event_directories(extract_dir) else args.raw_dir
        nodes_path, edges_path = process_pheme_dataset(
            raw_root=parse_root,
            output_dir=args.output_dir,
            max_threads=args.max_threads,
        )
    elif args.skip_download or (default_extract_dir / "all-rnr-annotated-threads").exists():
        parse_root = default_extract_dir if _find_event_directories(default_extract_dir) else args.raw_dir
        nodes_path, edges_path = process_pheme_dataset(
            raw_root=parse_root,
            output_dir=args.output_dir,
            max_threads=args.max_threads,
        )
    elif discovered_extracted is not None:
        print(f"Using pre-extracted PHEME directory: {discovered_extracted}")
        nodes_path, edges_path = process_pheme_dataset(
            raw_root=discovered_extracted,
            output_dir=args.output_dir,
            max_threads=args.max_threads,
        )
    else:
        archive_path = (
            discovered_archive
            if discovered_archive is not None
            else download_pheme_from_figshare(raw_dir=args.raw_dir)
        )
        if discovered_archive is not None:
            verify_archive_integrity(discovered_archive, expected_md5=PHEME_EXPECTED_MD5)
        with tempfile.TemporaryDirectory(prefix="pheme_extract_") as tmp_dir:
            tmp_extract = Path(tmp_dir)
            extract_pheme_archive(archive_path, tmp_extract)
            parse_root = tmp_extract if _find_event_directories(tmp_extract) else args.raw_dir
            nodes_path, edges_path = process_pheme_dataset(
                raw_root=parse_root,
                output_dir=args.output_dir,
                max_threads=args.max_threads,
            )
    print(f"Saved validated PHEME Parquet tables: {nodes_path}, {edges_path}")


if __name__ == "__main__":
    main()
