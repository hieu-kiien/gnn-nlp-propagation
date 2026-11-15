"""Unit tests for the SBi-FAGAT Data Contract (`src/data/schema.py`)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from src.data.schema import (
    EDGES_ARROW_SCHEMA,
    NODES_ARROW_SCHEMA,
    EdgeRecord,
    ErrorCode,
    NodeRecord,
    SchemaValidationError,
    build_edges_table,
    build_nodes_table,
    load_cascade_parquet,
    parse_strict_utc_timestamp,
    save_cascade_parquet,
    validate_cascade_contract,
    validate_edges_table,
    validate_nodes_table,
)


def _make_valid_pheme_cascade() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes = [
        {
            "node_id": "pheme_001_r",
            "cascade_id": "pheme_001",
            "topic_id": "charliehebdo",
            "parent_id": None,
            "is_root": True,
            "text": "Breaking news update from Paris headquarters.",
            "created_at_utc": "2026-03-15T08:00:00Z",
            "lang": "en",
            "source_dataset": "pheme",
            "user_id_hash": "u_hash_0",
            "rumor_label": "false",
        },
        {
            "node_id": "pheme_001_c1",
            "cascade_id": "pheme_001",
            "topic_id": "charliehebdo",
            "parent_id": "pheme_001_r",
            "is_root": False,
            "text": "Wait, official police statement says this report is false.",
            "created_at_utc": "2026-03-15T08:01:30+00:00",
            "lang": "en",
            "source_dataset": "pheme",
            "user_id_hash": "u_hash_1",
            "rumor_label": "false",
        },
        {
            "node_id": "pheme_001_c2",
            "cascade_id": "pheme_001",
            "topic_id": "charliehebdo",
            "parent_id": "pheme_001_c1",
            "is_root": False,
            "text": "Agreed, no credible outlet has confirmed it.",
            "created_at_utc": datetime(2026, 3, 15, 8, 3, 0, tzinfo=timezone.utc),
            "lang": "en",
            "source_dataset": "pheme",
            "user_id_hash": "u_hash_2",
            "rumor_label": "false",
        },
    ]
    edges = [
        {
            "edge_id": "e_pheme_001_1",
            "cascade_id": "pheme_001",
            "source_id": "pheme_001_r",
            "target_id": "pheme_001_c1",
            "created_at_utc": "2026-03-15T08:01:30Z",
            "time_delay_sec": 90.0,
            "edge_type": "reply",
        },
        {
            "edge_id": "e_pheme_001_2",
            "cascade_id": "pheme_001",
            "source_id": "pheme_001_c1",
            "target_id": "pheme_001_c2",
            "created_at_utc": "2026-03-15T08:03:00Z",
            "time_delay_sec": 180.0,
            "edge_type": "reply",
        },
    ]
    return nodes, edges


def _make_valid_vi_cascade() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes = [
        {
            "node_id": "vi_001_r",
            "cascade_id": "vi_001",
            "topic_id": "chinh_sach_giao_thong",
            "parent_id": None,
            "is_root": True,
            "text": "Đề xuất điều chỉnh mức xử phạt vi phạm nồng độ cồn mới.",
            "created_at_utc": "2026-04-01T01:00:00Z",
            "lang": "vi",
            "source_dataset": "vn_social_2026",
            "user_id_hash": "vn_u0",
            "rumor_label": None,
        },
        {
            "node_id": "vi_001_c1",
            "cascade_id": "vi_001",
            "topic_id": "chinh_sach_giao_thong",
            "parent_id": "vi_001_r",
            "is_root": False,
            "text": "Mình ủng hộ hoàn toàn để giảm tai nạn giao thông.",
            "created_at_utc": "2026-04-01T01:05:00-00:00",
            "lang": "vi",
            "source_dataset": "vn_social_2026",
            "user_id_hash": "vn_u1",
            "rumor_label": None,
        },
    ]
    edges = [
        {
            "edge_id": "e_vi_001_1",
            "cascade_id": "vi_001",
            "source_id": "vi_001_r",
            "target_id": "vi_001_c1",
            "created_at_utc": "2026-04-01T01:05:00Z",
            "time_delay_sec": 300.0,
            "edge_type": "quote",
        }
    ]
    return nodes, edges


# ---------------------------------------------------------------------------
# 1. Happy Path: Bilingual Schema + Parquet Round-Trip
# ---------------------------------------------------------------------------


def test_valid_bilingual_parquet_roundtrip(tmp_path: Path) -> None:
    en_nodes, en_edges = _make_valid_pheme_cascade()
    vi_nodes, vi_edges = _make_valid_vi_cascade()

    nodes_table = build_nodes_table(en_nodes + vi_nodes)
    edges_table = build_edges_table(en_edges + vi_edges)

    assert nodes_table.schema.equals(NODES_ARROW_SCHEMA, check_metadata=False)
    assert edges_table.schema.equals(EDGES_ARROW_SCHEMA, check_metadata=False)

    issues = validate_cascade_contract(nodes_table, edges_table, raise_on_error=True)
    assert issues == []

    nodes_path, edges_path = save_cascade_parquet(nodes_table, edges_table, tmp_path)
    assert nodes_path.exists()
    assert edges_path.exists()

    loaded_nodes, loaded_edges = load_cascade_parquet(tmp_path, validate=True)
    assert loaded_nodes.num_rows == 5
    assert loaded_edges.num_rows == 3
    assert loaded_nodes.schema.field("created_at_utc").type == pa.timestamp("us", tz="UTC")


def test_pydantic_models_accept_valid_and_reject_leakage_extra() -> None:
    en_nodes, en_edges = _make_valid_pheme_cascade()
    node_obj = NodeRecord(**en_nodes[0])
    edge_obj = EdgeRecord(**en_edges[0])
    assert node_obj.created_at_utc.tzinfo == timezone.utc
    assert edge_obj.time_delay_sec == 90.0

    with pytest.raises(Exception):
        NodeRecord(**{**en_nodes[0], "y_size": 4.2})


# ---------------------------------------------------------------------------
# 2. Missing Parent Node & Dangling / Cross-Cascade References
# ---------------------------------------------------------------------------


def test_missing_parent_node_nonexistent_id() -> None:
    nodes, edges = _make_valid_pheme_cascade()
    nodes[1]["parent_id"] = "non_existent_parent"
    edges[0]["source_id"] = "non_existent_parent"

    with pytest.raises(SchemaValidationError) as exc_info:
        validate_cascade_contract(nodes, edges)

    assert ErrorCode.MISSING_PARENT_NODE in exc_info.value.error_codes
    assert ErrorCode.DANGLING_EDGE_ENDPOINT in exc_info.value.error_codes


def test_missing_parent_node_null_on_non_root() -> None:
    nodes, edges = _make_valid_pheme_cascade()
    nodes[1]["parent_id"] = None

    with pytest.raises(SchemaValidationError) as exc_info:
        validate_cascade_contract(nodes, edges)

    assert ErrorCode.MISSING_PARENT_NODE in exc_info.value.error_codes


def test_dangling_edge_source_id() -> None:
    nodes, edges = _make_valid_pheme_cascade()
    edges[0]["source_id"] = "ghost_node_999"

    with pytest.raises(SchemaValidationError) as exc_info:
        validate_cascade_contract(nodes, edges)

    assert ErrorCode.DANGLING_EDGE_ENDPOINT in exc_info.value.error_codes


def test_cross_cascade_parent_reference_rejected() -> None:
    en_nodes, en_edges = _make_valid_pheme_cascade()
    vi_nodes, vi_edges = _make_valid_vi_cascade()

    # Point Vietnamese child to English cascade root
    vi_nodes[1]["parent_id"] = en_nodes[0]["node_id"]
    vi_edges[0]["source_id"] = en_nodes[0]["node_id"]

    with pytest.raises(SchemaValidationError) as exc_info:
        validate_cascade_contract(en_nodes + vi_nodes, en_edges + vi_edges)

    assert ErrorCode.CROSS_CASCADE_REFERENCE in exc_info.value.error_codes


# ---------------------------------------------------------------------------
# 3. Strict UTC Timestamp Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_timestamp",
    [
        "2026-03-15T08:30:00",           # Missing timezone suffix
        "2026-03-15T15:30:00+07:00",     # Non-UTC offset
        "2026-03-15",                    # Date only
        "not-a-utc-timestamp",           # Malformed string
        datetime(2026, 3, 15, 8, 30),    # Naive datetime
        datetime(2026, 3, 15, 15, 30, tzinfo=timezone(timedelta(hours=7))),
        1773563400,                      # Raw integer epoch
    ],
)
def test_invalid_utc_timestamp_rejected_in_records_and_parser(bad_timestamp: Any) -> None:
    with pytest.raises(ValueError):
        parse_strict_utc_timestamp(bad_timestamp)

    nodes, edges = _make_valid_pheme_cascade()
    nodes[1]["created_at_utc"] = bad_timestamp

    with pytest.raises(SchemaValidationError) as exc_info:
        build_nodes_table(nodes)
    assert ErrorCode.INVALID_UTC_TIMESTAMP in exc_info.value.error_codes

    with pytest.raises(SchemaValidationError) as contract_exc:
        validate_cascade_contract(nodes, edges)
    assert ErrorCode.INVALID_UTC_TIMESTAMP in contract_exc.value.error_codes


@pytest.mark.parametrize(
    "bad_arrow_ts_type",
    [
        pa.timestamp("us", tz=None),
        pa.timestamp("us", tz="Asia/Ho_Chi_Minh"),
        pa.string(),
    ],
)
def test_invalid_utc_arrow_column_type_rejected(bad_arrow_ts_type: pa.DataType) -> None:
    nodes, edges = _make_valid_pheme_cascade()
    valid_nodes = build_nodes_table(nodes)
    valid_edges = build_edges_table(edges)

    if pa.types.is_string(bad_arrow_ts_type):
        bad_col = pa.array(["2026-03-15T08:00:00Z", "2026-03-15T08:01:30Z", "2026-03-15T08:03:00Z"])
    else:
        bad_col = valid_nodes["created_at_utc"].cast(bad_arrow_ts_type)

    col_idx = valid_nodes.schema.get_field_index("created_at_utc")
    bad_nodes = valid_nodes.set_column(
        col_idx,
        pa.field("created_at_utc", bad_arrow_ts_type, nullable=False),
        bad_col,
    )

    issues = validate_nodes_table(bad_nodes)
    assert any(i.code == ErrorCode.INVALID_UTC_TIMESTAMP for i in issues)

    with pytest.raises(SchemaValidationError) as exc_info:
        validate_cascade_contract(bad_nodes, valid_edges)
    assert ErrorCode.INVALID_UTC_TIMESTAMP in exc_info.value.error_codes


# ---------------------------------------------------------------------------
# 4. Cycle Detection (Self-loop, 2-node Disconnected Cycle, 3-node Cycle)
# ---------------------------------------------------------------------------


def test_cycle_self_loop_detected() -> None:
    nodes, edges = _make_valid_pheme_cascade()
    nodes[1]["parent_id"] = nodes[1]["node_id"]
    edges[0]["source_id"] = nodes[1]["node_id"]

    with pytest.raises(SchemaValidationError) as exc_info:
        validate_cascade_contract(nodes, edges)

    assert ErrorCode.CYCLE_DETECTED in exc_info.value.error_codes


def test_cycle_two_node_disconnected_from_root_detected() -> None:
    nodes, edges = _make_valid_pheme_cascade()
    # Create a 2-node cycle c1 <-> c2 completely disconnected from root r
    nodes[1]["parent_id"] = "pheme_001_c2"
    nodes[2]["parent_id"] = "pheme_001_c1"
    edges[0]["source_id"] = "pheme_001_c2"
    edges[0]["target_id"] = "pheme_001_c1"

    with pytest.raises(SchemaValidationError) as exc_info:
        validate_cascade_contract(nodes, edges)

    assert ErrorCode.CYCLE_DETECTED in exc_info.value.error_codes


def test_cycle_three_node_directed_cycle_detected() -> None:
    nodes, edges = _make_valid_pheme_cascade()
    # Add 3rd child and form a 3-node cycle c1 -> c2 -> c3 -> c1
    nodes.append(
        {
            "node_id": "pheme_001_c3",
            "cascade_id": "pheme_001",
            "topic_id": "charliehebdo",
            "parent_id": "pheme_001_c2",
            "is_root": False,
            "text": "Third reply in cycle.",
            "created_at_utc": "2026-03-15T08:05:00Z",
            "lang": "en",
            "source_dataset": "pheme",
            "user_id_hash": "u_hash_3",
            "rumor_label": "false",
        }
    )
    nodes[1]["parent_id"] = "pheme_001_c3"
    edges[0]["source_id"] = "pheme_001_c3"
    edges[0]["target_id"] = "pheme_001_c1"
    edges.append(
        {
            "edge_id": "e_pheme_001_3",
            "cascade_id": "pheme_001",
            "source_id": "pheme_001_c2",
            "target_id": "pheme_001_c3",
            "created_at_utc": "2026-03-15T08:05:00Z",
            "time_delay_sec": 300.0,
            "edge_type": "reply",
        }
    )

    with pytest.raises(SchemaValidationError) as exc_info:
        validate_cascade_contract(nodes, edges)

    assert ErrorCode.CYCLE_DETECTED in exc_info.value.error_codes


# ---------------------------------------------------------------------------
# 5. Empty / Whitespace / Null Text
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("empty_val", ["", "   \t\n  ", None])
def test_empty_text_rejected(empty_val: str | None) -> None:
    nodes, edges = _make_valid_pheme_cascade()
    nodes[1]["text"] = empty_val

    with pytest.raises(SchemaValidationError) as exc_info:
        build_nodes_table(nodes)
    assert ErrorCode.EMPTY_TEXT in exc_info.value.error_codes

    with pytest.raises(SchemaValidationError) as contract_exc:
        validate_cascade_contract(nodes, edges)
    assert ErrorCode.EMPTY_TEXT in contract_exc.value.error_codes


def test_empty_text_rejected_in_arrow_table() -> None:
    nodes, edges = _make_valid_pheme_cascade()
    valid_nodes = build_nodes_table(nodes)
    valid_edges = build_edges_table(edges)

    idx = valid_nodes.schema.get_field_index("text")
    bad_nodes = valid_nodes.set_column(
        idx,
        pa.field("text", pa.string(), nullable=True),
        pa.array(["Root post", "   ", None]),
    )
    issues = validate_nodes_table(bad_nodes)
    assert sum(1 for i in issues if i.code == ErrorCode.EMPTY_TEXT) == 2

    with pytest.raises(SchemaValidationError) as exc_info:
        validate_cascade_contract(bad_nodes, valid_edges)
    assert ErrorCode.EMPTY_TEXT in exc_info.value.error_codes


# ---------------------------------------------------------------------------
# 6. Temporal Causality, Root Invariants & Zero Data Leakage
# ---------------------------------------------------------------------------


def test_temporal_order_violation_child_before_parent() -> None:
    nodes, edges = _make_valid_pheme_cascade()
    # c2 created before c1 (08:01:00 < 08:01:30)
    nodes[2]["created_at_utc"] = "2026-03-15T08:01:00Z"
    edges[1]["created_at_utc"] = "2026-03-15T08:01:00Z"

    with pytest.raises(SchemaValidationError) as exc_info:
        validate_cascade_contract(nodes, edges)

    assert ErrorCode.TEMPORAL_ORDER_VIOLATION in exc_info.value.error_codes


def test_negative_time_delay_rejected() -> None:
    nodes, edges = _make_valid_pheme_cascade()
    edges[0]["time_delay_sec"] = -5.0

    with pytest.raises(SchemaValidationError) as exc_info:
        validate_cascade_contract(nodes, edges)

    assert ErrorCode.TEMPORAL_ORDER_VIOLATION in exc_info.value.error_codes


def test_multiple_roots_and_missing_root_rejected() -> None:
    nodes, edges = _make_valid_pheme_cascade()
    # Two roots
    nodes[1]["is_root"] = True
    nodes[1]["parent_id"] = None
    with pytest.raises(SchemaValidationError) as exc_multi:
        validate_cascade_contract(nodes, edges)
    assert ErrorCode.MULTIPLE_ROOT_NODES in exc_multi.value.error_codes

    # Zero roots
    nodes2, edges2 = _make_valid_pheme_cascade()
    nodes2[0]["is_root"] = False
    with pytest.raises(SchemaValidationError) as exc_missing:
        validate_cascade_contract(nodes2, edges2)
    assert ErrorCode.MISSING_ROOT_NODE in exc_missing.value.error_codes


def test_root_with_parent_rejected() -> None:
    nodes, edges = _make_valid_pheme_cascade()
    nodes[0]["parent_id"] = "pheme_001_c1"

    with pytest.raises(SchemaValidationError) as exc_info:
        validate_cascade_contract(nodes, edges)

    assert ErrorCode.ROOT_HAS_PARENT in exc_info.value.error_codes


@pytest.mark.parametrize("leak_col", ["y_size", "y_polar", "n_total"])
def test_zero_data_leakage_forbidden_columns_rejected(leak_col: str) -> None:
    nodes, edges = _make_valid_pheme_cascade()
    valid_nodes = build_nodes_table(nodes)
    valid_edges = build_edges_table(edges)

    leaked_nodes = valid_nodes.append_column(leak_col, pa.array([1.5, 1.5, 1.5]))
    with pytest.raises(SchemaValidationError) as exc_nodes:
        validate_cascade_contract(leaked_nodes, valid_edges)
    assert ErrorCode.DATA_LEAKAGE_COLUMN in exc_nodes.value.error_codes

    leaked_edges = valid_edges.append_column(leak_col, pa.array([0.8, 0.8]))
    with pytest.raises(SchemaValidationError) as exc_edges:
        validate_cascade_contract(valid_nodes, leaked_edges)
    assert ErrorCode.DATA_LEAKAGE_COLUMN in exc_edges.value.error_codes


def test_validate_edges_table_catches_invalid_edge_type_and_bad_utc() -> None:
    _, edges = _make_valid_pheme_cascade()
    valid_edges = build_edges_table(edges)
    col_idx = valid_edges.schema.get_field_index("edge_type")
    bad_edges = valid_edges.set_column(
        col_idx,
        pa.field("edge_type", pa.string(), nullable=False),
        pa.array(["reply", "unknown_edge"]),
    )
    ts_idx = bad_edges.schema.get_field_index("created_at_utc")
    bad_edges = bad_edges.set_column(
        ts_idx,
        pa.field("created_at_utc", pa.timestamp("us", tz=None), nullable=False),
        bad_edges["created_at_utc"].cast(pa.timestamp("us", tz=None)),
    )
    issues = validate_edges_table(bad_edges)
    codes = {i.code for i in issues}
    assert ErrorCode.INVALID_EDGE_TYPE in codes
    assert ErrorCode.INVALID_UTC_TIMESTAMP in codes
