"""Data Contract & Parquet Schema Validator for SBi-FAGAT.

Defines the unified Arrow/Parquet schemas for propagation trees across both
PHEME (English) and Vietnamese social media datasets (`nodes.parquet` and
`edges.parquet`), enforces strict UTC timestamps, non-empty text, tree
structural integrity (unique root, valid parents, zero cycles), temporal
causality, and Zero Data Leakage invariants.

Core runtime dependencies are `pyarrow` and `pydantic` (no hard `pandas`
dependency; optional DataFrame inputs are supported if `pandas` is present).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


# ---------------------------------------------------------------------------
# Constants & Allowed Domain Values
# ---------------------------------------------------------------------------

ALLOWED_LANGUAGES: frozenset[str] = frozenset({"en", "vi"})
ALLOWED_EDGE_TYPES: frozenset[str] = frozenset({"reply", "quote", "repost"})
ALLOWED_RUMOR_LABELS: frozenset[str] = frozenset(
    {"true", "false", "unverified", "non-rumor", "rumor"}
)

# Columns containing post-hoc cascade labels must NEVER enter node/edge feature tables.
FORBIDDEN_LEAKAGE_COLUMNS: frozenset[str] = frozenset(
    {
        "y_size",
        "y_polar",
        "n_total",
        "final_cascade_size",
        "full_tree_polarization",
    }
)

_STRICT_UTC_ISO_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|z|[+-]00:00)$"
)


class ErrorCode(str, Enum):
    """Machine-readable validation error codes for schema and graph checks."""

    MISSING_COLUMN = "MISSING_COLUMN"
    INVALID_DTYPE = "INVALID_DTYPE"
    DATA_LEAKAGE_COLUMN = "DATA_LEAKAGE_COLUMN"
    NULL_VALUE = "NULL_VALUE"
    DUPLICATE_NODE_ID = "DUPLICATE_NODE_ID"
    DUPLICATE_EDGE_ID = "DUPLICATE_EDGE_ID"
    DUPLICATE_TARGET_PARENT = "DUPLICATE_TARGET_PARENT"
    EMPTY_TEXT = "EMPTY_TEXT"
    INVALID_UTC_TIMESTAMP = "INVALID_UTC_TIMESTAMP"
    MISSING_PARENT_NODE = "MISSING_PARENT_NODE"
    DANGLING_EDGE_ENDPOINT = "DANGLING_EDGE_ENDPOINT"
    CROSS_CASCADE_REFERENCE = "CROSS_CASCADE_REFERENCE"
    PARENT_EDGE_MISMATCH = "PARENT_EDGE_MISMATCH"
    MISSING_ROOT_NODE = "MISSING_ROOT_NODE"
    MULTIPLE_ROOT_NODES = "MULTIPLE_ROOT_NODES"
    ROOT_HAS_PARENT = "ROOT_HAS_PARENT"
    CYCLE_DETECTED = "CYCLE_DETECTED"
    TEMPORAL_ORDER_VIOLATION = "TEMPORAL_ORDER_VIOLATION"
    INVALID_LANGUAGE = "INVALID_LANGUAGE"
    INVALID_EDGE_TYPE = "INVALID_EDGE_TYPE"
    INVALID_RUMOR_LABEL = "INVALID_RUMOR_LABEL"


@dataclass(frozen=True)
class ValidationIssue:
    """Single validation violation discovered in node/edge tables or graph structure."""

    code: ErrorCode
    table: str
    message: str
    record_id: str | None = None


class SchemaValidationError(ValueError):
    """Raised when one or more data-contract violations are found."""

    def __init__(self, issues: Sequence[ValidationIssue]) -> None:
        self.issues: list[ValidationIssue] = list(issues)
        summary = "; ".join(
            f"[{issue.code.value}] ({issue.table}"
            + (f":{issue.record_id}" if issue.record_id else "")
            + f") {issue.message}"
            for issue in self.issues
        )
        super().__init__(f"Data contract validation failed ({len(self.issues)} issue(s)): {summary}")

    @property
    def error_codes(self) -> set[ErrorCode]:
        """Return the set of distinct ErrorCode values collected."""
        return {issue.code for issue in self.issues}


# ---------------------------------------------------------------------------
# Arrow Parquet Schemas
# ---------------------------------------------------------------------------

NODES_ARROW_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("node_id", pa.string(), nullable=False),
        pa.field("cascade_id", pa.string(), nullable=False),
        pa.field("topic_id", pa.string(), nullable=False),
        pa.field("parent_id", pa.string(), nullable=True),
        pa.field("is_root", pa.bool_(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
        pa.field("created_at_utc", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("lang", pa.string(), nullable=False),
        pa.field("source_dataset", pa.string(), nullable=False),
        pa.field("user_id_hash", pa.string(), nullable=True),
        pa.field("rumor_label", pa.string(), nullable=True),
    ]
)

EDGES_ARROW_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("edge_id", pa.string(), nullable=False),
        pa.field("cascade_id", pa.string(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("target_id", pa.string(), nullable=False),
        pa.field("created_at_utc", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("time_delay_sec", pa.float64(), nullable=False),
        pa.field("edge_type", pa.string(), nullable=False),
    ]
)

REQUIRED_NODE_COLUMNS: tuple[str, ...] = tuple(NODES_ARROW_SCHEMA.names)
REQUIRED_EDGE_COLUMNS: tuple[str, ...] = tuple(EDGES_ARROW_SCHEMA.names)
_NON_NULLABLE_NODE_COLUMNS: frozenset[str] = frozenset(
    f.name for f in NODES_ARROW_SCHEMA if not f.nullable
)
_NON_NULLABLE_EDGE_COLUMNS: frozenset[str] = frozenset(
    f.name for f in EDGES_ARROW_SCHEMA if not f.nullable
)


# ---------------------------------------------------------------------------
# Strict Parsing Helpers & Pydantic Record Models
# ---------------------------------------------------------------------------


def parse_strict_utc_timestamp(value: Any) -> datetime:
    """Parse and strictly validate that a timestamp is explicitly UTC-aware.

    Accepts:
      - timezone-aware `datetime` whose `utcoffset()` equals `timedelta(0)`
      - ISO-8601 datetime strings with an explicit UTC suffix (`Z`, `z`, `+00:00`, `-00:00`)

    Rejects naive `datetime` instances, non-UTC offsets (e.g. `+07:00`), date-only
    strings, timestamps without timezone info, booleans, and numeric values.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError(
                f"Datetime must be timezone-aware with UTC offset +00:00, got {value!r}"
            )
        return value.astimezone(timezone.utc)

    if isinstance(value, str):
        stripped = value.strip()
        if not _STRICT_UTC_ISO_RE.match(stripped):
            raise ValueError(
                f"Timestamp string must be ISO-8601 with explicit UTC suffix (Z/+00:00/-00:00), got {value!r}"
            )
        normalized = stripped[:-1] + "+00:00" if stripped[-1] in ("Z", "z") else stripped
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
            raise ValueError(f"Parsed timestamp is not UTC: {value!r}")
        return parsed.astimezone(timezone.utc)

    raise ValueError(f"Unsupported timestamp type {type(value).__name__}: {value!r}")


def _is_non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


class NodeRecord(BaseModel):
    """Pydantic specification for a single row in `nodes.parquet`."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    node_id: str
    cascade_id: str
    topic_id: str
    parent_id: str | None = None
    is_root: bool
    text: str
    created_at_utc: datetime
    lang: str
    source_dataset: str
    user_id_hash: str | None = None
    rumor_label: str | None = None

    @field_validator("node_id", "cascade_id", "topic_id", "source_dataset")
    @classmethod
    def _validate_non_empty_identifier(cls, value: str) -> str:
        if not _is_non_empty_str(value):
            raise ValueError("Identifier field must be a non-empty string")
        return value

    @field_validator("text")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        if not _is_non_empty_str(value):
            raise ValueError("Node text must not be empty or whitespace-only")
        return value

    @field_validator("created_at_utc", mode="before")
    @classmethod
    def _validate_utc(cls, value: Any) -> datetime:
        return parse_strict_utc_timestamp(value)

    @field_validator("lang")
    @classmethod
    def _validate_lang(cls, value: str) -> str:
        if value not in ALLOWED_LANGUAGES:
            raise ValueError(f"Unsupported language {value!r}; allowed: {sorted(ALLOWED_LANGUAGES)}")
        return value

    @field_validator("rumor_label")
    @classmethod
    def _validate_rumor_label(cls, value: str | None) -> str | None:
        if value is not None and value not in ALLOWED_RUMOR_LABELS:
            raise ValueError(
                f"Unsupported rumor_label {value!r}; allowed: {sorted(ALLOWED_RUMOR_LABELS)}"
            )
        return value

    @model_validator(mode="after")
    def _validate_root_parent_consistency(self) -> NodeRecord:
        if self.is_root and self.parent_id is not None:
            raise ValueError("Root node must have parent_id=None")
        if not self.is_root and not _is_non_empty_str(self.parent_id):
            raise ValueError("Non-root node must specify a non-empty parent_id")
        return self


class EdgeRecord(BaseModel):
    """Pydantic specification for a single row in `edges.parquet`."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    edge_id: str
    cascade_id: str
    source_id: str
    target_id: str
    created_at_utc: datetime
    time_delay_sec: float
    edge_type: str

    @field_validator("edge_id", "cascade_id", "source_id", "target_id")
    @classmethod
    def _validate_non_empty_identifier(cls, value: str) -> str:
        if not _is_non_empty_str(value):
            raise ValueError("Edge identifier field must be a non-empty string")
        return value

    @field_validator("created_at_utc", mode="before")
    @classmethod
    def _validate_utc(cls, value: Any) -> datetime:
        return parse_strict_utc_timestamp(value)

    @field_validator("time_delay_sec")
    @classmethod
    def _validate_delay(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"time_delay_sec must be finite and >= 0, got {value!r}")
        return float(value)

    @field_validator("edge_type")
    @classmethod
    def _validate_edge_type(cls, value: str) -> str:
        if value not in ALLOWED_EDGE_TYPES:
            raise ValueError(
                f"Unsupported edge_type {value!r}; allowed: {sorted(ALLOWED_EDGE_TYPES)}"
            )
        return value


# ---------------------------------------------------------------------------
# Record & Arrow Table Normalization / Validation
# ---------------------------------------------------------------------------


def _is_pandas_dataframe(obj: Any) -> bool:
    cls = type(obj)
    return cls.__name__ == "DataFrame" and cls.__module__.startswith("pandas")


def _coerce_input_to_table_or_records(
    data: pa.Table | Sequence[Mapping[str, Any]] | Any,
) -> tuple[pa.Table | None, list[dict[str, Any]]]:
    if isinstance(data, pa.Table):
        return data, data.to_pylist()
    if _is_pandas_dataframe(data):
        table = pa.Table.from_pandas(data, preserve_index=False)
        return table, table.to_pylist()
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        return None, [dict(row) for row in data]
    raise TypeError(
        f"Expected pyarrow.Table, sequence of mappings, or pandas.DataFrame; got {type(data).__name__}"
    )


def _check_leakage_columns(column_names: Iterable[str], table_name: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for col in sorted(set(column_names)):
        if col in FORBIDDEN_LEAKAGE_COLUMNS:
            issues.append(
                ValidationIssue(
                    code=ErrorCode.DATA_LEAKAGE_COLUMN,
                    table=table_name,
                    message=f"Forbidden target/leakage column '{col}' found in feature table",
                )
            )
    return issues


def _is_arrow_string_type(dtype: pa.DataType) -> bool:
    return pa.types.is_string(dtype) or pa.types.is_large_string(dtype)


def _validate_arrow_schema_structure(
    table: pa.Table,
    expected_schema: pa.Schema,
    table_name: str,
) -> list[ValidationIssue]:
    issues = _check_leakage_columns(table.column_names, table_name)
    actual_fields = {field.name: field for field in table.schema}

    for expected_field in expected_schema:
        name = expected_field.name
        if name not in actual_fields:
            issues.append(
                ValidationIssue(
                    code=ErrorCode.MISSING_COLUMN,
                    table=table_name,
                    message=f"Missing required column '{name}'",
                )
            )
            continue

        actual_type = actual_fields[name].type
        if name == "created_at_utc":
            if not (pa.types.is_timestamp(actual_type) and actual_type.tz == "UTC"):
                issues.append(
                    ValidationIssue(
                        code=ErrorCode.INVALID_UTC_TIMESTAMP,
                        table=table_name,
                        message=(
                            f"Column 'created_at_utc' must have Arrow type timestamp[*, tz=UTC], "
                            f"got {actual_type}"
                        ),
                    )
                )
        elif _is_arrow_string_type(expected_field.type):
            if not (_is_arrow_string_type(actual_type) or pa.types.is_null(actual_type)):
                issues.append(
                    ValidationIssue(
                        code=ErrorCode.INVALID_DTYPE,
                        table=table_name,
                        message=f"Column '{name}' expected string, got {actual_type}",
                    )
                )
        elif pa.types.is_boolean(expected_field.type):
            if not pa.types.is_boolean(actual_type):
                issues.append(
                    ValidationIssue(
                        code=ErrorCode.INVALID_DTYPE,
                        table=table_name,
                        message=f"Column '{name}' expected bool, got {actual_type}",
                    )
                )
        elif pa.types.is_floating(expected_field.type):
            if not (pa.types.is_floating(actual_type) or pa.types.is_integer(actual_type)):
                issues.append(
                    ValidationIssue(
                        code=ErrorCode.INVALID_DTYPE,
                        table=table_name,
                        message=f"Column '{name}' expected float64, got {actual_type}",
                    )
                )
    return issues


def _validate_node_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    validate_timestamps: bool,
) -> tuple[list[ValidationIssue], dict[int, datetime]]:
    issues: list[ValidationIssue] = []
    parsed_timestamps: dict[int, datetime] = {}

    if rows:
        all_keys: set[str] = set()
        for row in rows:
            all_keys.update(row.keys())
        issues.extend(_check_leakage_columns(all_keys, "nodes"))

    for idx, row in enumerate(rows):
        row_id = str(row.get("node_id") or f"row#{idx}")

        missing_cols = [col for col in REQUIRED_NODE_COLUMNS if col not in row]
        for col in missing_cols:
            issues.append(
                ValidationIssue(
                    code=ErrorCode.MISSING_COLUMN,
                    table="nodes",
                    message=f"Row missing required key '{col}'",
                    record_id=row_id,
                )
            )

        for col in _NON_NULLABLE_NODE_COLUMNS:
            if col in row and row[col] is None:
                code = ErrorCode.EMPTY_TEXT if col == "text" else ErrorCode.NULL_VALUE
                msg = (
                    "Node text must not be None or empty"
                    if col == "text"
                    else f"Non-nullable column '{col}' is None"
                )
                issues.append(
                    ValidationIssue(
                        code=code,
                        table="nodes",
                        message=msg,
                        record_id=row_id,
                    )
                )

        if "text" in row and row["text"] is not None:
            if not isinstance(row["text"], str):
                issues.append(
                    ValidationIssue(
                        code=ErrorCode.INVALID_DTYPE,
                        table="nodes",
                        message=f"Column 'text' must be str, got {type(row['text']).__name__}",
                        record_id=row_id,
                    )
                )
            elif row["text"].strip() == "":
                issues.append(
                    ValidationIssue(
                        code=ErrorCode.EMPTY_TEXT,
                        table="nodes",
                        message="Node text must not be empty or whitespace-only",
                        record_id=row_id,
                    )
                )

        for str_col in ("node_id", "cascade_id", "topic_id", "source_dataset"):
            val = row.get(str_col)
            if val is not None and not _is_non_empty_str(val):
                issues.append(
                    ValidationIssue(
                        code=ErrorCode.INVALID_DTYPE,
                        table="nodes",
                        message=f"Column '{str_col}' must be a non-empty string",
                        record_id=row_id,
                    )
                )

        is_root = row.get("is_root")
        if "is_root" in row and is_root is not None and not isinstance(is_root, bool):
            issues.append(
                ValidationIssue(
                    code=ErrorCode.INVALID_DTYPE,
                    table="nodes",
                    message=f"Column 'is_root' must be bool, got {type(is_root).__name__}",
                    record_id=row_id,
                )
            )

        if "lang" in row and row["lang"] is not None and row["lang"] not in ALLOWED_LANGUAGES:
            issues.append(
                ValidationIssue(
                    code=ErrorCode.INVALID_LANGUAGE,
                    table="nodes",
                    message=f"Invalid language {row['lang']!r}; allowed: {sorted(ALLOWED_LANGUAGES)}",
                    record_id=row_id,
                )
            )

        rumor_label = row.get("rumor_label")
        if rumor_label is not None and rumor_label not in ALLOWED_RUMOR_LABELS:
            issues.append(
                ValidationIssue(
                    code=ErrorCode.INVALID_RUMOR_LABEL,
                    table="nodes",
                    message=(
                        f"Invalid rumor_label {rumor_label!r}; allowed: {sorted(ALLOWED_RUMOR_LABELS)}"
                    ),
                    record_id=row_id,
                )
            )

        if "created_at_utc" in row and row["created_at_utc"] is not None:
            if validate_timestamps:
                try:
                    parsed_timestamps[idx] = parse_strict_utc_timestamp(row["created_at_utc"])
                except ValueError as exc:
                    issues.append(
                        ValidationIssue(
                            code=ErrorCode.INVALID_UTC_TIMESTAMP,
                            table="nodes",
                            message=str(exc),
                            record_id=row_id,
                        )
                    )
            elif isinstance(row["created_at_utc"], datetime):
                dt = row["created_at_utc"]
                if dt.tzinfo is not None and dt.utcoffset() == timedelta(0):
                    parsed_timestamps[idx] = dt.astimezone(timezone.utc)

    return issues, parsed_timestamps


def _validate_edge_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    validate_timestamps: bool,
) -> tuple[list[ValidationIssue], dict[int, datetime]]:
    issues: list[ValidationIssue] = []
    parsed_timestamps: dict[int, datetime] = {}

    if rows:
        all_keys: set[str] = set()
        for row in rows:
            all_keys.update(row.keys())
        issues.extend(_check_leakage_columns(all_keys, "edges"))

    for idx, row in enumerate(rows):
        row_id = str(row.get("edge_id") or f"row#{idx}")

        missing_cols = [col for col in REQUIRED_EDGE_COLUMNS if col not in row]
        for col in missing_cols:
            issues.append(
                ValidationIssue(
                    code=ErrorCode.MISSING_COLUMN,
                    table="edges",
                    message=f"Row missing required key '{col}'",
                    record_id=row_id,
                )
            )

        for col in _NON_NULLABLE_EDGE_COLUMNS:
            if col in row and row[col] is None:
                issues.append(
                    ValidationIssue(
                        code=ErrorCode.NULL_VALUE,
                        table="edges",
                        message=f"Non-nullable column '{col}' is None",
                        record_id=row_id,
                    )
                )

        for str_col in ("edge_id", "cascade_id", "source_id", "target_id"):
            val = row.get(str_col)
            if val is not None and not _is_non_empty_str(val):
                issues.append(
                    ValidationIssue(
                        code=ErrorCode.INVALID_DTYPE,
                        table="edges",
                        message=f"Column '{str_col}' must be a non-empty string",
                        record_id=row_id,
                    )
                )

        if (
            "edge_type" in row
            and row["edge_type"] is not None
            and row["edge_type"] not in ALLOWED_EDGE_TYPES
        ):
            issues.append(
                ValidationIssue(
                    code=ErrorCode.INVALID_EDGE_TYPE,
                    table="edges",
                    message=(
                        f"Invalid edge_type {row['edge_type']!r}; allowed: {sorted(ALLOWED_EDGE_TYPES)}"
                    ),
                    record_id=row_id,
                )
            )

        if "time_delay_sec" in row and row["time_delay_sec"] is not None:
            delay = row["time_delay_sec"]
            if isinstance(delay, bool) or not isinstance(delay, (int, float)):
                issues.append(
                    ValidationIssue(
                        code=ErrorCode.INVALID_DTYPE,
                        table="edges",
                        message=f"Column 'time_delay_sec' must be numeric, got {type(delay).__name__}",
                        record_id=row_id,
                    )
                )
            elif not math.isfinite(float(delay)) or float(delay) < 0.0:
                issues.append(
                    ValidationIssue(
                        code=ErrorCode.TEMPORAL_ORDER_VIOLATION,
                        table="edges",
                        message=f"Edge time_delay_sec must be finite and >= 0, got {delay!r}",
                        record_id=row_id,
                    )
                )

        if "created_at_utc" in row and row["created_at_utc"] is not None:
            if validate_timestamps:
                try:
                    parsed_timestamps[idx] = parse_strict_utc_timestamp(row["created_at_utc"])
                except ValueError as exc:
                    issues.append(
                        ValidationIssue(
                            code=ErrorCode.INVALID_UTC_TIMESTAMP,
                            table="edges",
                            message=str(exc),
                            record_id=row_id,
                        )
                    )
            elif isinstance(row["created_at_utc"], datetime):
                dt = row["created_at_utc"]
                if dt.tzinfo is not None and dt.utcoffset() == timedelta(0):
                    parsed_timestamps[idx] = dt.astimezone(timezone.utc)

    return issues, parsed_timestamps


def _detect_directed_cycles(
    vertices: Iterable[str],
    directed_edges: Iterable[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Detect directed cycles across all vertices using 3-color DFS.

    Starts from every unvisited vertex (not only the root) so disconnected
    cyclic components are guaranteed to be caught.
    """
    adj: dict[str, list[str]] = {v: [] for v in vertices}
    seen_pairs: set[tuple[str, str]] = set()
    for src, tgt in directed_edges:
        adj.setdefault(src, [])
        adj.setdefault(tgt, [])
        if (src, tgt) not in seen_pairs:
            adj[src].append(tgt)
            seen_pairs.add((src, tgt))

    white, gray, black = 0, 1, 2
    color: dict[str, int] = {v: white for v in adj}
    cycle_edges: list[tuple[str, str]] = []

    for start in sorted(adj):
        if color[start] != white:
            continue
        stack: list[tuple[str, int]] = [(start, 0)]
        color[start] = gray

        while stack:
            node, next_idx = stack[-1]
            neighbors = adj[node]
            if next_idx < len(neighbors):
                stack[-1] = (node, next_idx + 1)
                nbr = neighbors[next_idx]
                nbr_color = color[nbr]
                if nbr_color == gray:
                    cycle_edges.append((node, nbr))
                elif nbr_color == white:
                    color[nbr] = gray
                    stack.append((nbr, 0))
            else:
                color[node] = black
                stack.pop()

    return cycle_edges


# ---------------------------------------------------------------------------
# Public Table Builders & Validators
# ---------------------------------------------------------------------------


def build_nodes_table(records: Sequence[Mapping[str, Any]]) -> pa.Table:
    """Validate node dictionaries strictly and construct a typed `nodes.parquet` Arrow table."""
    issues, parsed_ts = _validate_node_rows(records, validate_timestamps=True)
    if issues:
        raise SchemaValidationError(issues)

    normalized: list[dict[str, Any]] = []
    for idx, row in enumerate(records):
        item = dict(row)
        item["created_at_utc"] = parsed_ts[idx]
        normalized.append(item)

    if not normalized:
        return NODES_ARROW_SCHEMA.empty_table()
    return pa.Table.from_pylist(normalized, schema=NODES_ARROW_SCHEMA)


def build_edges_table(records: Sequence[Mapping[str, Any]]) -> pa.Table:
    """Validate edge dictionaries strictly and construct a typed `edges.parquet` Arrow table."""
    issues, parsed_ts = _validate_edge_rows(records, validate_timestamps=True)
    if issues:
        raise SchemaValidationError(issues)

    normalized: list[dict[str, Any]] = []
    for idx, row in enumerate(records):
        item = dict(row)
        item["created_at_utc"] = parsed_ts[idx]
        item["time_delay_sec"] = float(item["time_delay_sec"])
        normalized.append(item)

    if not normalized:
        return EDGES_ARROW_SCHEMA.empty_table()
    return pa.Table.from_pylist(normalized, schema=EDGES_ARROW_SCHEMA)


def validate_nodes_table(table: pa.Table) -> list[ValidationIssue]:
    """Validate schema, Arrow types (including UTC tz), and row constraints on `nodes`."""
    issues = _validate_arrow_schema_structure(table, NODES_ARROW_SCHEMA, "nodes")
    ts_valid = not any(i.code == ErrorCode.INVALID_UTC_TIMESTAMP for i in issues)
    row_issues, _ = _validate_node_rows(table.to_pylist(), validate_timestamps=not ts_valid)
    issues.extend(row_issues)
    return issues


def validate_edges_table(table: pa.Table) -> list[ValidationIssue]:
    """Validate schema, Arrow types (including UTC tz), and row constraints on `edges`."""
    issues = _validate_arrow_schema_structure(table, EDGES_ARROW_SCHEMA, "edges")
    ts_valid = not any(i.code == ErrorCode.INVALID_UTC_TIMESTAMP for i in issues)
    row_issues, _ = _validate_edge_rows(table.to_pylist(), validate_timestamps=not ts_valid)
    issues.extend(row_issues)
    return issues


def validate_cascade_contract(
    nodes: pa.Table | Sequence[Mapping[str, Any]] | Any,
    edges: pa.Table | Sequence[Mapping[str, Any]] | Any,
    *,
    raise_on_error: bool = True,
) -> list[ValidationIssue]:
    """Run full Data Contract validation across `nodes` and `edges`.

    Verifies:
      1. Schema completeness, data types, and Zero Data Leakage forbidden columns.
      2. Strict UTC timestamps and non-empty text content.
      3. Unique node_id / edge_id and single-parent tree constraints.
      4. Unique root per cascade, valid parent references, and no cross-cascade edges.
      5. Absence of directed cycles (3-color DFS across both `parent_id` and `edges`).
      6. Temporal causality (`child.created_at_utc >= parent.created_at_utc` and `>= root`).
    """
    issues: list[ValidationIssue] = []

    nodes_table, node_rows = _coerce_input_to_table_or_records(nodes)
    edges_table, edge_rows = _coerce_input_to_table_or_records(edges)

    if nodes_table is not None:
        issues.extend(_validate_arrow_schema_structure(nodes_table, NODES_ARROW_SCHEMA, "nodes"))
        nodes_ts_ok = not any(
            i.table == "nodes" and i.code == ErrorCode.INVALID_UTC_TIMESTAMP for i in issues
        )
        node_row_issues, node_ts = _validate_node_rows(node_rows, validate_timestamps=not nodes_ts_ok)
    else:
        node_row_issues, node_ts = _validate_node_rows(node_rows, validate_timestamps=True)
    issues.extend(node_row_issues)

    if edges_table is not None:
        issues.extend(_validate_arrow_schema_structure(edges_table, EDGES_ARROW_SCHEMA, "edges"))
        edges_ts_ok = not any(
            i.table == "edges" and i.code == ErrorCode.INVALID_UTC_TIMESTAMP for i in issues
        )
        edge_row_issues, edge_ts = _validate_edge_rows(edge_rows, validate_timestamps=not edges_ts_ok)
    else:
        edge_row_issues, edge_ts = _validate_edge_rows(edge_rows, validate_timestamps=True)
    issues.extend(edge_row_issues)

    # Index nodes and check duplicates
    node_by_id: dict[str, dict[str, Any]] = {}
    node_dt_by_id: dict[str, datetime] = {}
    nodes_by_cascade: dict[str, list[dict[str, Any]]] = {}

    for idx, row in enumerate(node_rows):
        nid = row.get("node_id")
        cid = row.get("cascade_id")
        if _is_non_empty_str(nid):
            if nid in node_by_id:
                issues.append(
                    ValidationIssue(
                        code=ErrorCode.DUPLICATE_NODE_ID,
                        table="nodes",
                        message=f"Duplicate node_id '{nid}'",
                        record_id=nid,
                    )
                )
            else:
                node_by_id[nid] = row
                if idx in node_ts:
                    node_dt_by_id[nid] = node_ts[idx]
        if _is_non_empty_str(cid):
            nodes_by_cascade.setdefault(cid, []).append(row)

    # Index edges and check duplicates
    seen_edge_ids: set[str] = set()
    edges_by_cascade: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for idx, row in enumerate(edge_rows):
        eid = row.get("edge_id")
        cid = row.get("cascade_id")
        if _is_non_empty_str(eid):
            if eid in seen_edge_ids:
                issues.append(
                    ValidationIssue(
                        code=ErrorCode.DUPLICATE_EDGE_ID,
                        table="edges",
                        message=f"Duplicate edge_id '{eid}'",
                        record_id=eid,
                    )
                )
            seen_edge_ids.add(eid)
        if _is_non_empty_str(cid):
            edges_by_cascade.setdefault(cid, []).append((idx, row))

    all_cascades = sorted(set(nodes_by_cascade) | set(edges_by_cascade))
    for cid in all_cascades:
        c_nodes = nodes_by_cascade.get(cid, [])
        c_edges = edges_by_cascade.get(cid, [])

        # 1. Root integrity
        roots = [n for n in c_nodes if n.get("is_root") is True]
        if len(roots) == 0:
            issues.append(
                ValidationIssue(
                    code=ErrorCode.MISSING_ROOT_NODE,
                    table="nodes",
                    message=f"Cascade '{cid}' has no root node (is_root=True)",
                    record_id=cid,
                )
            )
        elif len(roots) > 1:
            issues.append(
                ValidationIssue(
                    code=ErrorCode.MULTIPLE_ROOT_NODES,
                    table="nodes",
                    message=f"Cascade '{cid}' has {len(roots)} root nodes; expected exactly 1",
                    record_id=cid,
                )
            )

        root_ids = {r.get("node_id") for r in roots if _is_non_empty_str(r.get("node_id"))}
        root_dt = (
            node_dt_by_id.get(roots[0]["node_id"])
            if len(roots) == 1 and _is_non_empty_str(roots[0].get("node_id"))
            else None
        )

        # 2. Parent integrity per node
        expected_parent_edges: set[tuple[str, str]] = set()
        cascade_vertices: set[str] = set()
        cascade_graph_edges: list[tuple[str, str]] = []

        for n in c_nodes:
            nid = n.get("node_id")
            pid = n.get("parent_id")
            is_root = n.get("is_root")
            if _is_non_empty_str(nid):
                cascade_vertices.add(nid)

            if is_root is True:
                if pid is not None:
                    issues.append(
                        ValidationIssue(
                            code=ErrorCode.ROOT_HAS_PARENT,
                            table="nodes",
                            message=f"Root node '{nid}' must have parent_id=None, got {pid!r}",
                            record_id=str(nid),
                        )
                    )
            elif is_root is False:
                if not _is_non_empty_str(pid):
                    issues.append(
                        ValidationIssue(
                            code=ErrorCode.MISSING_PARENT_NODE,
                            table="nodes",
                            message=f"Non-root node '{nid}' is missing parent_id",
                            record_id=str(nid),
                        )
                    )
                elif pid not in node_by_id:
                    issues.append(
                        ValidationIssue(
                            code=ErrorCode.MISSING_PARENT_NODE,
                            table="nodes",
                            message=f"Node '{nid}' references missing parent_id '{pid}'",
                            record_id=str(nid),
                        )
                    )
                else:
                    parent_row = node_by_id[pid]
                    if parent_row.get("cascade_id") != cid:
                        issues.append(
                            ValidationIssue(
                                code=ErrorCode.CROSS_CASCADE_REFERENCE,
                                table="nodes",
                                message=(
                                    f"Node '{nid}' in cascade '{cid}' references parent '{pid}' "
                                    f"in cascade '{parent_row.get('cascade_id')}'"
                                ),
                                record_id=str(nid),
                            )
                        )
                    elif _is_non_empty_str(nid):
                        expected_parent_edges.add((pid, nid))
                        cascade_graph_edges.append((pid, nid))

                        # Temporal causality against parent
                        if nid in node_dt_by_id and pid in node_dt_by_id:
                            if node_dt_by_id[nid] < node_dt_by_id[pid]:
                                issues.append(
                                    ValidationIssue(
                                        code=ErrorCode.TEMPORAL_ORDER_VIOLATION,
                                        table="nodes",
                                        message=(
                                            f"Node '{nid}' timestamp ({node_dt_by_id[nid].isoformat()}) "
                                            f"precedes parent '{pid}' ({node_dt_by_id[pid].isoformat()})"
                                        ),
                                        record_id=str(nid),
                                    )
                                )

            if root_dt is not None and _is_non_empty_str(nid) and nid in node_dt_by_id:
                if node_dt_by_id[nid] < root_dt:
                    issues.append(
                        ValidationIssue(
                            code=ErrorCode.TEMPORAL_ORDER_VIOLATION,
                            table="nodes",
                            message=(
                                f"Node '{nid}' timestamp ({node_dt_by_id[nid].isoformat()}) "
                                f"precedes cascade root ({root_dt.isoformat()})"
                            ),
                            record_id=str(nid),
                        )
                    )

        # 3. Edge endpoint integrity, target uniqueness, and temporal checks
        actual_edges: set[tuple[str, str]] = set()
        seen_targets: set[str] = set()

        for edge_idx, e in c_edges:
            eid = str(e.get("edge_id") or f"row#{edge_idx}")
            src = e.get("source_id")
            tgt = e.get("target_id")

            src_valid = _is_non_empty_str(src) and src in node_by_id
            tgt_valid = _is_non_empty_str(tgt) and tgt in node_by_id

            if not src_valid:
                issues.append(
                    ValidationIssue(
                        code=ErrorCode.DANGLING_EDGE_ENDPOINT,
                        table="edges",
                        message=f"Edge '{eid}' references missing source_id '{src}'",
                        record_id=eid,
                    )
                )
            if not tgt_valid:
                issues.append(
                    ValidationIssue(
                        code=ErrorCode.DANGLING_EDGE_ENDPOINT,
                        table="edges",
                        message=f"Edge '{eid}' references missing target_id '{tgt}'",
                        record_id=eid,
                    )
                )

            if src_valid and node_by_id[src].get("cascade_id") != cid:
                issues.append(
                    ValidationIssue(
                        code=ErrorCode.CROSS_CASCADE_REFERENCE,
                        table="edges",
                        message=(
                            f"Edge '{eid}' in cascade '{cid}' references source '{src}' "
                            f"from cascade '{node_by_id[src].get('cascade_id')}'"
                        ),
                        record_id=eid,
                    )
                )
            if tgt_valid and node_by_id[tgt].get("cascade_id") != cid:
                issues.append(
                    ValidationIssue(
                        code=ErrorCode.CROSS_CASCADE_REFERENCE,
                        table="edges",
                        message=(
                            f"Edge '{eid}' in cascade '{cid}' references target '{tgt}' "
                            f"from cascade '{node_by_id[tgt].get('cascade_id')}'"
                        ),
                        record_id=eid,
                    )
                )

            if _is_non_empty_str(tgt):
                if tgt in root_ids:
                    issues.append(
                        ValidationIssue(
                            code=ErrorCode.ROOT_HAS_PARENT,
                            table="edges",
                            message=f"Edge '{eid}' targets root node '{tgt}'",
                            record_id=eid,
                        )
                    )
                if tgt in seen_targets:
                    issues.append(
                        ValidationIssue(
                            code=ErrorCode.DUPLICATE_TARGET_PARENT,
                            table="edges",
                            message=f"Target node '{tgt}' has multiple incoming edges in cascade '{cid}'",
                            record_id=eid,
                        )
                    )
                seen_targets.add(tgt)

            if _is_non_empty_str(src) and _is_non_empty_str(tgt):
                cascade_vertices.add(src)
                cascade_vertices.add(tgt)
                cascade_graph_edges.append((src, tgt))
                if (
                    src_valid
                    and tgt_valid
                    and node_by_id[src].get("cascade_id") == cid
                    and node_by_id[tgt].get("cascade_id") == cid
                ):
                    actual_edges.add((src, tgt))

                # Edge temporal check against source and root
                e_dt = edge_ts.get(edge_idx)
                if src in node_dt_by_id and tgt in node_dt_by_id:
                    if node_dt_by_id[tgt] < node_dt_by_id[src]:
                        issues.append(
                            ValidationIssue(
                                code=ErrorCode.TEMPORAL_ORDER_VIOLATION,
                                table="edges",
                                message=(
                                    f"Edge '{eid}': target '{tgt}' precedes source '{src}'"
                                ),
                                record_id=eid,
                            )
                        )
                if e_dt is not None and src in node_dt_by_id and e_dt < node_dt_by_id[src]:
                    issues.append(
                        ValidationIssue(
                            code=ErrorCode.TEMPORAL_ORDER_VIOLATION,
                            table="edges",
                            message=f"Edge '{eid}' timestamp precedes source '{src}'",
                            record_id=eid,
                        )
                    )
                if e_dt is not None and root_dt is not None and e_dt < root_dt:
                    issues.append(
                        ValidationIssue(
                            code=ErrorCode.TEMPORAL_ORDER_VIOLATION,
                            table="edges",
                            message=f"Edge '{eid}' timestamp precedes cascade root",
                            record_id=eid,
                        )
                    )

        # 4. Parent-edge correspondence check
        if expected_parent_edges != actual_edges:
            missing_in_edges = sorted(expected_parent_edges - actual_edges)
            extra_in_edges = sorted(actual_edges - expected_parent_edges)
            issues.append(
                ValidationIssue(
                    code=ErrorCode.PARENT_EDGE_MISMATCH,
                    table="edges",
                    message=(
                        f"Cascade '{cid}' parent-child links mismatch edges table "
                        f"(missing_edges={missing_in_edges}, extra_edges={extra_in_edges})"
                    ),
                    record_id=cid,
                )
            )

        # 5. Cycle detection (3-color DFS across union of parent links and edges)
        cycle_edges = _detect_directed_cycles(cascade_vertices, cascade_graph_edges)
        for src, tgt in cycle_edges:
            issues.append(
                ValidationIssue(
                    code=ErrorCode.CYCLE_DETECTED,
                    table="edges",
                    message=f"Directed cycle detected in cascade '{cid}' involving edge {src} -> {tgt}",
                    record_id=f"{src}->{tgt}",
                )
            )

    if raise_on_error and issues:
        raise SchemaValidationError(issues)
    return issues


# ---------------------------------------------------------------------------
# Parquet Persistence Helpers
# ---------------------------------------------------------------------------


def save_cascade_parquet(
    nodes_table: pa.Table,
    edges_table: pa.Table,
    output_dir: str | Path,
    *,
    validate: bool = True,
) -> tuple[Path, Path]:
    """Validate (optional) and write `nodes.parquet` and `edges.parquet` to `output_dir`."""
    if validate:
        validate_cascade_contract(nodes_table, edges_table, raise_on_error=True)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    nodes_path = out_path / "nodes.parquet"
    edges_path = out_path / "edges.parquet"

    pq.write_table(nodes_table, nodes_path, compression="zstd")
    pq.write_table(edges_table, edges_path, compression="zstd")
    return nodes_path, edges_path


def load_cascade_parquet(
    input_dir: str | Path,
    *,
    validate: bool = True,
) -> tuple[pa.Table, pa.Table]:
    """Read `nodes.parquet` and `edges.parquet` from `input_dir` and optionally validate."""
    in_path = Path(input_dir)
    nodes_table = pq.read_table(in_path / "nodes.parquet")
    edges_table = pq.read_table(in_path / "edges.parquet")
    if validate:
        validate_cascade_contract(nodes_table, edges_table, raise_on_error=True)
    return nodes_table, edges_table
