"""Leakage-Free PyTorch Geometric Dataset & Inductive Topic Splitter for SBi-FAGAT.

Reads validated `nodes.parquet`, `edges.parquet`, and precomputed offline NLP
features (`nlp_features.pt`) from `data/processed/`, truncates every cascade
strictly to its first `K` chronological interactions (default `K = 20`, also
supporting `{10, 20, 30, 50}`), constructs independent Top-Down and Bottom-Up
directed edge lists with 2D edge attributes `[s_uv, h_uv]`, and enforces
inductive 70/15/15 topic-disjoint splitting.
"""

from __future__ import annotations

import math
import random
from collections import deque
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from src.data.schema import load_cascade_parquet, validate_cascade_contract

SUPPORTED_HORIZONS: tuple[int, ...] = (10, 20, 30, 50)
TEMPORAL_REF_SECONDS: float = 7.0 * 86400.0  # Fixed causal 7-day logarithmic reference

RUMOR_LABEL_TO_IDX: dict[str, int] = {
    "non-rumor": 0,
    "true": 1,
    "false": 2,
    "unverified": 3,
    "rumor": 3,
}

BINARY_RUMOR_LABEL_TO_IDX: dict[str, int] = {
    "non-rumor": 0,
    "true": 1,
    "false": 1,
    "unverified": 1,
    "rumor": 1,
}

NLP_NODE_DIM: int = 768
TEMPORAL_DEPTH_DIM: int = 2
TOTAL_NODE_FEATURE_DIM: int = NLP_NODE_DIM + TEMPORAL_DEPTH_DIM  # 770
CASCADE_STATS_DIM: int = 8


class CascadeData(Data):
    """PyG `Data` subclass with proper batch offsets for TD/BU edges and root nodes."""

    def __inc__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key in {"edge_index", "edge_index_td", "edge_index_bu", "root_index"}:
            return int(self.num_nodes)
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key in {"edge_index", "edge_index_td", "edge_index_bu"}:
            return 1
        if key == "root_index":
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


def _get_topic_id(item: Any) -> str:
    if isinstance(item, Mapping):
        return str(item["topic_id"])
    return str(getattr(item, "topic_id"))


def _get_item_rumor_label(item: Any) -> int | None:
    raw = item.get("y_rumor") if isinstance(item, Mapping) else getattr(item, "y_rumor", None)
    if raw is None:
        return None
    if isinstance(raw, torch.Tensor):
        val = int(raw.view(-1)[0].item()) if raw.numel() > 0 else -1
    else:
        val = int(raw)
    return val if val >= 0 else None


def _partition_topics_greedy(
    ordered_topics: Sequence[str],
    topic_to_indices: Mapping[str, list[int]],
    total_items: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[str], list[str], list[str]]:
    target_train = total_items * train_ratio
    target_val = total_items * val_ratio

    train_topics: list[str] = []
    val_topics: list[str] = []
    test_topics: list[str] = []
    train_count = 0
    val_count = 0

    idx = 0
    while idx < len(ordered_topics) - 2:
        t = ordered_topics[idx]
        t_size = len(topic_to_indices[t])
        if len(train_topics) == 0 or (train_count + t_size <= target_train) or (
            abs((train_count + t_size) - target_train) < abs(train_count - target_train)
            and (total_items - (train_count + t_size)) >= 2
        ):
            train_topics.append(t)
            train_count += t_size
            idx += 1
        else:
            break

    while idx < len(ordered_topics) - 1:
        t = ordered_topics[idx]
        t_size = len(topic_to_indices[t])
        if len(val_topics) == 0 or (val_count + t_size <= target_val) or (
            abs((val_count + t_size) - target_val) < abs(val_count - target_val)
        ):
            val_topics.append(t)
            val_count += t_size
            idx += 1
        else:
            break

    test_topics.extend(ordered_topics[idx:])
    return train_topics, val_topics, test_topics


def inductive_topic_split(
    items: Sequence[Any],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    num_trials: int = 64,
) -> dict[str, list[int]]:
    """Split cascade indices into topic-disjoint 70/15/15 Train/Val/Test subsets.

    All cascades sharing the same `topic_id` are assigned atomically to the same
    split so zero topic overlap exists between Train, Val, and Test. Uses multiple
    deterministic topic permutations to optimize size and label coverage balance.
    """
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, abs_tol=1e-6):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")
    if len(items) == 0:
        raise ValueError("Cannot split an empty collection of cascades")

    topic_to_indices: dict[str, list[int]] = {}
    for idx, item in enumerate(items):
        topic = _get_topic_id(item)
        topic_to_indices.setdefault(topic, []).append(idx)

    topics = sorted(topic_to_indices.keys())
    if len(topics) < 3:
        raise ValueError(
            f"Inductive topic split requires at least 3 distinct topics; found {len(topics)}"
        )

    total_items = len(items)
    item_labels = [_get_item_rumor_label(item) for item in items]
    present_labels = {lbl for lbl in item_labels if lbl is not None}

    best_score: tuple[float, float] | None = None
    best_partition: tuple[list[str], list[str], list[str]] | None = None

    for trial in range(max(1, int(num_trials))):
        trial_topics = list(topics)
        rng = random.Random(seed + trial * 1009)
        rng.shuffle(trial_topics)
        tr_t, va_t, te_t = _partition_topics_greedy(
            trial_topics, topic_to_indices, total_items, train_ratio, val_ratio
        )
        tr_idx = [i for t in tr_t for i in topic_to_indices[t]]
        va_idx = [i for t in va_t for i in topic_to_indices[t]]
        te_idx = [i for t in te_t for i in topic_to_indices[t]]

        size_err = (
            abs(len(tr_idx) / total_items - train_ratio)
            + abs(len(va_idx) / total_items - val_ratio)
            + abs(len(te_idx) / total_items - test_ratio)
        )
        missing_cls_penalty = 0.0
        if present_labels:
            for split_indices in (tr_idx, va_idx, te_idx):
                split_cls = {item_labels[i] for i in split_indices if item_labels[i] is not None}
                missing_cls_penalty += float(len(present_labels - split_cls))
        score = (missing_cls_penalty, size_err)
        if best_score is None or score < best_score:
            best_score = score
            best_partition = (tr_t, va_t, te_t)

    assert best_partition is not None
    train_topics, val_topics, test_topics = best_partition

    # Post-condition assertions for Zero Topic Leakage
    train_set, val_set, test_set = set(train_topics), set(val_topics), set(test_topics)
    assert train_set and val_set and test_set, "All three splits must contain >= 1 topic"
    assert train_set.isdisjoint(val_set), "Topic leakage detected between train and val"
    assert train_set.isdisjoint(test_set), "Topic leakage detected between train and test"
    assert val_set.isdisjoint(test_set), "Topic leakage detected between val and test"

    return {
        "train": [i for t in train_topics for i in topic_to_indices[t]],
        "val": [i for t in val_topics for i in topic_to_indices[t]],
        "test": [i for t in test_topics for i in topic_to_indices[t]],
    }


def leave_one_event_out_splits(
    items: Sequence[Any],
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Build Leave-One-Event-Out (LOO) inductive splits grouped by `topic_id`.

    For each held-out `test` topic, selects one remaining `val` topic and puts
    all other topics in `train`. Requires at least 3 distinct topics.
    """
    if len(items) == 0:
        raise ValueError("Cannot build LOO splits for empty items")

    topic_counts: dict[str, int] = {}
    for item in items:
        t = _get_topic_id(item)
        topic_counts[t] = topic_counts.get(t, 0) + 1

    topics = sorted(topic_counts.keys())
    if len(topics) < 3:
        raise ValueError(
            f"Leave-One-Event-Out requires at least 3 distinct topics; found {len(topics)}"
        )

    rng = random.Random(seed)
    ordered = sorted(topics)
    rng.shuffle(ordered)

    folds: list[dict[str, Any]] = []
    for idx, test_topic in enumerate(ordered):
        remaining = [t for t in ordered if t != test_topic]
        val_topic = ordered[(idx + 1) % len(ordered)]
        if val_topic == test_topic:
            val_topic = remaining[0]
        train_topics = [t for t in remaining if t != val_topic]
        folds.append(
            {
                "fold_name": test_topic,
                "train": set(train_topics),
                "val": {val_topic},
                "test": {test_topic},
            }
        )
    return folds


def _resolve_node_embeddings(
    nlp_payload: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    raw_emb = nlp_payload["node_embeddings"]
    if isinstance(raw_emb, Mapping):
        return {
            str(k): torch.as_tensor(v, dtype=torch.float32).view(-1)
            for k, v in raw_emb.items()
        }
    if "node_id_to_idx" in nlp_payload and isinstance(raw_emb, torch.Tensor):
        mapping = nlp_payload["node_id_to_idx"]
        return {
            str(nid): raw_emb[int(idx)].to(dtype=torch.float32).view(-1)
            for nid, idx in mapping.items()
        }
    raise ValueError(
        "nlp_features.pt['node_embeddings'] must be a dict[node_id -> Tensor[768]] "
        "or paired with 'node_id_to_idx'"
    )


def _lookup_edge_scalar(
    mapping: Mapping[Any, Any] | None,
    edge_id: str,
    src_id: str,
    dst_id: str,
    default: float = 0.0,
) -> float:
    if mapping is None:
        return default
    if edge_id in mapping:
        val = mapping[edge_id]
        return float(val.item() if isinstance(val, torch.Tensor) else val)
    pair_key = (src_id, dst_id)
    if pair_key in mapping:
        val = mapping[pair_key]
        return float(val.item() if isinstance(val, torch.Tensor) else val)
    str_pair = f"{src_id}->{dst_id}"
    if str_pair in mapping:
        val = mapping[str_pair]
        return float(val.item() if isinstance(val, torch.Tensor) else val)
    return default


def _lookup_edge_oppose_flag(
    nlp_payload: Mapping[str, Any],
    edge_id: str,
    src_id: str,
    dst_id: str,
    stance_val: float,
) -> bool:
    oppose_map = nlp_payload.get("edge_oppose_flag")
    if isinstance(oppose_map, Mapping):
        if edge_id in oppose_map:
            return bool(oppose_map[edge_id])
        if (src_id, dst_id) in oppose_map:
            return bool(oppose_map[(src_id, dst_id)])
    label_map = nlp_payload.get("edge_pred_label")
    if isinstance(label_map, Mapping):
        if edge_id in label_map:
            return str(label_map[edge_id]).lower() == "oppose"
        if (src_id, dst_id) in label_map:
            return str(label_map[(src_id, dst_id)]).lower() == "oppose"
    return stance_val < -0.1


def _compute_root_aligned_stances(
    root_id: str,
    children_adj: Mapping[str, Sequence[str]],
    edge_lookup: Mapping[tuple[str, str], Mapping[str, Any]],
    stance_map: Mapping[Any, Any] | None,
    *,
    root_stance_map: Mapping[Any, Any] | None = None,
    excluded_pairs: set[tuple[str, str]] | None = None,
    neutral_eps: float = 0.1,
) -> list[float]:
    """Propagate local parent-child stances into root-aligned orientations via BFS.

    When direct `(root, reply)` NLI stance scores (`root_stance_map`) are provided,
    they are used directly for any reply `v` present in `root_stance_map`, eliminating
    multi-hop sign-product drift on deep replies while falling back to signed parent
    propagation (`parent_sign * s_uv`) when `root_stance_map` is absent.
    """
    node_orientations: dict[str, float] = {root_id: 1.0}
    aligned_stances: list[float] = []
    queue: deque[str] = deque([root_id])

    while queue:
        u = queue.popleft()
        u_ori = node_orientations.get(u, 1.0)
        parent_sign = 1.0 if abs(u_ori) <= neutral_eps else (1.0 if u_ori > 0.0 else -1.0)
        for v in children_adj.get(u, []):
            edge_row = edge_lookup.get((u, v))
            eid = str(edge_row["edge_id"]) if edge_row is not None else f"{u}->{v}"
            s_uv = float(min(1.0, max(-1.0, _lookup_edge_scalar(stance_map, eid, u, v, 0.0))))
            if root_stance_map is not None and v in root_stance_map:
                aligned = float(min(1.0, max(-1.0, float(root_stance_map[v]))))
            else:
                aligned = float(min(1.0, max(-1.0, parent_sign * s_uv)))
            if v not in node_orientations:
                node_orientations[v] = aligned
                queue.append(v)
            if excluded_pairs is None or (u, v) not in excluded_pairs:
                aligned_stances.append(aligned)

    return aligned_stances


def compute_bimodality_coefficient(stances: Sequence[float]) -> float:
    """Compute Sarle's Bimodality Coefficient BC in [0, 1] (> 0.555 indicates bimodal split).

    Independent construct-validity companion to `y_polar` (Pfister et al., 2013).
    """
    if len(stances) < 4:
        return 0.0
    arr = np.clip(np.asarray(stances, dtype=np.float64), -1.0, 1.0)
    n = int(arr.size)
    diff = arr - float(np.mean(arr))
    m2 = float(np.mean(diff**2))
    if m2 <= 1e-12:
        return 0.0
    m3 = float(np.mean(diff**3))
    m4 = float(np.mean(diff**4))
    b1 = m3 / (m2**1.5)
    g1 = math.sqrt(n * (n - 1)) / float(n - 2) * b1
    g2_raw = m4 / (m2**2) - 3.0
    g2 = ((n - 1) / float((n - 2) * (n - 3))) * ((n + 1) * g2_raw + 6.0)
    finite_sample_norm = 3.0 * ((n - 1) ** 2) / float((n - 2) * (n - 3))
    denom = max(g2 + finite_sample_norm, g1**2 + 1.0)
    if denom <= 1e-12:
        return 0.0
    return float(min(1.0, max(0.0, (g1**2 + 1.0) / denom)))


def compute_signed_assortativity(
    edge_pairs: Sequence[tuple[str, str]],
    node_stance: Mapping[str, float],
    *,
    root_id: str | None = None,
) -> float:
    """Compute scalar Newman assortativity r in [-1, 1] of parent-child stances across edges."""
    u_vals: list[float] = []
    v_vals: list[float] = []
    for src, dst in edge_pairs:
        s_u = node_stance.get(src, 1.0 if (root_id is not None and src == root_id) else None)
        s_v = node_stance.get(dst)
        if s_u is None or s_v is None:
            continue
        u_vals.append(float(s_u))
        v_vals.append(float(s_v))
    if len(u_vals) < 2:
        return 0.0
    u_arr = np.asarray(u_vals, dtype=np.float64)
    v_arr = np.asarray(v_vals, dtype=np.float64)
    if float(np.std(u_arr)) <= 1e-12 or float(np.std(v_arr)) <= 1e-12:
        return 0.0
    r = float(np.corrcoef(u_arr, v_arr)[0, 1])
    if math.isnan(r):
        return 0.0
    return float(min(1.0, max(-1.0, r)))


def _compute_full_tree_polarization(
    stances: Sequence[float],
    oppose_flags: Sequence[bool] | None = None,
    lambda_polar: float = 0.5,
    neutral_eps: float = 0.1,
) -> float:
    """Compute symmetric bipolar full-tree polarization target y_polar in [0, 1].

    Combines normalized population variance sigma_s^2 (Popoviciu upper bound 1.0 on [-1, 1])
    with symmetric bipolar camp balance `4 * p_+ * p_-`, where p_+ = P(s > eps) and
    p_- = P(s < -eps). Unanimous support or unanimous opposition both yield 0.0, while an
    equal 50/50 split (+1 vs -1) attains the maximum 1.0.
    """
    del oppose_flags  # Retained in signature for backward compatibility
    if len(stances) == 0:
        return 0.0
    arr = np.asarray(stances, dtype=np.float64)
    arr = np.clip(arr, -1.0, 1.0)
    var_s = float(np.var(arr, ddof=0))
    pro_frac = float(np.mean(arr > neutral_eps))
    con_frac = float(np.mean(arr < -neutral_eps))
    bipolar_balance = 4.0 * pro_frac * con_frac
    y_polar = lambda_polar * var_s + (1.0 - lambda_polar) * bipolar_balance
    return float(min(1.0, max(0.0, y_polar)))


def compute_early_cascade_burst_stats(
    kept_nodes: Sequence[Mapping[str, Any]],
    delays_k: Sequence[float],
    depth_map: Mapping[str, int],
    edge_attr_rows: Sequence[Sequence[float]],
    *,
    k_horizon: int = 20,
    substantive_min_tokens: int = 8,
) -> torch.Tensor:
    """Compute leakage-free 8D `[1, 8]` early burst-velocity & structural descriptor strictly from V_K.

    Features (all bounded in [0, 1]):
      0. `elapsed_span_norm`: log-normalized time to reach the K-th reply
      1. `mean_interarrival_norm`: log-normalized mean gap between consecutive early replies
      2. `burst_acceleration`: 1 / (1 + exp(-(log1p(early_gap) - log1p(late_gap)))) (>0.5 means accelerating)
      3. `root_direct_ratio`: fraction of V_K replies directly replying to the root post
      4. `max_depth_norm`: maximum topological depth inside V_K divided by k_horizon
      5. `unique_user_ratio`: distinct reply author ratio (high -> broad crowd, low -> 2-user duel)
      6. `substantive_text_ratio`: fraction of V_K replies with >= 8 tokens (low -> empty chatter warning)
      7. `one_sided_hostility_index`: |mean(s_uv)| * mean(h_uv) catching early hostile echo chambers
    """
    if not kept_nodes or len(kept_nodes) <= 1:
        return torch.zeros((1, CASCADE_STATS_DIM), dtype=torch.float32)

    denom_t = math.log1p(TEMPORAL_REF_SECONDS)
    reply_nodes = kept_nodes[1:]
    root_id = str(kept_nodes[0]["node_id"])
    reply_delays = sorted(max(0.0, float(d)) for d in delays_k[1:])

    elapsed_span_norm = min(1.0, math.log1p(max(reply_delays)) / denom_t) if reply_delays else 0.0
    if len(reply_delays) >= 2:
        gaps = [max(0.0, reply_delays[i] - reply_delays[i - 1]) for i in range(1, len(reply_delays))]
        mean_gap = float(sum(gaps)) / float(len(gaps))
        mean_interarrival_norm = min(1.0, math.log1p(mean_gap) / denom_t)
        if len(gaps) >= 2:
            mid = len(gaps) // 2
            first_gap = float(sum(gaps[:mid])) / float(max(1, mid))
            second_gap = float(sum(gaps[mid:])) / float(max(1, len(gaps) - mid))
            diff_log = math.log1p(first_gap) - math.log1p(second_gap)
            diff_log = max(-20.0, min(20.0, diff_log))
            burst_acceleration = 1.0 / (1.0 + math.exp(-diff_log))
        else:
            burst_acceleration = 0.5
    else:
        mean_interarrival_norm = elapsed_span_norm
        burst_acceleration = 0.5

    root_direct_count = sum(1 for n in reply_nodes if str(n.get("parent_id")) == root_id)
    root_direct_ratio = float(root_direct_count) / float(len(reply_nodes))

    max_depth = max((int(depth_map.get(str(n["node_id"]), 0)) for n in reply_nodes), default=0)
    max_depth_norm = min(1.0, float(max_depth) / float(max(1, k_horizon)))

    hashed_users = [str(n["user_id_hash"]) for n in reply_nodes if n.get("user_id_hash")]
    if hashed_users:
        unique_user_ratio = min(1.0, float(len(set(hashed_users))) / float(len(reply_nodes)))
    else:
        unique_user_ratio = 1.0

    substantive_count = sum(
        1 for n in reply_nodes if len(str(n.get("text") or "").split()) >= substantive_min_tokens
    )
    substantive_text_ratio = float(substantive_count) / float(len(reply_nodes))

    if edge_attr_rows:
        s_vals = [float(r[0]) for r in edge_attr_rows]
        h_vals = [float(r[1]) for r in edge_attr_rows]
        mean_s = float(sum(s_vals)) / float(len(s_vals))
        mean_h = float(sum(h_vals)) / float(len(h_vals))
        one_sided_hostility_index = min(1.0, max(0.0, abs(mean_s) * mean_h))
    else:
        one_sided_hostility_index = 0.0

    vec = [
        elapsed_span_norm,
        mean_interarrival_norm,
        burst_acceleration,
        root_direct_ratio,
        max_depth_norm,
        unique_user_ratio,
        substantive_text_ratio,
        one_sided_hostility_index,
    ]
    return torch.tensor([vec], dtype=torch.float32)


class EarlyCascadeDataset(Dataset):
    """PyG Dataset building early-truncated (`K` replies) bidirectional signed graphs.

    Guarantees Zero Data Leakage:
      1. Labels `y_size`, `y_polar`, `y_rumor` are computed from G_full and stored ONLY
         as targets — never concatenated into `data.x` or `data.edge_attr`.
      2. Input node/edge tensors and intra-cascade temporal normalization (`t_norm`)
         are restricted strictly to `{root} U first_K_replies`.
      3. `split` in `{"train", "val", "test"}` partitions strictly by disjoint `topic_id`.
    """

    def __init__(
        self,
        root_dir: str | Path = "data/processed",
        k_horizon: int = 20,
        split: Literal["train", "val", "test", "all"] = "all",
        split_seed: int = 42,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        lambda_polar: float = 0.5,
        nodes_table: pa.Table | None = None,
        edges_table: pa.Table | None = None,
        nlp_features: Mapping[str, Any] | None = None,
        min_total_replies: int = 0,
        polar_target_mode: Literal["full", "future_only"] = "full",
        label_scheme: Literal["veracity4", "binary"] = "veracity4",
        custom_topic_splits: Mapping[str, set[str]] | None = None,
    ) -> None:
        super().__init__()
        if k_horizon < 1:
            raise ValueError(f"k_horizon must be >= 1, got {k_horizon}")
        if split not in {"train", "val", "test", "all"}:
            raise ValueError(f"Unsupported split {split!r}")
        if polar_target_mode not in {"full", "future_only"}:
            raise ValueError(f"Unsupported polar_target_mode {polar_target_mode!r}")
        if label_scheme not in {"veracity4", "binary"}:
            raise ValueError(f"Unsupported label_scheme {label_scheme!r}")

        self.root_dir = Path(root_dir)
        self.k_horizon = int(k_horizon)
        self.split = split
        self.split_seed = int(split_seed)
        self.lambda_polar = float(lambda_polar)
        self.min_total_replies = int(min_total_replies)
        self.polar_target_mode = polar_target_mode
        self.label_scheme = label_scheme

        if nodes_table is None or edges_table is None:
            nodes_table, edges_table = load_cascade_parquet(self.root_dir, validate=True)
        else:
            validate_cascade_contract(nodes_table, edges_table, raise_on_error=True)

        if nlp_features is None:
            feat_path = self.root_dir / "nlp_features.pt"
            if not feat_path.exists():
                raise FileNotFoundError(
                    f"Missing precomputed NLP features file: {feat_path}"
                )
            nlp_features = torch.load(feat_path, map_location="cpu", weights_only=False)

        self._all_cascades = self._build_all_cascades(
            nodes_table=nodes_table,
            edges_table=edges_table,
            nlp_features=nlp_features,
        )

        if split == "all":
            self.data_list: list[CascadeData] = list(self._all_cascades)
        elif custom_topic_splits is not None:
            allowed_topics = set(custom_topic_splits[split])
            self.data_list = [
                c for c in self._all_cascades if str(c.topic_id) in allowed_topics
            ]
        else:
            splits = inductive_topic_split(
                self._all_cascades,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=self.split_seed,
            )
            self.data_list = [self._all_cascades[i] for i in splits[split]]

    @property
    def topic_ids(self) -> set[str]:
        """Set of distinct topic_ids present in this split."""
        return {str(d.topic_id) for d in self.data_list}

    @property
    def cascade_ids(self) -> set[str]:
        """Set of distinct cascade_ids present in this split."""
        return {str(d.cascade_id) for d in self.data_list}

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> CascadeData:
        return self.data_list[idx]

    def _build_all_cascades(
        self,
        nodes_table: pa.Table,
        edges_table: pa.Table,
        nlp_features: Mapping[str, Any],
    ) -> list[CascadeData]:
        node_emb_map = _resolve_node_embeddings(nlp_features)
        stance_map = nlp_features.get("edge_stance")
        hostility_map = nlp_features.get("edge_hostility")
        irr_map = nlp_features.get("edge_irrationality")
        con_map = nlp_features.get("edge_constructive")
        root_stance_map = nlp_features.get("node_root_stance")

        raw_feat_edge_attr = nlp_features.get("edge_attr")
        use_4d_edges = bool(
            irr_map is not None
            or con_map is not None
            or int(nlp_features.get("edge_dim", 2)) == 4
            or (
                isinstance(raw_feat_edge_attr, torch.Tensor)
                and raw_feat_edge_attr.ndim == 2
                and raw_feat_edge_attr.size(1) == 4
            )
        )
        active_edge_dim = 4 if use_4d_edges else 2

        nodes_by_cascade: dict[str, list[dict[str, Any]]] = {}
        for row in nodes_table.to_pylist():
            nodes_by_cascade.setdefault(str(row["cascade_id"]), []).append(row)

        edges_by_cascade: dict[str, list[dict[str, Any]]] = {}
        for row in edges_table.to_pylist():
            edges_by_cascade.setdefault(str(row["cascade_id"]), []).append(row)

        cascades: list[CascadeData] = []
        for cid in sorted(nodes_by_cascade.keys()):
            c_nodes = nodes_by_cascade[cid]
            c_edges = edges_by_cascade.get(cid, [])

            # Locate unique root and build parent->children adjacency on G_full
            root_row = next(n for n in c_nodes if bool(n["is_root"]))
            root_id = str(root_row["node_id"])
            root_ts = root_row["created_at_utc"]
            topic_id = str(root_row["topic_id"])

            children_adj: dict[str, list[str]] = {str(n["node_id"]): [] for n in c_nodes}
            edge_lookup: dict[tuple[str, str], dict[str, Any]] = {}
            for e in c_edges:
                src = str(e["source_id"])
                dst = str(e["target_id"])
                children_adj.setdefault(src, []).append(dst)
                edge_lookup[(src, dst)] = e

            # Compute topological depths on G_full via BFS from root
            depth_map: dict[str, int] = {root_id: 0}
            queue: deque[str] = deque([root_id])
            while queue:
                curr = queue.popleft()
                for child in children_adj.get(curr, []):
                    if child not in depth_map:
                        depth_map[child] = depth_map[curr] + 1
                        queue.append(child)

            # 1. Early filter & post-hoc targets on G_full (NEVER leaked into node/edge features)
            n_total = len(c_nodes) - 1
            if n_total < self.min_total_replies:
                continue
            y_size_val = math.log1p(max(0, n_total))

            label_map = (
                BINARY_RUMOR_LABEL_TO_IDX
                if self.label_scheme == "binary"
                else RUMOR_LABEL_TO_IDX
            )
            raw_label = root_row.get("rumor_label")
            y_rumor_val = (
                label_map[str(raw_label)]
                if raw_label is not None and str(raw_label) in label_map
                else -1
            )

            # 2. Early chronological truncation to V_K = {root} U first K replies
            non_root_nodes = [n for n in c_nodes if not bool(n["is_root"])]
            non_root_nodes.sort(
                key=lambda n: (
                    n["created_at_utc"],
                    depth_map.get(str(n["node_id"]), 10**6),
                    str(n["node_id"]),
                )
            )
            kept_nodes = [root_row] + non_root_nodes[: self.k_horizon]
            kept_id_to_idx = {str(n["node_id"]): idx for idx, n in enumerate(kept_nodes)}

            kept_pair_set = {
                (str(n["parent_id"]), str(n["node_id"]))
                for n in kept_nodes[1:]
                if str(n["parent_id"]) in kept_id_to_idx
            }
            aligned_stances = _compute_root_aligned_stances(
                root_id=root_id,
                children_adj=children_adj,
                edge_lookup=edge_lookup,
                stance_map=stance_map,
                root_stance_map=root_stance_map,
                excluded_pairs=kept_pair_set if self.polar_target_mode == "future_only" else None,
            )
            y_polar_val = _compute_full_tree_polarization(
                aligned_stances,
                lambda_polar=self.lambda_polar,
            )
            y_bimodality_val = compute_bimodality_coefficient(aligned_stances)

            target_toxicity_scores: list[float] = []
            for e in c_edges:
                pair = (str(e["source_id"]), str(e["target_id"]))
                if self.polar_target_mode == "future_only" and pair in kept_pair_set:
                    continue
                eid_t = str(e["edge_id"])
                h_t = max(0.0, min(1.0, _lookup_edge_scalar(hostility_map, eid_t, pair[0], pair[1], 0.0)))
                r_t = max(0.0, min(1.0, _lookup_edge_scalar(irr_map, eid_t, pair[0], pair[1], 0.0)))
                target_toxicity_scores.append(0.6 * h_t + 0.4 * r_t)
            mean_tox = float(np.mean(target_toxicity_scores)) if target_toxicity_scores else 0.0
            y_toxic_polar_val = float(min(1.0, max(0.0, 0.5 * y_polar_val + 0.5 * mean_tox)))

            # 3. Leakage-free causal node features on V_K only (fixed TEMPORAL_REF_SECONDS horizon)
            delays_k = [
                max(0.0, (n["created_at_utc"] - root_ts).total_seconds())
                for n in kept_nodes
            ]
            denom_t = math.log1p(TEMPORAL_REF_SECONDS)

            node_feature_rows: list[torch.Tensor] = []
            kept_ids: list[str] = []
            kept_texts: list[str] = []
            for n, dt_sec in zip(kept_nodes, delays_k):
                nid = str(n["node_id"])
                if nid not in node_emb_map:
                    raise KeyError(f"Missing 768-d embedding for node_id '{nid}'")
                sem = node_emb_map[nid]
                if sem.numel() != NLP_NODE_DIM:
                    raise ValueError(
                        f"Expected {NLP_NODE_DIM}-d embedding for '{nid}', got {sem.numel()}"
                    )
                t_norm = min(1.0, math.log1p(dt_sec) / denom_t)
                d_norm = min(1.0, float(depth_map.get(nid, 0)) / float(self.k_horizon))
                meta_vec = torch.tensor([t_norm, d_norm], dtype=torch.float32)
                node_feature_rows.append(torch.cat([sem, meta_vec], dim=0))
                kept_ids.append(nid)
                kept_texts.append(str(n["text"]))

            x = torch.stack(node_feature_rows, dim=0)  # [N_K, 770]

            # 4. Bidirectional edges restricted to V_K
            td_src: list[int] = []
            td_dst: list[int] = []
            edge_attr_rows: list[list[float]] = []
            edge_stance_rows: list[list[float]] = []
            kept_edge_pairs: list[tuple[str, str]] = []

            for n in kept_nodes[1:]:
                dst_id = str(n["node_id"])
                src_id = str(n["parent_id"])
                if src_id not in kept_id_to_idx:
                    # Should never happen under causal timestamps, skip defensively
                    continue
                u_idx = kept_id_to_idx[src_id]
                v_idx = kept_id_to_idx[dst_id]
                edge_row = edge_lookup.get((src_id, dst_id))
                eid = str(edge_row["edge_id"]) if edge_row is not None else f"{src_id}->{dst_id}"

                s_uv = float(
                    min(1.0, max(-1.0, _lookup_edge_scalar(stance_map, eid, src_id, dst_id, 0.0)))
                )
                h_uv = float(
                    min(1.0, max(0.0, _lookup_edge_scalar(hostility_map, eid, src_id, dst_id, 0.0)))
                )

                td_src.append(u_idx)
                td_dst.append(v_idx)
                if use_4d_edges:
                    r_uv = float(
                        min(1.0, max(0.0, _lookup_edge_scalar(irr_map, eid, src_id, dst_id, 0.05)))
                    )
                    c_uv = float(
                        min(1.0, max(0.0, _lookup_edge_scalar(con_map, eid, src_id, dst_id, 0.50)))
                    )
                    edge_attr_rows.append([s_uv, h_uv, r_uv, c_uv])
                else:
                    edge_attr_rows.append([s_uv, h_uv])
                edge_stance_rows.append([s_uv])
                kept_edge_pairs.append((src_id, dst_id))

            if td_src:
                edge_index_td = torch.tensor([td_src, td_dst], dtype=torch.long)
                edge_index_bu = torch.tensor([td_dst, td_src], dtype=torch.long)
                edge_attr = torch.tensor(edge_attr_rows, dtype=torch.float32)
                edge_stance = torch.tensor(edge_stance_rows, dtype=torch.float32)
            else:
                edge_index_td = torch.empty((2, 0), dtype=torch.long)
                edge_index_bu = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, active_edge_dim), dtype=torch.float32)
                edge_stance = torch.empty((0, 1), dtype=torch.float32)

            cascade_stats = compute_early_cascade_burst_stats(
                kept_nodes=kept_nodes,
                delays_k=delays_k,
                depth_map=depth_map,
                edge_attr_rows=edge_attr_rows,
                k_horizon=self.k_horizon,
            )

            data = CascadeData(
                x=x,
                edge_index=edge_index_td,
                edge_index_td=edge_index_td,
                edge_index_bu=edge_index_bu,
                edge_attr=edge_attr,
                edge_stance=edge_stance,
                cascade_stats=cascade_stats,
                root_index=torch.tensor([0], dtype=torch.long),
                y_size=torch.tensor([y_size_val], dtype=torch.float32),
                y_polar=torch.tensor([y_polar_val], dtype=torch.float32),
                y_toxic_polar=torch.tensor([y_toxic_polar_val], dtype=torch.float32),
                y_bimodality=torch.tensor([y_bimodality_val], dtype=torch.float32),
                y_rumor=torch.tensor([y_rumor_val], dtype=torch.long),
                cascade_id=cid,
                topic_id=topic_id,
                k_horizon=self.k_horizon,
                node_ids=kept_ids,
                node_texts=kept_texts,
                edge_pairs=kept_edge_pairs,
            )
            cascades.append(data)

        return cascades


class UPFDAdapterTransform:
    """Adapt PyG `UPFD` graph samples into the bidirectional `CascadeData` contract.

    In PyG `UPFD`, node `0` is the source news post (`root_index = 0`), edges are
    directed outward/inward along the retweet/reply tree, and `x` holds precomputed
    768-d BERT node embeddings without raw text.
    """

    def __call__(self, data: Data) -> CascadeData:
        num_nodes = int(data.num_nodes)
        edge_index_td = data.edge_index.long()
        edge_index_bu = edge_index_td[[1, 0], :] if edge_index_td.numel() > 0 else edge_index_td
        y_cls = data.y.view(-1).long()
        y_size = torch.tensor([math.log1p(max(0, num_nodes - 1))], dtype=torch.float32)
        y_polar = torch.tensor([0.0], dtype=torch.float32)

        stats = torch.zeros((1, CASCADE_STATS_DIM), dtype=torch.float32)
        if edge_index_td.numel() > 0 and num_nodes > 1:
            root_direct = float((edge_index_td[0] == 0).sum().item()) / float(max(1, num_nodes - 1))
            stats[0, 3] = min(1.0, max(0.0, root_direct))

        return CascadeData(
            x=data.x.float(),
            edge_index=edge_index_td,
            edge_index_td=edge_index_td,
            edge_index_bu=edge_index_bu,
            edge_attr=torch.zeros((edge_index_td.size(1), 2), dtype=torch.float32),
            edge_stance=torch.zeros((edge_index_td.size(1), 1), dtype=torch.float32),
            cascade_stats=stats,
            root_index=torch.tensor([0], dtype=torch.long),
            y_size=y_size,
            y_polar=y_polar,
            y_toxic_polar=torch.tensor([0.0], dtype=torch.float32),
            y_bimodality=torch.tensor([0.0], dtype=torch.float32),
            y_rumor=y_cls,
            y=y_cls,
        )
