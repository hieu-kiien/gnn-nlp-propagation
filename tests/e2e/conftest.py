"""Shared fixtures, offline HF builders, synthetic corpus factories, and HTTP/disk helpers for 4-Tier E2E tests."""

from __future__ import annotations

import http.client
import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import (
    PreTrainedTokenizerFast,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaModel,
)

from scripts.demo_web_app import create_server
from src.data.pyg_dataset import (
    CascadeData,
    CASCADE_STATS_DIM,
    STRUCTURAL_NODE_DIM,
    TOTAL_NODE_FEATURE_DIM,
    save_topic_splits_json,
)
from src.data.schema import (
    EDGES_ARROW_SCHEMA,
    NODES_ARROW_SCHEMA,
    save_cascade_parquet,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_VN_DIR = REPO_ROOT / "data" / "processed" / "vn"
DATA_PHEME_DIR = REPO_ROOT / "data" / "processed" / "pheme"
INTERIM_DIR = REPO_ROOT / "data" / "interim"
WEB_DIR = REPO_ROOT / "web_demo"
PAPER_DIR = REPO_ROOT / "results"
CKPT_VN_PATH = REPO_ROOT / "checkpoints" / "best_sbifagat_vn.pt"
CKPT_PHEME_PATH = REPO_ROOT / "checkpoints" / "best_sbifagat_pheme.pt"
CKPT_PHEME_LOO_PATH = REPO_ROOT / "checkpoints" / "best_sbifagat_pheme_loo.pt"
CE_VN_DIR = REPO_ROOT / "checkpoints" / "dual_ce_vn"
CE_PHEME_DIR = REPO_ROOT / "checkpoints" / "dual_ce_pheme"
VIHSD_CKPT_DIR = REPO_ROOT / "checkpoints" / "visobert_vihsd"


def has_processed_corpus(data_dir: Path) -> bool:
    return (
        (data_dir / "nodes.parquet").is_file()
        and (data_dir / "edges.parquet").is_file()
        and (data_dir / "nlp_features.pt").is_file()
        and (data_dir / "topic_splits.json").is_file()
    )


def has_dual_ce_weights(ce_dir: Path) -> bool:
    return (ce_dir / "dual_head_weights.pt").is_file()


def has_paper_tier2_jsons(paper_dir: Path = PAPER_DIR) -> bool:
    return all(
        (paper_dir / name).is_file()
        for name in (
            "results_macro_network_vn.json",
            "results_macro_network_pheme.json",
            "results_signed_link_vn.json",
            "results_signed_link_pheme.json",
        )
    )


HAS_REAL_VN = has_processed_corpus(DATA_VN_DIR)
HAS_REAL_PHEME = has_processed_corpus(DATA_PHEME_DIR)
HAS_VN_CKPT = CKPT_VN_PATH.is_file()
HAS_PHEME_CKPT = CKPT_PHEME_PATH.is_file()
HAS_DUAL_CE_VN = has_dual_ce_weights(CE_VN_DIR)
HAS_DUAL_CE_PHEME = has_dual_ce_weights(CE_PHEME_DIR)
HAS_HUMAN_AUDIT_VN = (
    (INTERIM_DIR / "human_audit_vi_120.csv").is_file()
    and (INTERIM_DIR / "human_audit_vi_120_machine_key.csv").is_file()
    and (INTERIM_DIR / "human_audit_vn_30_trees.csv").is_file()
    and (INTERIM_DIR / "human_audit_vn_30_trees_machine_key.csv").is_file()
)
HAS_PAPER_TIER2_JSONS = has_paper_tier2_jsons()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "tier1: Isolated feature contract E2E checks (fast offline)")
    config.addinivalue_line("markers", "tier2: Boundary, corner, and negative rejection checks (fast offline)")
    config.addinivalue_line("markers", "tier3: Pairwise cross-feature integration & CLI subprocess workflows")
    config.addinivalue_line("markers", "tier4: Real-world application workflow scenarios")
    config.addinivalue_line("markers", "tier4_realdata: Tier 4 tests reading real disk datasets/checkpoints")


# ---------------------------------------------------------------------------
# Offline Tiny Hugging Face Model & Tokenizer Builders (Zero Network Calls)
# ---------------------------------------------------------------------------


def build_offline_roberta_tokenizer() -> PreTrainedTokenizerFast:
    vocab_tokens = [
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "root",
        "topic",
        "reply",
        "agree",
        "support",
        "good",
        "yes",
        "oppose",
        "hate",
        "trash",
        "wrong",
        "neutral",
        "question",
        "maybe",
        "chuan",
        "ung",
        "ho",
        "phan",
        "doi",
        "tuyet",
        "te",
        "nguoi",
        "dan",
        "the",
        "parent",
        "comment",
        "disagrees",
        "agrees",
        "targetreplytoken",
        "longword",
        "phat",
        "nguoi_xe",
        "bien",
        "mat",
    ]
    vocab = {tok: idx for idx, tok in enumerate(vocab_tokens)}
    tok_obj = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok_obj.pre_tokenizer = Whitespace()
    tok_obj.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> </s> $B </s>",
        special_tokens=[("<s>", 0), ("</s>", 2)],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tok_obj,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
    )


def build_tiny_roberta_config(
    *,
    vocab_size: int = 256,
    hidden_size: int = 32,
    num_hidden_layers: int = 1,
    num_attention_heads: int = 4,
    max_position_embeddings: int = 128,
    num_labels: int = 3,
    id2label: dict[int, str] | None = None,
) -> RobertaConfig:
    cfg = RobertaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=max(64, hidden_size * 2),
        max_position_embeddings=max_position_embeddings,
        type_vocab_size=1,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        num_labels=num_labels,
    )
    if id2label is not None:
        cfg.id2label = id2label
        cfg.label2id = {v: k for k, v in id2label.items()}
    return cfg


def build_offline_encoder_and_classifiers(
    *,
    hidden_size: int = 32,
    max_position_embeddings: int = 128,
    seed: int = 42,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    tokenizer = build_offline_roberta_tokenizer()
    base_cfg = build_tiny_roberta_config(
        vocab_size=len(tokenizer),
        hidden_size=hidden_size,
        max_position_embeddings=max_position_embeddings,
        num_labels=3,
    )
    encoder = RobertaModel(base_cfg, add_pooling_layer=False)

    vihsd_cfg = build_tiny_roberta_config(
        vocab_size=len(tokenizer),
        hidden_size=hidden_size,
        max_position_embeddings=max_position_embeddings,
        num_labels=3,
        id2label={0: "clean", 1: "offensive", 2: "hate"},
    )
    hostility_model = RobertaForSequenceClassification(vihsd_cfg)

    nli_cfg = build_tiny_roberta_config(
        vocab_size=len(tokenizer),
        hidden_size=hidden_size,
        max_position_embeddings=max_position_embeddings,
        num_labels=3,
        id2label={0: "contradiction", 1: "neutral", 2: "entailment"},
    )
    nli_model = RobertaForSequenceClassification(nli_cfg)

    return {
        "tokenizer": tokenizer,
        "encoder": encoder,
        "hostility_model": hostility_model,
        "nli_model": nli_model,
    }


# ---------------------------------------------------------------------------
# Synthetic Cascade & Corpus Builders (Strictly inside tmp_path / tests/e2e)
# ---------------------------------------------------------------------------


def make_synthetic_cascade_data_list(
    num_graphs: int = 8,
    *,
    in_dim: int = TOTAL_NODE_FEATURE_DIM,
    edge_dim: int = 2,
    num_nodes_per_graph: int = 6,
    num_rumor_classes: int = 4,
    include_tier2_prior: bool = False,
    tier2_prior_dim: int = 4,
    seed: int = 42,
    num_topics: int | None = None,
) -> list[CascadeData]:
    rng = torch.Generator().manual_seed(seed)
    graphs: list[CascadeData] = []
    n_topics = num_topics if num_topics is not None else num_graphs
    for i in range(num_graphs):
        n_nodes = max(2, num_nodes_per_graph)
        src_list = [0 if k == 1 else (k - 1) // 2 for k in range(1, n_nodes)]
        dst_list = list(range(1, n_nodes))
        td = torch.tensor([src_list, dst_list], dtype=torch.long)
        bu = td[[1, 0], :]
        stances = torch.tensor(
            [0.8 if (k % 2 == 1) else -0.75 for k in range(td.size(1))],
            dtype=torch.float32,
        )
        host = torch.tensor(
            [0.1 if (k % 2 == 1) else 0.7 for k in range(td.size(1))],
            dtype=torch.float32,
        )
        ea = torch.stack([stances, host], dim=-1)
        if edge_dim > 2:
            ea = torch.cat([ea, torch.zeros((ea.size(0), edge_dim - 2), dtype=torch.float32)], dim=-1)

        x = torch.randn((n_nodes, in_dim), generator=rng, dtype=torch.float32) * 0.25
        # Put meaningful temporal & depth slots in last 2 dims
        x[:, -2] = torch.linspace(0.0, 0.5, n_nodes)
        x[:, -1] = torch.linspace(0.0, 0.4, n_nodes)
        struct_x = torch.rand((n_nodes, STRUCTURAL_NODE_DIM), generator=rng, dtype=torch.float32) * 0.3
        c_stats = torch.rand((1, CASCADE_STATS_DIM), generator=rng, dtype=torch.float32)
        early_polar = torch.tensor([0.15 + 0.08 * (i % 5)], dtype=torch.float32)
        y_size = torch.tensor([2.5 + 0.35 * (i % 6)], dtype=torch.float32)
        y_polar = torch.tensor([min(0.95, 0.12 + 0.11 * (i % 6))], dtype=torch.float32)

        kwargs: dict[str, Any] = {
            "x": x,
            "struct_x": struct_x,
            "edge_index": td,
            "edge_index_td": td,
            "edge_index_bu": bu,
            "edge_attr": ea,
            "edge_stance": stances.unsqueeze(-1),
            "early_node_root_stance": torch.cat([torch.zeros(1), stances]),
            "cascade_stats": c_stats,
            "root_index": torch.tensor([0], dtype=torch.long),
            "y_size": y_size,
            "y_polar": y_polar,
            "early_y_polar": early_polar,
            "root_ts": float(1_700_000_000 + i * 3600),
            "y_rumor": torch.tensor([i % num_rumor_classes], dtype=torch.long),
            "cascade_id": f"cas_{i}",
            "topic_id": f"topic_{i % n_topics}",
            "k_horizon": 20,
            "node_ids": [f"cas_{i}_n{k}" for k in range(n_nodes)],
            "node_texts": [f"node text {k}" for k in range(n_nodes)],
            "edge_pairs": [(f"cas_{i}_n{s}", f"cas_{i}_n{d}") for s, d in zip(src_list, dst_list)],
        }
        if include_tier2_prior:
            kwargs["tier2_user_prior"] = torch.rand((n_nodes, tier2_prior_dim), generator=rng, dtype=torch.float32)
        graphs.append(CascadeData(**kwargs))
    return graphs


def build_synthetic_processed_corpus(
    data_dir: Path,
    *,
    num_topics: int = 9,
    cascades_per_topic: int = 2,
    replies_per_cascade: int = 28,
    emb_dim: int = 768,
    lang: str = "vi",
    min_total_replies_extracted: int = 25,
    zero_row_node_indices: set[int] | None = None,
    add_future_unseen_user: bool = False,
    rumor_labels: tuple[str, ...] | None = ("true", "false", "unverified", "non-rumor"),
    seed: int = 42,
) -> dict[str, Any]:
    """Create a full schema-valid processed corpus (Parquet + nlp_features.pt + topic_splits.json)."""
    rng = np.random.default_rng(seed)
    nodes_rows: list[dict[str, Any]] = []
    edges_rows: list[dict[str, Any]] = []
    edge_stance_map: dict[str, float] = {}
    edge_host_map: dict[str, float] = {}
    teacher_edge_stance_map: dict[str, float] = {}
    teacher_edge_host_map: dict[str, float] = {}

    users_pool = [f"u_{i:02d}" for i in range(10)]
    camp_a = set(users_pool[:5])

    base_ts = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    t_offset = 0.0
    topics = [f"topic_{t:02d}" for t in range(num_topics)]

    for t_idx, tid in enumerate(topics):
        for c_sub in range(cascades_per_topic):
            cid = f"cas_{t_idx}_{c_sub}"
            root_id = f"{cid}_r"
            root_user = users_pool[(t_idx + c_sub) % len(users_pool)]
            root_camp = 0 if root_user in camp_a else 1
            r_label = rumor_labels[(t_idx + c_sub) % len(rumor_labels)] if rumor_labels else None

            root_ts = base_ts + timedelta(seconds=t_offset)
            nodes_rows.append(
                {
                    "node_id": root_id,
                    "cascade_id": cid,
                    "topic_id": tid,
                    "parent_id": None,
                    "is_root": True,
                    "text": f"Chủ đề thảo luận gốc số {t_idx} {cid} với nội dung dài đủ tám từ trở lên",
                    "created_at_utc": root_ts,
                    "lang": lang,
                    "source_dataset": "synthetic_e2e",
                    "user_id_hash": root_user,
                    "rumor_label": r_label,
                }
            )
            node_ids_in_cas = [root_id]
            for k in range(1, replies_per_cascade + 1):
                t_offset += 60.0
                nid = f"{cid}_c{k:02d}"
                eid = f"{cid}_e{k:02d}"
                parent_id = root_id if k <= 5 else node_ids_in_cas[(k - 1) // 2]
                rep_user = users_pool[(t_idx + c_sub + k) % len(users_pool)]
                rep_camp = 0 if rep_user in camp_a else 1
                same_camp = root_camp == rep_camp

                # Make late replies (k > 20) slightly more polarized/opposing if same/cross varies
                s_val = 0.80 if same_camp else -0.80
                h_val = 0.08 if same_camp else 0.78
                if k > 20:
                    s_val = -0.85 if (k % 2 == 0) else 0.75
                    h_val = 0.82 if (k % 2 == 0) else 0.12

                rep_ts = base_ts + timedelta(seconds=t_offset)
                nodes_rows.append(
                    {
                        "node_id": nid,
                        "cascade_id": cid,
                        "topic_id": tid,
                        "parent_id": parent_id,
                        "is_root": False,
                        "text": f"Bình luận phản hồi thứ {k} trong chủ đề {cid} trình bày quan điểm rõ ràng",
                        "created_at_utc": rep_ts,
                        "lang": lang,
                        "source_dataset": "synthetic_e2e",
                        "user_id_hash": rep_user,
                        "rumor_label": None,
                    }
                )
                edges_rows.append(
                    {
                        "edge_id": eid,
                        "cascade_id": cid,
                        "source_id": parent_id,
                        "target_id": nid,
                        "created_at_utc": rep_ts,
                        "time_delay_sec": float((rep_ts - root_ts).total_seconds()),
                        "edge_type": "reply",
                    }
                )
                edge_stance_map[eid] = s_val
                edge_host_map[eid] = h_val
                teacher_edge_stance_map[eid] = s_val
                teacher_edge_host_map[eid] = h_val
                node_ids_in_cas.append(nid)
            t_offset += 600.0

    unseen_user = "u_future_only"
    if add_future_unseen_user:
        for extra_i in range(3):
            t_offset += 3600.0
            cid = f"cas_future_{extra_i}"
            tid = topics[-1]
            root_id = f"{cid}_r"
            root_ts = base_ts + timedelta(seconds=t_offset)
            nodes_rows.append(
                {
                    "node_id": root_id,
                    "cascade_id": cid,
                    "topic_id": tid,
                    "parent_id": None,
                    "is_root": True,
                    "text": "Chủ đề tương lai với người dùng mới hoàn toàn chưa xuất hiện trước đó",
                    "created_at_utc": root_ts,
                    "lang": lang,
                    "source_dataset": "synthetic_e2e",
                    "user_id_hash": users_pool[0],
                    "rumor_label": rumor_labels[0] if rumor_labels else None,
                }
            )
            for k in range(1, replies_per_cascade + 1):
                t_offset += 60.0
                nid = f"{cid}_c{k:02d}"
                eid = f"{cid}_e{k:02d}"
                rep_user = unseen_user if k == 1 else users_pool[k % len(users_pool)]
                rep_ts = base_ts + timedelta(seconds=t_offset)
                nodes_rows.append(
                    {
                        "node_id": nid,
                        "cascade_id": cid,
                        "topic_id": tid,
                        "parent_id": root_id,
                        "is_root": False,
                        "text": f"Bình luận tương lai số {k} có nội dung dài hơn tám từ",
                        "created_at_utc": rep_ts,
                        "lang": lang,
                        "source_dataset": "synthetic_e2e",
                        "user_id_hash": rep_user,
                        "rumor_label": None,
                    }
                )
                edges_rows.append(
                    {
                        "edge_id": eid,
                        "cascade_id": cid,
                        "source_id": root_id,
                        "target_id": nid,
                        "created_at_utc": rep_ts,
                        "time_delay_sec": float((rep_ts - root_ts).total_seconds()),
                        "edge_type": "reply",
                    }
                )
                edge_stance_map[eid] = -0.85 if k % 2 else 0.80
                edge_host_map[eid] = 0.80 if k % 2 else 0.05
                teacher_edge_stance_map[eid] = edge_stance_map[eid]
                teacher_edge_host_map[eid] = edge_host_map[eid]

    data_dir.mkdir(parents=True, exist_ok=True)
    nodes_tbl = pa.Table.from_pylist(nodes_rows, schema=NODES_ARROW_SCHEMA)
    edges_tbl = pa.Table.from_pylist(edges_rows, schema=EDGES_ARROW_SCHEMA)
    save_cascade_parquet(nodes_tbl, edges_tbl, data_dir, validate=True)

    embs = torch.tensor(rng.normal(1.5, 0.2, size=(len(nodes_rows), emb_dim)), dtype=torch.float32)
    if zero_row_node_indices:
        for idx in zero_row_node_indices:
            if 0 <= idx < embs.size(0):
                embs[idx] = 0.0

    edge_attr_tensor = torch.tensor(
        [[edge_stance_map[e["edge_id"]], edge_host_map[e["edge_id"]]] for e in edges_rows],
        dtype=torch.float32,
    )
    node_id_to_idx = {r["node_id"]: i for i, r in enumerate(nodes_rows)}
    edge_id_to_idx = {e["edge_id"]: i for i, e in enumerate(edges_rows)}
    feat_payload = {
        "dataset": "synthetic_e2e",
        "lang": lang,
        "edge_dim": 2,
        "min_total_replies_extracted": min_total_replies_extracted,
        "node_embeddings": embs,
        "node_id_to_idx": node_id_to_idx,
        "edge_attr": edge_attr_tensor,
        "edge_id_to_idx": edge_id_to_idx,
        "edge_stance": edge_stance_map,
        "edge_hostility": edge_host_map,
        "teacher_edge_stance": teacher_edge_stance_map,
        "teacher_edge_hostility": teacher_edge_host_map,
    }
    torch.save(feat_payload, data_dir / "nlp_features.pt")

    n_tr = max(1, int(round(0.65 * len(topics))))
    n_va = max(1, (len(topics) - n_tr) // 2)
    splits = {
        "train": set(topics[:n_tr]),
        "val": set(topics[n_tr : n_tr + n_va]),
        "test": set(topics[n_tr + n_va :]),
    }
    if not splits["test"]:
        splits["test"] = {topics[-1]}
        splits["val"] = {topics[-2]} if len(topics) >= 2 else set()
        splits["train"] = set(topics[:-2])
    save_topic_splits_json(splits, data_dir / "topic_splits.json", meta={"min_total_replies": min_total_replies_extracted})
    return {
        "data_dir": data_dir,
        "nodes_rows": nodes_rows,
        "edges_rows": edges_rows,
        "topics": topics,
        "splits": splits,
        "unseen_user": unseen_user,
        "feat_payload": feat_payload,
    }


# ---------------------------------------------------------------------------
# HTTP Server & Client Helpers for Demo Web App
# ---------------------------------------------------------------------------


def start_demo_server(
    *,
    data_dir: Path,
    ckpt_path: Path,
    ce_dir: Path,
    paper_dir: Path = PAPER_DIR,
    web_dir: Path = WEB_DIR,
    preload: bool = False,
) -> tuple[Any, threading.Thread, int]:
    server = create_server(
        host="127.0.0.1",
        port=0,
        data_dir=data_dir,
        ckpt_path=ckpt_path,
        ce_dir=ce_dir,
        paper_dir=paper_dir,
        web_dir=web_dir,
        preload=preload,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, int(server.server_address[1])


def http_get(port: int, path: str, *, timeout: float = 20.0) -> tuple[int, dict[str, str], bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        body = resp.read()
        headers = {k.lower(): v for k, v in resp.getheaders()}
        return resp.status, headers, body
    finally:
        conn.close()


def http_post_raw(
    port: int,
    path: str,
    raw_body: bytes,
    *,
    content_type: str = "application/json",
    timeout: float = 30.0,
) -> tuple[int, dict[str, str], bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request(
            "POST",
            path,
            body=raw_body,
            headers={"Content-Type": content_type, "Content-Length": str(len(raw_body))},
        )
        resp = conn.getresponse()
        body = resp.read()
        headers = {k.lower(): v for k, v in resp.getheaders()}
        return resp.status, headers, body
    finally:
        conn.close()


def http_post_json(port: int, path: str, payload: dict[str, Any]) -> tuple[int, dict[str, str], dict[str, Any]]:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    status, headers, body = http_post_raw(port, path, raw)
    return status, headers, json.loads(body.decode("utf-8"))


@pytest.fixture(scope="session")
def offline_hf_bundle() -> dict[str, Any]:
    return build_offline_encoder_and_classifiers()
