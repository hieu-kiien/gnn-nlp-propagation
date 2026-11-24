"""Macro-scale signed user-interaction social network analytics and temporal GNN datasets.

Bridges SBi-FAGAT's micro early-cascade tree modeling (|V_K| <= 21) with macro
cross-thread user-reply social network polarization across existing VN Reddit and
English PHEME corpora without requiring new data collection or NLP re-scoring:
  1. Structural Balance Theory (signed triangle census + sign-permutation z-score).
  2. Spectral Signed Laplacian bipartition, Signed Modularity Q_signed, and
     Garimella Random Walk Controversy (RWC) over echo-chamber hub cores.
  3. Leakage-free chronological Temporal Signed Link Prediction datasets where
     all node features and message-passing edges strictly satisfy t <= t_train_max.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import eigsh, lobpcg

from src.data.pyg_dataset import _is_valid_author_hash
from src.data.schema import load_cascade_parquet

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ts_seconds(val: Any) -> float:
    if hasattr(val, "timestamp"):
        return float(val.timestamp())
    return float(val)


def build_user_interaction_records(
    data_dir: Path,
    *,
    prefer_teacher: bool = True,
) -> dict[str, Any]:
    """Load valid-author user-user reply interactions aligned with 2D `[s_uv, h_uv]`."""
    root = Path(data_dir)
    nodes_tbl, edges_tbl = load_cascade_parquet(root, validate=True)
    feats_path = root / "nlp_features.pt"
    if not feats_path.is_file():
        raise FileNotFoundError(f"Missing processed NLP feature bank: {feats_path}")
    feats = torch.load(feats_path, map_location="cpu", weights_only=False)

    if prefer_teacher and feats.get("teacher_edge_stance"):
        stance_map = feats["teacher_edge_stance"]
        host_map = feats.get("teacher_edge_hostility") or feats.get("edge_hostility") or {}
        edge_label_source = "teacher"
    else:
        stance_map = feats.get("edge_stance") or {}
        host_map = feats.get("edge_hostility") or {}
        edge_label_source = "student"

    node_id_to_idx: dict[str, int] = {
        str(k): int(v) for k, v in (feats.get("node_id_to_idx") or {}).items()
    }
    node_embeddings: torch.Tensor | None = feats.get("node_embeddings")

    user_of_node: dict[str, str] = {}
    cascade_of_node: dict[str, str] = {}
    topic_of_node: dict[str, str] = {}
    cascades_by_user: dict[str, set[str]] = defaultdict(set)
    user_posts: dict[str, list[tuple[float, int]]] = defaultdict(list)

    for row in nodes_tbl.to_pylist():
        nid = str(row["node_id"])
        cid = str(row["cascade_id"])
        tid = str(row["topic_id"])
        cascade_of_node[nid] = cid
        topic_of_node[nid] = tid
        uh = row.get("user_id_hash")
        if _is_valid_author_hash(uh):
            u_str = str(uh)
            user_of_node[nid] = u_str
            cascades_by_user[u_str].add(cid)
            emb_idx = node_id_to_idx.get(nid)
            if emb_idx is not None:
                user_posts[u_str].append((_ts_seconds(row["created_at_utc"]), emb_idx))

    interactions: list[dict[str, Any]] = []
    for edge in edges_tbl.to_pylist():
        eid = str(edge["edge_id"])
        if eid not in stance_map:
            continue
        src_nid = str(edge["source_id"])
        dst_nid = str(edge["target_id"])
        src_u = user_of_node.get(src_nid)
        dst_u = user_of_node.get(dst_nid)
        if not src_u or not dst_u or src_u == dst_u:
            continue
        interactions.append(
            {
                "ts": _ts_seconds(edge["created_at_utc"]),
                "edge_id": eid,
                "cascade_id": str(edge.get("cascade_id") or cascade_of_node.get(dst_nid) or ""),
                "topic_id": topic_of_node.get(dst_nid, ""),
                "src_user": src_u,
                "dst_user": dst_u,
                "s_uv": float(min(1.0, max(-1.0, float(stance_map[eid])))),
                "h_uv": float(min(1.0, max(0.0, float(host_map.get(eid, 0.0))))),
            }
        )

    interactions.sort(key=lambda r: (r["ts"], r["cascade_id"], r["edge_id"]))
    return {
        "data_dir": str(root),
        "dataset": str(feats.get("dataset") or root.name),
        "edge_label_source": edge_label_source,
        "cascades_by_user": cascades_by_user,
        "user_posts": user_posts,
        "node_embeddings": node_embeddings,
        "interactions": interactions,
    }


def _largest_connected_component_nodes(
    adj_sets: Mapping[int, set[int]],
) -> list[int]:
    visited: set[int] = set()
    best_comp: list[int] = []
    for start in sorted(adj_sets):
        if start in visited or not adj_sets[start]:
            continue
        comp: list[int] = []
        queue: deque[int] = deque([start])
        visited.add(start)
        while queue:
            u = queue.popleft()
            comp.append(u)
            for v in adj_sets[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)
        if len(comp) > len(best_comp):
            best_comp = comp
    return sorted(best_comp)


def _signed_spectral_bipartition(
    lcc_nodes: Sequence[int],
    pair_signs: Mapping[tuple[int, int], tuple[int, float, int, int]],
) -> np.ndarray:
    """Compute 2-camp bipartition via normalized Signed Laplacian eigenvector."""
    n = len(lcc_nodes)
    if n <= 1:
        return np.zeros(n, dtype=np.int64)
    local_idx = {u: idx for idx, u in enumerate(lcc_nodes)}
    rows: list[int] = []
    cols: list[int] = []
    signed_vals: list[float] = []
    abs_deg = np.zeros(n, dtype=np.float64)

    for (u, v), (sign_val, weight, _, _) in pair_signs.items():
        lu = local_idx.get(u)
        lv = local_idx.get(v)
        if lu is None or lv is None or lu == lv:
            continue
        w = max(float(weight), 1e-6)
        sw = float(sign_val) * w
        rows.extend([lu, lv])
        cols.extend([lv, lu])
        signed_vals.extend([sw, sw])
        abs_deg[lu] += w
        abs_deg[lv] += w

    inv_sqrt_d = np.where(abs_deg > 1e-12, 1.0 / np.sqrt(np.maximum(abs_deg, 1e-12)), 0.0)
    a_signed = sparse.csr_matrix((signed_vals, (rows, cols)), shape=(n, n), dtype=np.float64)
    d_mat = sparse.diags(inv_sqrt_d, format="csr")
    lap = sparse.eye(n, format="csr", dtype=np.float64) - d_mat @ a_signed @ d_mat
    lap = 0.5 * (lap + lap.T)

    if n <= 250:
        evals, evecs = np.linalg.eigh(lap.toarray())
        order = np.argsort(evals)
        evecs = evecs[:, order]
    else:
        rng = np.random.default_rng(42)
        v0 = rng.normal(size=n)
        k_eig = min(3, n - 1)
        try:
            evals, evecs = eigsh(lap, k=k_eig, which="SA", tol=1e-4, maxiter=5000, v0=v0)
            evecs = evecs[:, np.argsort(evals)]
        except Exception:
            x0 = rng.normal(size=(n, k_eig))
            diag_pre = sparse.diags(1.0 / np.maximum(lap.diagonal(), 1e-3), format="csr")
            evals, evecs = lobpcg(lap, x0, M=diag_pre, largest=False, tol=1e-4, maxiter=500)
            evecs = evecs[:, np.argsort(evals)]

    vec = evecs[:, 0]
    camps = (vec >= 0.0).astype(np.int64)
    min_camp = min(int((camps == 0).sum()), int((camps == 1).sum()))
    if min_camp < max(1, int(0.05 * n)) and evecs.shape[1] >= 2:
        vec2 = evecs[:, 1]
        med = float(np.median(vec2))
        camps = (vec2 >= med).astype(np.int64)
        if min(int((camps == 0).sum()), int((camps == 1).sum())) == 0:
            order_idx = np.argsort(vec2)
            camps = np.zeros(n, dtype=np.int64)
            camps[order_idx[n // 2 :]] = 1
    return camps


def _compute_rwc_garimella(
    lcc_nodes: Sequence[int],
    camps: np.ndarray,
    pos_adj: Mapping[int, list[tuple[int, float]]],
    all_adj: Mapping[int, list[tuple[int, float]]],
    *,
    restart_alpha: float = 0.85,
    max_iters: int = 50,
) -> float:
    """Compute Random Walk Controversy (Garimella et al., ACM TSC 2018) over camp hubs."""
    n = len(lcc_nodes)
    if n < 4:
        return 0.0
    local_idx = {u: idx for idx, u in enumerate(lcc_nodes)}
    camp0 = np.where(camps == 0)[0]
    camp1 = np.where(camps == 1)[0]
    if camp0.size < 2 or camp1.size < 2:
        return 0.0

    # Prefer positive endorsement transitions; if too sparse (< 2 edges/node), fall back to all interactions
    use_adj = pos_adj
    pos_edges = sum(len(pos_adj.get(u, ())) for u in lcc_nodes)
    if pos_edges < n:
        use_adj = all_adj

    deg = np.zeros(n, dtype=np.float64)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for i, u in enumerate(lcc_nodes):
        nbrs = [(local_idx[v], max(w, 1e-6)) for v, w in use_adj.get(u, ()) if v in local_idx]
        if not nbrs and u in all_adj:
            nbrs = [(local_idx[v], 1e-3) for v, _ in all_adj.get(u, ()) if v in local_idx]
        s_w = sum(w for _, w in nbrs)
        deg[i] = s_w
        if s_w > 0.0:
            for j, w in nbrs:
                rows.append(i)
                cols.append(j)
                vals.append(w / s_w)

    if not vals:
        return 0.0
    p_row = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)
    p_col = p_row.T.tocsr()  # column transition: p_next = P^T @ p_curr

    k0 = max(1, min(50, max(2, camp0.size // 10), camp0.size - 1))
    k1 = max(1, min(50, max(2, camp1.size // 10), camp1.size - 1))
    hubs0 = camp0[np.argsort(deg[camp0])[-k0:]]
    hubs1 = camp1[np.argsort(deg[camp1])[-k1:]]
    is_hub0 = np.zeros(n, dtype=bool)
    is_hub1 = np.zeros(n, dtype=bool)
    is_hub0[hubs0] = True
    is_hub1[hubs1] = True
    non_hub = ~(is_hub0 | is_hub1)

    def _run_absorbing_walk(seed_nodes: np.ndarray) -> tuple[float, float]:
        init = np.zeros(n, dtype=np.float64)
        init[seed_nodes] = 1.0 / float(seed_nodes.size)
        active = init.copy()
        abs0 = 0.0
        abs1 = 0.0
        for _ in range(max_iters):
            nxt = restart_alpha * (p_col @ active)
            # Restart teleport returns mass back to same-camp distribution
            abs0 += float(nxt[is_hub0].sum())
            abs1 += float(nxt[is_hub1].sum())
            dangling = max(0.0, float(active.sum()) - float(nxt.sum()))
            restart_mass = (1.0 - restart_alpha) * float(active.sum()) + dangling
            nxt[~non_hub] = 0.0
            active = nxt + restart_mass * init
        total = abs0 + abs1
        if total <= 1e-12:
            return 0.5, 0.5
        return abs0 / total, abs1 / total

    p_00, p_01 = _run_absorbing_walk(camp0)
    p_10, p_11 = _run_absorbing_walk(camp1)
    rwc = p_00 * p_11 - p_01 * p_10
    return float(min(1.0, max(-1.0, rwc)))


def compute_macro_signed_network_stats(
    data_dir: Path,
    *,
    dataset_name: str | None = None,
    stance_threshold: float = 0.1,
    hostility_threshold: float = 0.5,
    num_null_perms: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute macro social network polarization metrics (Balance, Q_signed, RWC)."""
    rec = build_user_interaction_records(data_dir, prefer_teacher=True)
    interactions: list[dict[str, Any]] = rec["interactions"]
    if not interactions:
        raise ValueError(f"No valid user-user interactions found in '{data_dir}'")

    user_idx: dict[str, int] = {}
    pair_acc: dict[tuple[int, int], list[Any]] = defaultdict(lambda: [0.0, 0.0, 0, 0, 0])
    # pair_acc[u, v] = [sum_s, sum_h, total_cnt, neg_cnt, uncivil_cnt]
    directed_pairs: set[tuple[int, int]] = set()
    pos_events = 0
    neg_events = 0
    uncivil_events = 0

    for inter in interactions:
        su = user_idx.setdefault(inter["src_user"], len(user_idx))
        tu = user_idx.setdefault(inter["dst_user"], len(user_idx))
        directed_pairs.add((su, tu))
        sv = float(inter["s_uv"])
        hv = float(inter["h_uv"])
        if sv >= stance_threshold:
            pos_events += 1
        elif sv <= -stance_threshold:
            neg_events += 1
        if hv >= hostility_threshold:
            uncivil_events += 1
        a, b = (su, tu) if su < tu else (tu, su)
        slot = pair_acc[(a, b)]
        slot[0] += sv
        slot[1] += hv
        slot[2] += 1
        if sv <= -stance_threshold:
            slot[3] += 1
        if hv >= hostility_threshold:
            slot[4] += 1

    # Strong signed undirected user-pair graph for Structural Balance & Spectral Bipartition
    pair_signs: dict[tuple[int, int], tuple[int, float, int, int]] = {}
    adj_sets: dict[int, set[int]] = defaultdict(set)
    pos_adj: dict[int, list[tuple[int, float]]] = defaultdict(list)
    all_adj: dict[int, list[tuple[int, float]]] = defaultdict(list)

    for (a, b), (sum_s, _sum_h, cnt, _neg_c, _unc_c) in pair_acc.items():
        mean_s = sum_s / float(cnt)
        all_adj[a].append((b, float(cnt)))
        all_adj[b].append((a, float(cnt)))
        if abs(mean_s) >= stance_threshold:
            sgn = 1 if mean_s >= 0.0 else -1
            w = abs(mean_s) * math.log1p(cnt)
            pair_signs[(a, b)] = (sgn, w, cnt, 1)
            adj_sets[a].add(b)
            adj_sets[b].add(a)
            if sgn > 0:
                pos_adj[a].append((b, w))
                pos_adj[b].append((a, w))

    # Fast degree-oriented triangle enumeration
    edge_keys = sorted(pair_signs.keys())
    edge_to_id = {ek: idx for idx, ek in enumerate(edge_keys)}
    edge_sign_arr = np.asarray([pair_signs[ek][0] for ek in edge_keys], dtype=np.int8)
    deg_map = {u: len(nbrs) for u, nbrs in adj_sets.items()}
    tri_edge_ids: list[tuple[int, int, int]] = []

    for u, v in edge_keys:
        if deg_map.get(u, 0) > deg_map.get(v, 0) or (deg_map.get(u, 0) == deg_map.get(v, 0) and u > v):
            u_low, v_high = v, u
        else:
            u_low, v_high = u, v
        common = adj_sets[u_low].intersection(adj_sets[v_high])
        for w in common:
            # Keep unique canonical ordering (u_low, v_high, w) by degree then node id
            if (deg_map[v_high] < deg_map[w]) or (deg_map[v_high] == deg_map[w] and v_high < w):
                e_uv = edge_to_id[(min(u_low, v_high), max(u_low, v_high))]
                e_uw = edge_to_id[(min(u_low, w), max(u_low, w))]
                e_vw = edge_to_id[(min(v_high, w), max(v_high, w))]
                tri_edge_ids.append((e_uv, e_uw, e_vw))

    n_tri = len(tri_edge_ids)
    n_ppp = n_ppn = n_pnn = n_nnn = 0
    balanced_ratio = 0.0
    null_mean = 0.0
    null_std = 0.0
    balance_z = 0.0
    if n_tri > 0:
        tri_arr = np.asarray(tri_edge_ids, dtype=np.int64)
        tri_signs = edge_sign_arr[tri_arr]  # [n_tri, 3]
        pos_cnts = (tri_signs > 0).sum(axis=1)
        n_ppp = int((pos_cnts == 3).sum())
        n_ppn = int((pos_cnts == 2).sum())
        n_pnn = int((pos_cnts == 1).sum())
        n_nnn = int((pos_cnts == 0).sum())
        balanced_ratio = float((n_ppp + n_pnn) / float(n_tri))
        if num_null_perms > 0:
            rng = np.random.default_rng(seed)
            null_ratios = np.empty(num_null_perms, dtype=np.float64)
            perm_signs = edge_sign_arr.copy()
            for p_i in range(num_null_perms):
                rng.shuffle(perm_signs)
                ts_perm = perm_signs[tri_arr]
                balanced_perm = (ts_perm[:, 0] * ts_perm[:, 1] * ts_perm[:, 2]) > 0
                null_ratios[p_i] = float(balanced_perm.mean())
            null_mean = float(null_ratios.mean())
            null_std = float(null_ratios.std(ddof=1)) if num_null_perms >= 2 else 0.0
            balance_z = float((balanced_ratio - null_mean) / max(null_std, 1e-8))

    # Spectral bipartition on LCC + Signed Modularity + Echo-Chamber Homophily + RWC
    all_wcc = _largest_connected_component_nodes(adj_sets)
    camps = _signed_spectral_bipartition(all_wcc, pair_signs)
    camp_of_user = {u: int(camps[i]) for i, u in enumerate(all_wcc)}

    # Compute Signed Modularity Q_signed on LCC strong graph
    lcc_set = set(all_wcc)
    m_pos = m_neg = 0.0
    k_pos: dict[int, float] = defaultdict(float)
    k_neg: dict[int, float] = defaultdict(float)
    within_s: list[float] = []
    cross_s: list[float] = []
    within_neg = within_tot = within_unc = 0
    cross_neg = cross_tot = cross_unc = 0

    for (a, b), (sum_s, _sum_h, cnt, neg_c, unc_c) in pair_acc.items():
        if a in lcc_set and b in lcc_set:
            same = camp_of_user[a] == camp_of_user[b]
            mean_pair_s = sum_s / float(cnt)
            if same:
                within_s.append(mean_pair_s)
                within_neg += neg_c
                within_unc += unc_c
                within_tot += cnt
            else:
                cross_s.append(mean_pair_s)
                cross_neg += neg_c
                cross_unc += unc_c
                cross_tot += cnt
            if (a, b) in pair_signs:
                sgn, w, _, _ = pair_signs[(a, b)]
                if sgn > 0:
                    m_pos += w
                    k_pos[a] += w
                    k_pos[b] += w
                else:
                    m_neg += w
                    k_neg[a] += w
                    k_neg[b] += w

    def _sub_modularity(is_pos: bool) -> float:
        m_tot = m_pos if is_pos else m_neg
        if m_tot <= 1e-12:
            return 0.0
        k_map = k_pos if is_pos else k_neg
        target_sgn = 1 if is_pos else -1
        within_w = sum(
            w
            for (a, b), (sgn, w, _, _) in pair_signs.items()
            if a in lcc_set and b in lcc_set and sgn == target_sgn and camp_of_user[a] == camp_of_user[b]
        )
        sum_k_c0 = sum(k_map.get(u, 0.0) for u in all_wcc if camp_of_user[u] == 0)
        sum_k_c1 = sum(k_map.get(u, 0.0) for u in all_wcc if camp_of_user[u] == 1)
        expected = (sum_k_c0**2 + sum_k_c1**2) / (4.0 * m_tot)
        return float((within_w - expected) / m_tot)

    q_pos = _sub_modularity(True)
    q_neg = _sub_modularity(False)
    signed_modularity = float(q_pos - q_neg)
    rwc = _compute_rwc_garimella(all_wcc, camps, pos_adj, all_adj)

    cascades_by_user = rec["cascades_by_user"]
    total_users = len(cascades_by_user)
    multi_thread_users = sum(1 for cs in cascades_by_user.values() if len(cs) >= 2)
    active_5_users = sum(1 for cs in cascades_by_user.values() if len(cs) >= 5)

    return {
        "dataset": dataset_name or rec["dataset"],
        "edge_label_source": rec["edge_label_source"],
        "num_users": int(total_users),
        "active_graph_users": int(len(user_idx)),
        "multi_thread_users": int(multi_thread_users),
        "active_5plus_users": int(active_5_users),
        "num_interactions": int(len(interactions)),
        "unique_directed_pairs": int(len(directed_pairs)),
        "unique_undirected_pairs": int(len(pair_acc)),
        "signed_strong_pairs": int(len(pair_signs)),
        "pos_interactions": int(pos_events),
        "neg_interactions": int(neg_events),
        "uncivil_interactions": int(uncivil_events),
        "largest_signed_wcc_nodes": int(len(all_wcc)),
        "largest_signed_wcc_ratio": float(len(all_wcc) / max(1, len(user_idx))),
        "camp_sizes": [int((camps == 0).sum()), int((camps == 1).sum())],
        "structural_balance": {
            "num_triangles": int(n_tri),
            "n_ppp": n_ppp,
            "n_ppn": n_ppn,
            "n_pnn": n_pnn,
            "n_nnn": n_nnn,
            "balanced_ratio": balanced_ratio,
            "null_balanced_mean": null_mean,
            "null_balanced_std": null_std,
            "balance_z_score": balance_z,
        },
        "polarization": {
            "rwc": rwc,
            "signed_modularity": signed_modularity,
            "q_pos": q_pos,
            "q_neg": q_neg,
            "within_camp_mean_stance": float(np.mean(within_s)) if within_s else 0.0,
            "cross_camp_mean_stance": float(np.mean(cross_s)) if cross_s else 0.0,
            "within_camp_neg_rate": float(within_neg / max(1, within_tot)),
            "cross_camp_neg_rate": float(cross_neg / max(1, cross_tot)),
            "within_camp_uncivil_rate": float(within_unc / max(1, within_tot)),
            "cross_camp_uncivil_rate": float(cross_unc / max(1, cross_tot)),
        },
    }


def build_temporal_signed_link_dataset(
    data_dir: Path,
    *,
    dataset_name: str | None = None,
    stance_threshold: float = 0.1,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    prefer_teacher: bool = True,
) -> dict[str, Any]:
    """Build zero-leakage chronological signed user-link prediction dataset.

    Strict Zero-Leakage Guarantee:
      1. Signed interactions (`|s_uv| >= stance_threshold`) are sorted by UTC time
         and split strictly with `max(t_train) < min(t_val)` and `max(t_val) < min(t_test)`
         whenever timestamps differ.
      2. Node features (768-D mean SpanPool + 4-D unsigned activity stats) are aggregated
         ONLY over posts/edges occurring at or before `t_train_max`, keeping edge stance
         and hostility exclusively on `edge_attr`.
      3. Only `train` edges enter `train_edge_index` and `train_edge_attr`.
    """
    rec = build_user_interaction_records(data_dir, prefer_teacher=prefer_teacher)
    signed = [r for r in rec["interactions"] if abs(float(r["s_uv"])) >= stance_threshold]
    if len(signed) < 20:
        raise ValueError(f"Insufficient signed user interactions ({len(signed)}) in '{data_dir}'")

    n = len(signed)
    tr_end = max(2, int(round(train_ratio * n)))
    va_end = max(tr_end + 1, int(round((train_ratio + val_ratio) * n)))
    # Advance boundary past identical timestamps so time cutpoints remain strictly causal
    while tr_end < n - 2 and signed[tr_end]["ts"] <= signed[tr_end - 1]["ts"]:
        tr_end += 1
    while va_end < n - 1 and signed[va_end]["ts"] <= signed[va_end - 1]["ts"]:
        va_end += 1

    train_edges = signed[:tr_end]
    val_edges = signed[tr_end:va_end]
    test_edges = signed[va_end:]
    t_train_max = max(float(r["ts"]) for r in train_edges)

    all_users: list[str] = sorted({u for r in signed for u in (r["src_user"], r["dst_user"])})
    u_to_idx = {u: idx for idx, u in enumerate(all_users)}
    num_users = len(all_users)

    # Strictly Train-only node features (768-D SpanPool + 4-D unsigned activity stats)
    node_embeddings: torch.Tensor | None = rec["node_embeddings"]
    emb_dim = int(node_embeddings.size(1)) if isinstance(node_embeddings, torch.Tensor) and node_embeddings.ndim == 2 else 768
    x_sem = torch.zeros((num_users, emb_dim), dtype=torch.float32)

    user_posts = rec["user_posts"]
    cascades_tr: dict[str, set[str]] = defaultdict(set)
    post_cnt_tr: dict[str, int] = defaultdict(int)
    for inter in train_edges:
        cascades_tr[inter["src_user"]].add(inter["cascade_id"])
        cascades_tr[inter["dst_user"]].add(inter["cascade_id"])

    if isinstance(node_embeddings, torch.Tensor) and node_embeddings.numel() > 0:
        for u, idx in u_to_idx.items():
            tr_idxs = [e_i for ts, e_i in user_posts.get(u, ()) if ts <= t_train_max]
            post_cnt_tr[u] = len(tr_idxs)
            if tr_idxs:
                x_sem[idx] = node_embeddings[tr_idxs].float().mean(dim=0)

    out_cnt = np.zeros(num_users, dtype=np.float32)
    in_cnt = np.zeros(num_users, dtype=np.float32)

    train_user_set: set[int] = set()
    for r in train_edges:
        su = u_to_idx[r["src_user"]]
        tu = u_to_idx[r["dst_user"]]
        train_user_set.add(su)
        train_user_set.add(tu)
        out_cnt[su] += 1.0
        in_cnt[tu] += 1.0

    beh = np.zeros((num_users, 4), dtype=np.float32)
    for u, idx in u_to_idx.items():
        if idx not in train_user_set and post_cnt_tr[u] == 0:
            continue
        beh[idx] = np.asarray(
            [
                math.log1p(post_cnt_tr[u]) / 5.0,
                math.log1p(len(cascades_tr.get(u, ()))) / 5.0,
                math.log1p(float(out_cnt[idx])) / 5.0,
                math.log1p(float(in_cnt[idx])) / 5.0,
            ],
            dtype=np.float32,
        )

    x_full = torch.cat([x_sem, torch.from_numpy(beh)], dim=-1)

    def _pack_split(rows: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        pairs = torch.tensor(
            [[u_to_idx[r["src_user"]], u_to_idx[r["dst_user"]]] for r in rows],
            dtype=torch.long,
        ).t().contiguous()
        y_opp = torch.tensor(
            [1.0 if float(r["s_uv"]) <= -stance_threshold else 0.0 for r in rows],
            dtype=torch.float32,
        )
        attr = torch.tensor(
            [[float(r["s_uv"]), float(r["h_uv"])] for r in rows],
            dtype=torch.float32,
        )
        ts_t = torch.tensor([float(r["ts"]) for r in rows], dtype=torch.float64)
        both_seen = torch.tensor(
            [(u_to_idx[r["src_user"]] in train_user_set and u_to_idx[r["dst_user"]] in train_user_set) for r in rows],
            dtype=torch.bool,
        )
        return {
            "pairs": pairs,
            "y_oppose": y_opp,
            "edge_attr": attr,
            "ts": ts_t,
            "both_seen": both_seen,
        }

    return {
        "dataset": dataset_name or rec["dataset"],
        "edge_label_source": rec["edge_label_source"],
        "num_users": num_users,
        "in_dim": int(x_full.size(-1)),
        "x": x_full,
        "train_user_set": train_user_set,
        "t_train_max": t_train_max,
        "train": _pack_split(train_edges),
        "val": _pack_split(val_edges),
        "test": _pack_split(test_edges),
    }


def format_macro_network_latex(
    stats_list: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    *,
    caption: str = (
        "Macro Signed User-Interaction Social Network Polarization Metrics: "
        "Structural Balance Ratio and Permutation $z$-Score, Signed Modularity $Q_{\\text{signed}}$, "
        "Garimella Random Walk Controversy (RWC), and Cross- vs.\\ Within-Camp Conflict Rates."
    ),
    label: str = "tab:macro_network",
) -> str:
    """Format macro signed social network statistics into an IEEEtran `booktabs` LaTeX table."""
    items = [stats_list] if isinstance(stats_list, Mapping) else list(stats_list)
    rows: list[str] = []
    for item in items:
        ds = str(item.get("dataset", "dataset")).replace("_", "\\_")
        users = int(item.get("num_users", 0))
        pairs = int(item.get("unique_undirected_pairs", 0))
        wcc = int(item.get("largest_signed_wcc_nodes", 0))
        sb = item.get("structural_balance") or {}
        pol = item.get("polarization") or {}
        bal_pct = 100.0 * float(sb.get("balanced_ratio", 0.0))
        bal_z = float(sb.get("balance_z_score", 0.0))
        rwc = float(pol.get("rwc", 0.0))
        q_s = float(pol.get("signed_modularity", 0.0))
        w_neg = 100.0 * float(pol.get("within_camp_neg_rate", 0.0))
        c_neg = 100.0 * float(pol.get("cross_camp_neg_rate", 0.0))
        rows.append(
            f"{ds} & {users:,} & {pairs:,} & {wcc:,} & {bal_pct:.1f}\\% ({bal_z:+.2f}) & "
            f"{q_s:.4f} & {rwc:.4f} & {w_neg:.1f}\\% / {c_neg:.1f}\\% \\\\"
        )

    return "\n".join(
        [
            "\\begin{table*}[t]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\resizebox{\\textwidth}{!}{",
            "\\begin{tabular}{lccccccc}",
            "\\toprule",
            "Corpus & Users & User Pairs & Signed WCC & Bal. Tri. ($z$) $\\uparrow$ & $Q_{\\text{signed}}$ $\\uparrow$ & RWC $\\uparrow$ & Opp. Rate (In / Cross) \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\end{table*}",
            "",
        ]
    )


def plot_macro_signed_network(
    stats: Mapping[str, Any],
    *,
    output_dir: Path = Path("paper/figures/vn"),
) -> tuple[Path, Path]:
    """Export 2-panel vector figure for Macro Structural Balance and Camp Echo-Chamber contrast."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sb = stats.get("structural_balance") or {}
    pol = stats.get("polarization") or {}
    ds = str(stats.get("dataset", "corpus")).upper()
    n_tri = max(1, int(sb.get("num_triangles", 0)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)

    # Panel A: Signed Triangle Census vs Null Balanced Expectation
    tri_labels = ["(+,+,+)\nBalanced", "(+,+,−)\nUnbalanced", "(+,−,−)\nBalanced", "(−,−,−)\nUnbalanced"]
    tri_vals = [
        100.0 * float(sb.get("n_ppp", 0)) / n_tri,
        100.0 * float(sb.get("n_ppn", 0)) / n_tri,
        100.0 * float(sb.get("n_pnn", 0)) / n_tri,
        100.0 * float(sb.get("n_nnn", 0)) / n_tri,
    ]
    colors_a = ["#1b9e77", "#d95f02", "#66a61e", "#e7298a"]
    bars = ax1.bar(tri_labels, tri_vals, color=colors_a, width=0.58, edgecolor="black", linewidth=0.6)
    for bar, val in zip(bars, tri_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, val + 1.0, f"{val:.1f}%", ha="center", va="bottom", fontsize=8.5)
    bal_z = float(sb.get("balance_z_score", 0.0))
    bal_obs = 100.0 * float(sb.get("balanced_ratio", 0.0))
    bal_null = 100.0 * float(sb.get("null_balanced_mean", 0.0))
    ax1.set_ylabel("Share of Signed User Triangles (%)")
    ax1.set_ylim(0.0, max(50.0, max(tri_vals) + 12.0))
    ax1.set_title(f"{ds} Structural Balance ({n_tri:,} Triangles)\nObs={bal_obs:.1f}% vs Null={bal_null:.1f}% (z={bal_z:+.2f})")
    ax1.grid(axis="y", alpha=0.25, linestyle="--")

    # Panel B: Echo-Chamber Polarization Contrast (Within-Camp vs Cross-Camp)
    cats = ["Opposition Rate (%)", "Incivility Rate (%)"]
    in_vals = [100.0 * float(pol.get("within_camp_neg_rate", 0.0)), 100.0 * float(pol.get("within_camp_uncivil_rate", 0.0))]
    cr_vals = [100.0 * float(pol.get("cross_camp_neg_rate", 0.0)), 100.0 * float(pol.get("cross_camp_uncivil_rate", 0.0))]
    x = np.arange(len(cats))
    w = 0.34
    b1 = ax2.bar(x - w / 2, in_vals, width=w, label="Within Fiedler Camp", color="#386cb0", edgecolor="black", linewidth=0.6)
    b2 = ax2.bar(x + w / 2, cr_vals, width=w, label="Cross-Camp Reply", color="#f0027f", edgecolor="black", linewidth=0.6)
    for bar, val in list(zip(b1, in_vals)) + list(zip(b2, cr_vals)):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, val + 1.0, f"{val:.1f}%", ha="center", va="bottom", fontsize=8.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(cats)
    ax2.set_ylabel("Interaction Rate (%)")
    ax2.set_ylim(0.0, max(50.0, max(in_vals + cr_vals) + 14.0))
    ax2.set_title(f"{ds} Spectral Bipartition\nRWC={float(pol.get('rwc', 0.0)):.4f} | Q_signed={float(pol.get('signed_modularity', 0.0)):.4f}")
    ax2.legend(loc="upper right", frameon=True, fontsize=8.5)
    ax2.grid(axis="y", alpha=0.25, linestyle="--")

    pdf_path = out_dir / "fig5_macro_signed_network.pdf"
    svg_path = out_dir / "fig5_macro_signed_network.svg"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, svg_path
