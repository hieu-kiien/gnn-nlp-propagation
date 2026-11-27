"""Offline-first local Defense Demo Web MVP for SBi-FAGAT (Tier 1 & Tier 2).

Serves a hardened localhost (`127.0.0.1`) pure-stdlib HTTP server powering:
  - Tab 1 (Micro / Early Cascade, K=20):
      * 10 diverse real VN Test-split cascades cached from `data/processed/vn/`
        plus `checkpoints/best_sbifagat_vn.pt` (automatic 2-preset synthetic
        fallback when local data or checkpoints are absent on CI/fresh clones).
      * Interactive Tree <-> Thread User-Graph view toggle, top-5 trigger edge
        inspection, and optional hidden-by-default hostility badge toggle.
      * Live Sandbox demonstrating Path Contraction (`junk_mode="contract"`)
        preserving short Vietnamese stances while collapsing mid-chain junk.
  - Tab 2 (Macro / Signed Social Network):
      * Live reads of `results/results_macro_network_{vn,pheme}.json`,
        `results/results_signed_link_{vn,pheme}.json`, and static `fig5_*.svg`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
import webbrowser
from collections import defaultdict, deque
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence
from urllib.parse import parse_qs, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.predict_live_thread import (  # noqa: E402
    VN_POLAR_Q50,
    VN_POLAR_Q75,
    VN_POLAR_Q90,
    VN_QUANTILE_CALIBRATION_NOTE,
    _classify_calibrated_regime,
    _extract_post_id_from_url,
    _fetch_reddit_thread,
    _make_demo_offline_thread,
    predict_live_cascade,
)
from src.data.arctic_shift_vn import hash_author_sha256  # noqa: E402
from src.data.pyg_dataset import (  # noqa: E402
    EarlyCascadeDataset,
    _compute_root_aligned_stances_and_orientations,
    _is_valid_author_hash,
    load_topic_splits_json,
)
from src.models.sbi_fagat import SBiFAGAT  # noqa: E402
from src.nlp.topic_post_analyzer import (  # noqa: E402
    analyze_large_thread_dynamics,
    classify_post_topics,
)

_DEMO_2D_BEHAVIOR_KEYS: tuple[str, ...] = (
    "support",
    "neutral",
    "constructive_oppose",
    "hostile_conflict",
)
_REDACTED_MARKER_RE = re.compile(r"\[redacted", re.IGNORECASE)
_NSFW_MARKER_RE = re.compile(
    r"(?:\bnsfw\b|\b18\+\b|\bjav\b|\bporn\b|\bhentai\b|\bse\*?x\b|quan\s*h[ệe]\s*t[ìi]nh\s*d[ụu]c)",
    re.IGNORECASE,
)

ALLOWED_LOCAL_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
_REQUIRED_VN_FILES: tuple[str, ...] = (
    "nodes.parquet",
    "edges.parquet",
    "nlp_features.pt",
    "topic_splits.json",
)
_REQUIRED_CE_FILES: tuple[str, ...] = (
    "dual_head_weights.pt",
    "dual_head_meta.json",
    "config.json",
)
_MAX_POST_BYTES: int = 1024 * 1024

_EXPLICIT_REPLY_RE = re.compile(
    r"^\s*[-*]?\s*([A-Za-z0-9_]+)\s*\(\s*(?:trả\s*lời|tra\s*loi|reply(?:\s+to)?|->)\s*([A-Za-z0-9_]+)\s*\)\s*:\s*(.+)$",
    re.IGNORECASE,
)
_BRACKET_PREFIX_RE = re.compile(r"^\s*\[([^\]]+)\]\s*(.+)$")
_BRACKET_ARROW_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*->\s*([A-Za-z0-9_]+)\s*$")


def _safe_short_author_hash(raw_hash: Any, fallback_idx: int, *, is_root: bool = False) -> tuple[str, bool]:
    """Convert full author SHA-256 hash into a compact safe identifier without leaking raw usernames."""
    if _is_valid_author_hash(raw_hash):
        clean = str(raw_hash).strip()
        return f"u_{clean[:6]}", False
    if is_root:
        return "u_root", True
    return f"u_anon_{fallback_idx}", True


def _clean_title_snippet(text: str, max_chars: int = 120) -> str:
    collapsed = " ".join(str(text or "").split())
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max(0, max_chars - 1)].rstrip() + "…"


def _stance_class_from_score(s_uv: float) -> Literal["support", "oppose", "neutral"]:
    if s_uv > 0.1:
        return "support"
    if s_uv < -0.1:
        return "oppose"
    return "neutral"


def _behavior_from_2d(s_uv: float, h_uv: float) -> str:
    if s_uv > 0.1:
        return "support"
    if h_uv > 0.6:
        return "hostile_conflict"
    if s_uv < -0.1:
        return "constructive_oppose"
    return "neutral"


def _build_thread_user_graph(
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Project a single cascade reply tree into a signed thread-level user graph."""
    if not nodes:
        return {"nodes": [], "edges": [], "camp_counts": {"camp0": 0, "camp1": 0}}

    node_by_id: dict[str, Mapping[str, Any]] = {str(n["node_id"]): n for n in nodes}
    root_node = next((n for n in nodes if bool(n.get("is_root"))), nodes[0])
    root_id = str(root_node["node_id"])

    children_adj: dict[str, list[str]] = {str(n["node_id"]): [] for n in nodes}
    edge_lookup: dict[tuple[str, str], Mapping[str, Any]] = {}
    stance_map: dict[str, float] = {}
    for idx, e in enumerate(edges):
        src = str(e["source_id"])
        dst = str(e["target_id"])
        eid = f"{src}->{dst}#{idx}"
        children_adj.setdefault(src, []).append(dst)
        edge_lookup[(src, dst)] = {"edge_id": eid}
        stance_map[eid] = float(e.get("s_uv", 0.0))

    _, node_orientations = _compute_root_aligned_stances_and_orientations(
        root_id=root_id,
        children_adj=children_adj,
        edge_lookup=edge_lookup,
        stance_map=stance_map,
    )

    user_stats: dict[str, dict[str, Any]] = {}
    node_to_user: dict[str, str] = {}
    for n in nodes:
        nid = str(n["node_id"])
        uid = str(n["author_hash"])
        node_to_user[nid] = uid
        slot = user_stats.setdefault(
            uid,
            {
                "user_id": uid,
                "label": uid,
                "orientations": [],
                "reply_count": 0,
                "is_op": False,
                "uncivil_count": 0,
            },
        )
        if bool(n.get("is_root")):
            slot["is_op"] = True
            slot["orientations"].append(1.0)
        else:
            slot["reply_count"] += 1
            slot["orientations"].append(float(node_orientations.get(nid, 0.0)))
            if bool(n.get("is_uncivil")):
                slot["uncivil_count"] += 1

    pair_acc: dict[tuple[str, str], dict[str, Any]] = {}
    for e in edges:
        src_nid = str(e["source_id"])
        dst_nid = str(e["target_id"])
        src_u = node_to_user.get(src_nid)
        dst_u = node_to_user.get(dst_nid)
        if not src_u or not dst_u or src_u == dst_u:
            continue
        pair = (dst_u, src_u)  # responder -> target user
        slot = pair_acc.setdefault(
            pair,
            {"source": pair[0], "target": pair[1], "sum_s": 0.0, "max_h": 0.0, "count": 0},
        )
        slot["sum_s"] += float(e.get("s_uv", 0.0))
        slot["max_h"] = max(slot["max_h"], float(e.get("h_uv", 0.0)))
        slot["count"] += 1

    # Initial camp assignment by root-aligned stance balance; refine undecided neutral users via neighbors
    camp_map: dict[str, int] = {}
    balance_map: dict[str, float] = {}
    for uid, st in user_stats.items():
        oris = st["orientations"]
        bal = sum(oris) / float(len(oris)) if oris else 0.0
        balance_map[uid] = float(bal)
        if st["is_op"] or bal >= -0.05:
            camp_map[uid] = 0
        else:
            camp_map[uid] = 1

    for uid, bal in balance_map.items():
        if user_stats[uid]["is_op"] or abs(bal) > 0.05:
            continue
        vote_c0 = 0.0
        vote_c1 = 0.0
        for (u_a, u_b), acc in pair_acc.items():
            if u_a != uid and u_b != uid:
                continue
            other = u_b if u_a == uid else u_a
            mean_s = acc["sum_s"] / float(acc["count"])
            other_camp = camp_map.get(other, 0)
            if mean_s > 0.1:
                if other_camp == 0:
                    vote_c0 += 1.0
                else:
                    vote_c1 += 1.0
            elif mean_s < -0.1:
                if other_camp == 0:
                    vote_c1 += 1.0
                else:
                    vote_c0 += 1.0
        if vote_c1 > vote_c0:
            camp_map[uid] = 1

    graph_nodes: list[dict[str, Any]] = []
    for uid in sorted(
        user_stats.keys(),
        key=lambda k: (camp_map[k], not user_stats[k]["is_op"], -user_stats[k]["reply_count"], k),
    ):
        st = user_stats[uid]
        graph_nodes.append(
            {
                "user_id": uid,
                "label": f"{uid} (OP)" if st["is_op"] else uid,
                "camp": int(camp_map[uid]),
                "stance_balance": round(float(balance_map[uid]), 4),
                "reply_count": int(st["reply_count"] + (1 if st["is_op"] else 0)),
                "is_op": bool(st["is_op"]),
                "is_uncivil": st["uncivil_count"] > 0,
            }
        )

    graph_edges: list[dict[str, Any]] = []
    for (_, _), acc in sorted(pair_acc.items()):
        cnt = int(acc["count"])
        mean_s = float(acc["sum_s"] / float(cnt))
        max_h = float(acc["max_h"])
        graph_edges.append(
            {
                "source": acc["source"],
                "target": acc["target"],
                "s_uv": round(mean_s, 4),
                "h_uv": round(max_h, 4),
                "count": cnt,
                "stance_class": _stance_class_from_score(mean_s),
                "is_uncivil": max_h >= 0.5,
            }
        )

    camp0_n = sum(1 for n in graph_nodes if n["camp"] == 0)
    camp1_n = sum(1 for n in graph_nodes if n["camp"] == 1)
    return {
        "nodes": graph_nodes,
        "edges": graph_edges,
        "camp_counts": {"camp0": camp0_n, "camp1": camp1_n},
    }


def _make_synthetic_polarized_preset() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Second clearly labeled synthetic multi-camp VN debate for offline fallback."""
    base_ts = 1763300000.0
    post_id = "demo_vn_live02"
    post = {
        "id": post_id,
        "title": "[Mẫu tổng hợp] Tranh luận quy định làm việc tại văn phòng 5 ngày/tuần (bỏ cơ chế hybrid)",
        "selftext": (
            "Nhiều công ty công nghệ đang thu hẹp hình thức làm việc từ xa (WFH/hybrid) và yêu cầu "
            "nhân sự lên văn phòng toàn thời gian để tăng phối hợp trực tiếp."
        ),
        "subreddit": "vozforums",
        "author": "op_demo_2",
        "created_utc": base_ts,
    }
    rows = [
        ("r01", f"t3_{post_id}", 40, "u_a", "Đồng ý với công ty nếu dự án cần trao đổi kiến trúc liên tục, ngồi cạnh nhau chốt phương án nhanh hơn hẳn."),
        ("r02", f"t3_{post_id}", 85, "u_b", "Không đồng ý chút nào, kẹt xe 2 tiếng mỗi ngày chỉ để ngồi họp online qua Teams tại bàn văn phòng."),
        ("r03", "r01", 125, "u_c", "Chuẩn bác, với thực tập sinh và junior thì lên văn phòng dễ hỏi bài và bắt nhịp văn hóa đội ngũ hơn."),
        ("r04", "r02", 160, "u_d", "Chưa hợp lý ở chỗ đánh đồng mọi vị trí; lập trình viên làm việc độc lập hoàn toàn có thể đo bằng KPI sản phẩm."),
        ("r05", "r01", 205, "u_e", "Tuy nhiên nếu bắt lên đủ 5 ngày mà không hỗ trợ gửi xe hay trợ cấp đi lại thì nhân sự giỏi sẽ nhảy việc hết."),
        ("r06", "r04", 240, "u_b", "Hợp lý, giữ mô hình 3 ngày văn phòng + 2 ngày WFH là cân bằng nhất cho cả hai phía."),
        ("r07", "r03", 290, "u_f", "Não tàn mới tin mấy bài văn mẫu gắn kết đội nhóm, thực chất chỉ là sếp muốn soi màn hình nhân viên!"),
        ("r08", "r07", 330, "u_a", "Bạn tranh luận đàng hoàng đi chứ đừng chụp mũ công kích cá nhân như vậy, mỗi mô hình doanh nghiệp có lý do riêng."),
        ("r09", "r02", 380, "u_g", "Công ty mình giữ hybrid 2 năm nay hiệu suất vẫn tăng đều, tiết kiệm đáng kể chi phí thuê mặt bằng."),
        ("r10", f"t3_{post_id}", 420, "u_h", "Cho mình hỏi bộ phận nhân sự bên các bạn đo lường hiệu suất WFH bằng công cụ nào để tránh thiếu minh bạch?"),
        ("r11", "r05", 460, "u_c", "Đúng rồi, muốn bắt onsite 100% thì phải điều chỉnh lương thưởng tương xứng chi phí sinh hoạt đô thị."),
        ("r12", "r08", 510, "u_f", "Cút về mà bợ đỡ sếp đi đồ ngu, nói chuyện lý thuyết suông nghe ngứa mắt thực sự!"),
        ("r13", "r06", 550, "u_d", "Ủng hộ phương án hybrid linh hoạt theo từng nhóm dự án thay vì ra lệnh cứng nhắc toàn công ty."),
        ("r14", "r09", 590, "u_e", "Nhầm rồi nếu nói WFH lúc nào cũng tốt; nhiều team bị đứt gãy thông tin và làm việc quá giờ liên tục."),
        ("r15", "r10", 630, "u_g", "Bên mình dùng Jira kết hợp check-in ngắn 15 phút buổi sáng và đánh giá theo sprint 2 tuần."),
        ("r16", "r12", 670, "u_h", "Thôi bớt chửi bới công kích lại, cứ nhìn tỷ lệ nghỉ việc sau 6 tháng áp dụng là rõ ngay hiệu quả."),
        ("r17", "r13", 710, "u_b", "Chính xác, số liệu nghỉ việc và tiến độ bàn giao sẽ trả lời công bằng nhất."),
        ("r18", "r14", 750, "u_c", "Mình thấy giai đoạn chạy nước rút lên văn phòng rất hiệu quả, còn giai đoạn bảo trì cứ cho linh hoạt."),
        ("r19", "r11", 790, "u_a", "Hợp lý, chính sách phân tầng theo vai trò sẽ giảm xung đột giữa ban điều hành và nhân viên."),
        ("r20", "r15", 830, "u_d", "Cảm ơn bạn đã chia sẻ quy trình cụ thể, cách đo theo sprint khá khách quan."),
        # Post-K (> 20) for post-hoc check
        ("r21", "r18", 880, "u_e", "Tôi tán thành cách kết hợp theo chu kỳ sprint, vừa giữ kỷ luật vừa giảm mệt mỏi di chuyển."),
        ("r22", "r16", 930, "u_f", "Nói chung tụi quản lý bảo thủ vẫn thích nhìn thấy người ngồi kín bàn bất kể hiệu quả thế nào."),
        ("r23", "r19", 980, "u_b", "Mong các công ty lắng nghe khảo sát ẩn danh trước khi ban hành quyết định quay lại 100%."),
    ]
    comments = [
        {
            "fullname": f"t1_{cid}",
            "id": cid,
            "link_id": f"t3_{post_id}",
            "parent_id": pid if pid.startswith(("t3_", "t1_")) else f"t1_{pid}",
            "body": body,
            "text": body,
            "created_utc": base_ts + dt,
            "author": author,
            "author_raw": author,
            "author_hash": hash_author_sha256(author),
        }
        for cid, pid, dt, author, body in rows
    ]
    return post, comments


def build_sandbox_preset(preset_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return in-memory Vietnamese sandbox presets (including a 42-reply large AI/Tech thread)."""
    base_ts = 1763400000.0
    if preset_id == "preset_large_1":
        post_id = "sandbox_large_1"
        post = {
            "id": post_id,
            "title": "[Sandbox Bài lớn] AI tạo sinh trong đào tạo lập trình đại học và cơ hội việc làm IT của sinh viên",
            "selftext": (
                "Nhiều trường đại học đang cân nhắc cho sinh viên dùng trợ lý AI tạo sinh trong bài tập lớn "
                "lập trình, trong khi doanh nghiệp phần mềm lo ngại sinh viên hổng tư duy thuật toán và kỹ năng debug."
            ),
            "subreddit": "vozforums",
            "author": "op_large_ai",
            "created_utc": base_ts,
        }
        spec = [
            ("l01", f"t3_{post_id}", 40, "u_ai_01", "Ủng hộ đưa AI vào giảng dạy nhưng phải dạy cách kiểm chứng mã nguồn và viết test thay vì copy mù quáng."),
            ("l02", f"t3_{post_id}", 80, "u_ai_02", "Chưa đồng ý cho năm nhất dùng AI ngay vì sẽ làm mất nền tảng cấu trúc dữ liệu và tư duy giải thuật."),
            ("l03", "l01", 120, "u_ai_03", "Đồng ý với bác, năm 1-2 nên thi giấy/cấm AI hoàn toàn, lên năm 3 làm đồ án thực tế mới cho dùng."),
            ("l04", "l02", 160, "[deleted]", "[deleted]"),
            ("l05", "l04", 200, "u_ai_04", "Hợp lý, hồi phỏng vấn fresher vừa rồi nhiều bạn dùng AI viết API nhưng không giải thích nổi độ phức tạp O(n)."),
            ("l06", f"t3_{post_id}", 240, "u_ai_05", "Cho mình hỏi các trường Bách Khoa hay UIT hiện có môn riêng về kỹ thuật Prompt và AI Engineer chưa?"),
            ("l07", "l01", 280, "u_ai_06", "Doanh nghiệp bây giờ tuyển người biết dùng AI tăng năng suất 3 lần chứ ai còn cấm đoán nữa."),
            ("l08", "l03", 320, "u_ai_07", "Chuẩn rồi bạn, công cụ giúp viết boilerplate nhanh hơn hẳn để tập trung vào kiến trúc hệ thống."),
            ("l09", "l05", 360, "u_ai_08", "Phản đối việc đổ lỗi hoàn toàn cho AI; chương trình cũ dạy quá nặng lý thuyết suông mới là vấn đề."),
            ("l10", "l06", 400, "u_ai_09", "Bên mình có học phần ứng dụng mô hình ngôn ngữ lớn và tích hợp vector database vào đồ án tốt nghiệp."),
            ("l11", "l07", 440, "u_ai_10", "Không đồng ý chút nào, năng suất ảo thôi chứ sinh ra lắm lỗi bảo mật ngầm khi đưa lên production."),
            ("l12", "l09", 480, "u_ai_11", "hóng ké"),
            ("l13", "l12", 520, "u_ai_02", "Đúng thế, khi AI viết sai logic phân quyền mà người lập trình không đọc hiểu code thì hậu quả rất nặng."),
            ("l14", "l10", 560, "u_ai_12", "Cảm ơn thông tin của bác, giáo trình cập nhật hướng MLOps và kiểm thử tự động nghe thực tế hơn hẳn."),
            ("l15", "l11", 600, "u_ai_01", "Bởi vậy mới cần dạy kỹ năng code review, profiling và viết unit test tự động song song với AI."),
            ("l16", "l13", 640, "u_ai_13", "Toàn mấy ông lý thuyết cổ hủ sợ mất việc nên bài xích AI, thực tế ngoài thị trường AI làm hộ 70% việc rồi!"),
            ("l17", "l16", 680, "u_ai_14", "Nói chuyện trịch thượng thế bác, cứ thử vận hành hệ thống giao dịch thật xem AI có tự fix deadlock cho không."),
            ("l18", "l16", 720, "u_ai_15", "Não tàn mới tin AI thay thế hoàn toàn kỹ sư phần mềm, toàn lùa gà bán khóa học prompt!"),
            ("l19", "l18", 760, "u_ai_13", "Cút về code chay tiếp đi, vài năm nữa không biết dùng AI agent thì ra đường mà cạp đất!"),
            ("l20", f"t3_{post_id}", 800, "u_ai_16", "."),
            ("l21", "l20", 840, "u_ai_17", "Đồng ý giữ thái độ tranh luận xây dựng, chia 2 giai đoạn học thuật và thực hành là hợp lý nhất."),
            ("l22", "l17", 880, "u_ai_04", "Chính xác, debug đa luồng hay tối ưu truy vấn SQL phức tạp vẫn đòi hỏi kiến thức hệ điều hành vững."),
            ("l23", "l19", 920, "u_ai_18", "Bạn công kích cá nhân quá rồi, người phản biện chỉ lo sinh viên bị lệ thuộc khi chưa có tư duy nền."),
            ("l24", "l15", 960, "u_ai_06", "Ủng hộ chấm điểm bảo vệ đồ án bằng cách hỏi trực tiếp từng dòng code và bắt sửa bug tại chỗ."),
            ("l25", "l24", 1000, "u_ai_03", "Đồng ý"),
            ("l26", "l22", 1040, "u_ai_19", "Chưa hợp lý nếu không giảm bớt các bài tập gõ lại CRUD lặp đi lặp lại nhàm chán."),
            ("l27", "l23", 1080, "u_ai_15", "Thôi bỏ qua mấy thành phần hổ báo đi, quay lại chuyện lương junior IT giảm vì cung vượt cầu cũng đáng lo."),
            ("l28", "l27", 1120, "u_ai_20", "Đúng rồi, AI nâng chuẩn đầu vào: fresher bây giờ phải nắm cả hệ thống, kiểm thử và tích hợp AI."),
            ("l29", "l25", 1160, "u_ai_11", "Vấn đáp trực tiếp 1-1 với giảng viên là cách hiệu quả nhất để phân loại sinh viên hiểu thật hay copy AI."),
            ("l30", f"t3_{post_id}", 1200, "u_ai_21", "Cho hỏi các bạn sinh viên tự học mô hình đồ thị và NLP hiện nay hay dùng tài nguyên nào miễn phí?"),
            ("l31", "l30", 1240, "u_ai_12", "Kaggle T4 x2 và Google Colab đủ chạy phần lớn mô hình vừa, kết hợp Hugging Face mã nguồn mở."),
            ("l32", "l28", 1280, "u_ai_10", "Không đồng ý chuyện AI làm giảm lương ngay đâu, chủ yếu do thị trường outsourcing chững lại từ năm ngoái."),
            ("l33", "l32", 1320, "u_ai_20", "Cũng có lý, nhưng rõ ràng doanh nghiệp đang cắt bớt tuyển thực tập sinh đại trà vì công việc đơn giản đã có AI."),
            ("l34", "l29", 1360, "u_ai_08", "Hợp lý, đổi hình thức thi từ trắc nghiệm học vẹt sang bảo vệ sản phẩm thực tế có lịch sử commit Git."),
            ("l35", "l33", 1400, "u_ai_14", "Đồ ngu mới tin outsourcing tự hồi phục mà không đổi mới nâng cấp lên kỹ sư giải pháp AI."),
            ("l36", "l35", 1440, "u_ai_10", "Nói năng cho đàng hoàng đi bạn, không đồng quan điểm kinh tế vĩ mô thì đưa số liệu ra phản biện."),
            ("l37", "l34", 1480, "u_ai_01", "Ý tưởng chấm theo lịch sử commit Git + kiểm thử tự động rất chuẩn kỹ thuật doanh nghiệp."),
            ("l38", "l31", 1520, "u_ai_21", "Tuyệt vời, cảm ơn bác đã chia sẻ kinh nghiệm dùng GPU đám mây cho sinh viên tự nghiên cứu."),
            ("l39", "l37", 1560, "u_ai_07", "Tán thành phương án kết hợp đánh giá Git + vấn đáp; vừa rèn thói quen làm việc nhóm vừa chống sao chép."),
            ("l40", "l36", 1600, "u_ai_18", "Tranh luận đến đây rõ 2 luồng rồi: siết nền tảng năm đầu và mở công cụ AI có kiểm chứng từ năm ba."),
            ("l41", "l40", 1640, "u_ai_06", "Hợp lý, cân bằng giữa tư duy thuật toán cốt lõi và năng lực làm chủ công cụ AI."),
            ("l42", "l39", 1680, "u_ai_03", "Mong khoa CNTT các trường sớm ban hành quy chế minh bạch để giảng viên và sinh viên thống nhất."),
        ]
    elif preset_id == "preset_junk_2":
        post_id = "sandbox_junk_2"
        post = {
            "id": post_id,
            "title": "[Sandbox Preset 2] Tranh luận bỏ cơ chế hybrid, bắt buộc làm việc tại văn phòng 5 ngày/tuần",
            "selftext": (
                "Doanh nghiệp vừa thông báo nhân sự khối kỹ thuật phải quay lại văn phòng đủ 5 ngày/tuần "
                "từ tháng tới để đồng bộ tiến độ dự án mới."
            ),
            "subreddit": "vozforums",
            "author": "op_sb2",
            "created_utc": base_ts,
        }
        spec = [
            ("s01", f"t3_{post_id}", 30, "u_1", "Đồng ý với ban giám đốc ở giai đoạn khởi động dự án mới vì phối hợp trực tiếp giảm sai sót."),
            ("s02", "s01", 60, "u_junk1", "lót dép"),
            ("s03", "s02", 90, "u_2", "+1"),
            ("s04", "s03", 120, "u_3", "Tuy nhiên chỉ nên áp dụng trong 2 tháng đầu, sau đó giữ hybrid 3 ngày là hợp lý."),
            ("s05", f"t3_{post_id}", 150, "[removed]", "[removed]"),
            ("s06", "s05", 180, "u_4", "Không đồng ý với quyết định cào bằng, nhân viên ở xa tốn gần 2 tiếng kẹt xe mỗi ngày."),
            ("s07", "s06", 210, "u_5", "Phản đối"),
            ("s08", "s01", 240, "u_junk2", "chấm"),
            ("s09", "s04", 270, "u_junk3", "👏"),
            ("s10", "s06", 300, "u_6", "Đồ ngu mới không thấy bất cập kẹt xe giờ cao điểm, ra quy định quá cứng nhắc!"),
            ("s11", "s10", 330, "u_1", "Bạn góp ý ôn hòa thôi, mình cũng ủng hộ trợ cấp đi lại nếu bắt buộc lên văn phòng đủ tuần."),
            ("s12", "s07", 360, "u_7", "Cho hỏi công ty có giữ linh hoạt giờ vào làm để tránh khung giờ tắc đường không?"),
            ("s13", "s12", 390, "u_4", "Chưa có thông báo chính thức, phòng nhân sự mới gửi khảo sát thu thập ý kiến thêm."),
        ]
    else:
        post_id = "sandbox_junk_1"
        post = {
            "id": post_id,
            "title": "[Sandbox Preset 1] Đề xuất lắp camera AI phạt nguội xe máy vượt đèn đỏ tại ngã tư lớn",
            "selftext": (
                "Sở Giao thông đề xuất thí điểm camera AI tự động nhận diện biển số xe máy vượt đèn đỏ "
                "và gửi thông báo phạt nguội qua ứng dụng định danh."
            ),
            "subreddit": "vozforums",
            "author": "op_sb1",
            "created_utc": base_ts,
        }
        spec = [
            ("c01", f"t3_{post_id}", 30, "u_an", "Ủng hộ mạnh, có camera tự động thì ý thức chấp hành đèn đỏ sẽ cải thiện rõ rệt."),
            ("c02", "c01", 60, "[deleted]", "[deleted]"),
            ("c03", "c02", 90, "u_binh", "Đồng ý"),
            ("c04", "c03", 120, "u_chi", "Chưa hợp lý nếu không chuẩn hóa vạch dừng và đồng bộ tín hiệu đếm giây trước khi phạt tự động."),
            ("c05", f"t3_{post_id}", 150, "u_dung", "hóng ké"),
            ("c06", "c05", 180, "u_giang", "Không đồng ý với mức phạt cào bằng, cần xem lại hạ tầng ngã tư và thời gian đèn vàng trước."),
            ("c07", "c06", 210, "u_ha", "Phản đối"),
            ("c08", f"t3_{post_id}", 240, "u_khanh", "."),
            ("c09", "c08", 270, "u_linh", "Đồng ý với góc nhìn hạ tầng; tuy nhiên vẫn nên thí điểm ở 5 trục chính trước."),
            ("c10", "c01", 300, "u_minh", "hóng"),
            ("c11", "c04", 330, "u_nam", "👍"),
            ("c12", "c09", 360, "u_spam", "https://shope.ee/khuyenmai999"),
            ("c13", "c04", 390, "u_oanh", "Cho mình hỏi thời hạn gửi thông báo phạt nguội qua VNeID là bao lâu để người dân kịp đối soát?"),
            ("c14", "c06", 420, "u_phuc", "Cứ phạt nặng cho chừa đi, toàn đồ ngu vượt đèn đỏ gây tai nạn cho người khác!"),
            ("c15", "c14", 450, "u_chi", "Nhầm rồi bác, cần phân biệt lỗi cố ý với tình huống nhường xe cứu thương giữa giao lộ."),
        ]

    comments = [
        {
            "fullname": f"t1_{cid}",
            "id": cid,
            "link_id": f"t3_{post_id}",
            "parent_id": pid if pid.startswith(("t3_", "t1_")) else f"t1_{pid}",
            "body": body,
            "text": body,
            "created_utc": base_ts + dt,
            "author": None if author in {"[deleted]", "[removed]"} else author,
            "author_raw": None if author in {"[deleted]", "[removed]"} else author,
            "author_hash": hash_author_sha256(None if author in {"[deleted]", "[removed]"} else author),
        }
        for cid, pid, dt, author, body in spec
    ]
    return post, comments


def parse_sandbox_raw_text(raw_text: str, post_id: str, base_ts: float) -> list[dict[str, Any]]:
    """Parse pasted multi-line Vietnamese thread text into reply records cleanly."""
    lines = [ln.rstrip("\r\n") for ln in str(raw_text or "").splitlines() if ln.strip()]
    comments: list[dict[str, Any]] = []
    indent_stack: list[tuple[int, str]] = [(-1, f"t3_{post_id}")]
    known_ids: set[str] = {f"t3_{post_id}"}

    for idx, line in enumerate(lines, start=1):
        cid = f"m{idx}"
        parent_ref = f"t3_{post_id}"
        body = line.strip()

        m_exp = _EXPLICIT_REPLY_RE.match(line)
        if m_exp:
            cid = m_exp.group(1).removeprefix("t1_")
            raw_parent = m_exp.group(2).strip()
            body = m_exp.group(3).strip()
            if raw_parent.lower() in {"root", post_id, f"t3_{post_id}"}:
                parent_ref = f"t3_{post_id}"
            else:
                clean_p = raw_parent.removeprefix("t1_")
                parent_ref = f"t1_{clean_p}" if f"t1_{clean_p}" in known_ids else f"t3_{post_id}"
        else:
            m_brk = _BRACKET_PREFIX_RE.match(line)
            if m_brk:
                tag = m_brk.group(1).strip()
                body = m_brk.group(2).strip()
                m_arrow = _BRACKET_ARROW_RE.match(tag)
                if m_arrow:
                    cid = m_arrow.group(1).removeprefix("t1_")
                    raw_p = m_arrow.group(2).strip()
                    if raw_p.lower() in {"root", post_id, f"t3_{post_id}"}:
                        parent_ref = f"t3_{post_id}"
                    else:
                        clean_p = raw_p.removeprefix("t1_")
                        parent_ref = f"t1_{clean_p}" if f"t1_{clean_p}" in known_ids else f"t3_{post_id}"
                else:
                    cid = re.sub(r"[^A-Za-z0-9_]", "", tag.removeprefix("t1_")) or f"m{idx}"
                    leading = len(line) - len(line.lstrip(" \t"))
                    while len(indent_stack) > 1 and indent_stack[-1][0] >= leading:
                        indent_stack.pop()
                    parent_ref = indent_stack[-1][1]
            else:
                expanded = line.expandtabs(2)
                leading = len(expanded) - len(expanded.lstrip(" "))
                body = expanded.strip()
                if body.startswith(("- ", "* ")):
                    body = body[2:].strip()
                while len(indent_stack) > 1 and indent_stack[-1][0] >= leading:
                    indent_stack.pop()
                parent_ref = indent_stack[-1][1]

        fullname = f"t1_{cid}"
        if fullname in known_ids:
            cid = f"{cid}_{idx}"
            fullname = f"t1_{cid}"
        known_ids.add(fullname)
        expanded_indent = len(line.expandtabs(2)) - len(line.expandtabs(2).lstrip(" "))
        indent_stack.append((expanded_indent, fullname))

        author = None if body.lower() in {"[deleted]", "[removed]"} else f"custom_u_{idx}"
        comments.append(
            {
                "fullname": fullname,
                "id": cid,
                "link_id": f"t3_{post_id}",
                "parent_id": parent_ref,
                "body": body,
                "text": body,
                "created_utc": base_ts + idx * 45.0,
                "author": author,
                "author_raw": author,
                "author_hash": hash_author_sha256(author),
            }
        )
    return comments


def _format_live_result_payload(
    *,
    cascade_id: str,
    topic_id: str,
    post: Mapping[str, Any],
    res: Mapping[str, Any],
    is_synthetic_fallback: bool,
) -> dict[str, Any]:
    """Convert `predict_live_cascade` output into unified Tab-1 visualization JSON."""
    raw_nodes = list(res.get("kept_nodes") or [])
    raw_edges = list(res.get("kept_edges") or [])
    trigger_edges_raw = list(res.get("top_trigger_edges") or [])

    trigger_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for item in trigger_edges_raw:
        pair = item.get("pair")
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            trigger_by_pair[(str(pair[0]), str(pair[1]))] = dict(item)

    edge_by_dst: dict[str, Mapping[str, Any]] = {str(e["target_id"]): e for e in raw_edges}

    tree_nodes: list[dict[str, Any]] = []
    node_text_by_id: dict[str, str] = {}
    support_cnt = oppose_cnt = neutral_cnt = 0
    authors_seen: set[str] = set()

    for idx, n in enumerate(raw_nodes):
        nid = str(n["node_id"])
        is_root = idx == 0 or n.get("parent_id") is None
        short_ah, _ = _safe_short_author_hash(n.get("author_hash"), idx, is_root=is_root)
        authors_seen.add(short_ah)
        node_text_by_id[nid] = str(n.get("text") or "")

        if is_root:
            tree_nodes.append(
                {
                    "node_id": nid,
                    "parent_id": None,
                    "is_root": True,
                    "author_hash": short_ah,
                    "depth": 0,
                    "time_delay_sec": 0.0,
                    "text": str(n.get("text") or ""),
                    "stance_class": "root",
                    "behavior_label": "root",
                    "s_uv": 0.0,
                    "h_uv": 0.0,
                    "gamma_td": 0.0,
                    "is_uncivil": False,
                    "relinked": False,
                }
            )
            continue

        pid = str(n.get("parent_id"))
        e_meta = edge_by_dst.get(nid, {})
        s_uv = float(e_meta.get("s_uv", 0.0))
        h_uv = float(e_meta.get("h_uv", 0.0))
        stance_cls = str(e_meta.get("stance_class") or _stance_class_from_score(s_uv))
        if stance_cls == "support":
            support_cnt += 1
        elif stance_cls == "oppose":
            oppose_cnt += 1
        else:
            neutral_cnt += 1
        trig = trigger_by_pair.get((pid, nid))
        gamma_val = float(trig["gamma_td"]) if trig is not None and "gamma_td" in trig else None
        tree_nodes.append(
            {
                "node_id": nid,
                "parent_id": pid,
                "is_root": False,
                "author_hash": short_ah,
                "depth": int(n.get("depth", 1)),
                "time_delay_sec": float(n.get("time_delay_sec", 0.0)),
                "text": str(n.get("text") or ""),
                "stance_class": stance_cls,
                "behavior_label": str(e_meta.get("behavior_label") or _behavior_from_2d(s_uv, h_uv)),
                "s_uv": round(s_uv, 4),
                "h_uv": round(h_uv, 4),
                "gamma_td": round(gamma_val, 4) if gamma_val is not None else None,
                "is_uncivil": h_uv >= 0.5,
                "relinked": bool(e_meta.get("relinked", False)),
            }
        )

    tree_edges: list[dict[str, Any]] = []
    for e in raw_edges:
        src = str(e["source_id"])
        dst = str(e["target_id"])
        s_uv = float(e.get("s_uv", 0.0))
        h_uv = float(e.get("h_uv", 0.0))
        trig = trigger_by_pair.get((src, dst))
        gamma_val = float(trig["gamma_td"]) if trig is not None and "gamma_td" in trig else None
        tree_edges.append(
            {
                "source_id": src,
                "target_id": dst,
                "stance_class": str(e.get("stance_class") or _stance_class_from_score(s_uv)),
                "behavior_label": str(e.get("behavior_label") or _behavior_from_2d(s_uv, h_uv)),
                "s_uv": round(s_uv, 4),
                "h_uv": round(h_uv, 4),
                "gamma_td": round(gamma_val, 4) if gamma_val is not None else None,
                "is_uncivil": h_uv >= 0.5,
                "relinked": bool(e.get("relinked", False)),
            }
        )

    enriched_triggers: list[dict[str, Any]] = []
    for trig in trigger_edges_raw:
        pair = trig.get("pair")
        src_p = str(pair[0]) if isinstance(pair, (list, tuple)) and len(pair) == 2 else ""
        dst_p = str(pair[1]) if isinstance(pair, (list, tuple)) and len(pair) == 2 else ""
        s_v = float(trig.get("s_uv", 0.0))
        h_v = float(trig.get("h_uv", 0.0))
        enriched_triggers.append(
            {
                "edge_idx": int(trig.get("edge_idx", 0)),
                "source_id": src_p,
                "target_id": dst_p,
                "importance": round(float(trig.get("importance", 0.0)), 4),
                "gamma_td": round(float(trig.get("gamma_td", 0.0)), 4),
                "s_uv": round(s_v, 4),
                "h_uv": round(h_v, 4),
                "stance_class": _stance_class_from_score(s_v),
                "is_uncivil": h_v >= 0.5,
                "reply_text_snippet": _clean_title_snippet(node_text_by_id.get(dst_p, ""), 110),
            }
        )

    root_full_text = str(tree_nodes[0]["text"]) if tree_nodes else str(post.get("title") or "")
    title = str(post.get("title") or _clean_title_snippet(root_full_text, 120))
    act_comp = res.get("actual_post_hoc_comparison") or {}
    act_replies = int(act_comp.get("actual_total_replies", max(0, len(tree_nodes) - 1)))
    raw_act_polar = act_comp.get("actual_full_polarization")
    act_polar: float | None = round(float(raw_act_polar), 4) if raw_act_polar is not None else None
    raw_early_polar = res.get("early_y_polar")
    early_polar: float | None = round(float(raw_early_polar), 4) if raw_early_polar is not None else None
    raw_conf = res.get("confidence_score")
    conf_val: float | None = round(float(raw_conf), 4) if raw_conf is not None else None
    unc_avail = bool(res.get("uncertainty_available", raw_conf is not None))
    raw_bc = res.get("behavior_counts") or {}
    behavior_counts = {k: int(raw_bc.get(k, 0)) for k in _DEMO_2D_BEHAVIOR_KEYS}
    has_future = bool(act_comp.get("has_future_replies", False))
    act_mode = str(
        act_comp.get("actual_full_polarization_mode")
        or ("future_only" if has_future else "no_future_replies")
    )
    user_graph = _build_thread_user_graph(tree_nodes, tree_edges)
    kept_texts_short = [n["text"] for n in tree_nodes[1:] if len(n["text"].split()) <= 3]

    return {
        "meta": {
            "cascade_id": cascade_id,
            "topic_id": topic_id,
            "lang": str(res.get("lang") or "vi"),
            "root_title": title,
            "root_text": root_full_text,
            "analysis_mode": str(res.get("analysis_mode") or "early"),
            "is_synthetic_fallback": bool(is_synthetic_fallback),
            "checkpoint_loaded": bool(res.get("checkpoint_loaded", False)),
            "cross_encoder_loaded": bool(res.get("cross_encoder_loaded", False)),
            "nlp_backend": str(res.get("nlp_backend") or "heuristic_hash_sandbox"),
            "junk_mode": str(res.get("junk_mode") or "contract"),
            "author_count": len(authors_seen),
        },
        "forecast": {
            "gnn_k_horizon": int(res.get("gnn_k_horizon", res.get("k_observed", max(0, len(tree_nodes) - 1)))),
            "forecast_scope": str(res.get("forecast_scope") or "early_k_in_distribution"),
            "k_observed": int(res.get("k_observed", max(0, len(tree_nodes) - 1))),
            "predicted_total_replies": round(float(res.get("predicted_total_replies", 0.0)), 1),
            "actual_total_replies": act_replies,
            "has_future_replies": has_future,
            "predicted_polarization_score": round(float(res.get("predicted_polarization_score", 0.0)), 4),
            "actual_y_polar": act_polar,
            "actual_full_polarization_mode": act_mode,
            "early_y_polar": early_polar,
            "predicted_regime": str(res.get("predicted_regime") or ""),
            "confidence_score": conf_val,
            "uncertainty_available": unc_avail,
            "low_signal_warning": bool(res.get("low_signal_warning", False)),
            "behavior_counts": behavior_counts,
            "stance_counts": {
                "support": support_cnt,
                "oppose": oppose_cnt,
                "neutral": neutral_cnt,
            },
            "early_burst_metrics": dict(res.get("early_burst_metrics") or {}),
            "regime_thresholds": dict(res.get("regime_thresholds") or {"q50": VN_POLAR_Q50, "q75": VN_POLAR_Q75, "q90": VN_POLAR_Q90}),
        },
        "tree": {
            "nodes": tree_nodes,
            "edges": tree_edges,
        },
        "user_graph": user_graph,
        "top_trigger_edges": enriched_triggers,
        "topic_analysis": dict(res.get("topic_analysis") or {}),
        "large_thread_analysis": dict(res.get("large_thread_analysis") or {}),
        "junk_filter_stats": dict(res.get("junk_filter_stats") or {}),
        "preserved_short_stances": kept_texts_short,
        "kept_nodes": list(res.get("kept_nodes") or []),
        "kept_edges": list(res.get("kept_edges") or []),
    }


def _cascade_safety_flags(d: Any) -> tuple[bool, bool]:
    """Return `(has_redacted, has_nsfw)` by inspecting cascade node texts."""
    texts: list[str] = [str(t) for t in (getattr(d, "node_texts", None) or ()) if t is not None]
    for attr in ("root_text", "text", "cascade_id"):
        val = getattr(d, attr, None)
        if val is not None:
            texts.append(str(val))
    joined = "\n".join(texts)
    has_redacted = bool(_REDACTED_MARKER_RE.search(joined))
    has_nsfw = bool(_NSFW_MARKER_RE.search(joined))
    return has_redacted, has_nsfw


def _select_diverse_indices(cascades: Sequence[Any], count: int = 10) -> list[int]:
    """Select `count` deterministic diverse clean VN Test cascades spanning low->high polarization."""
    import torch

    clean_indices: list[int] = []
    non_redacted_indices: list[int] = []
    for idx, d in enumerate(cascades):
        has_redacted, has_nsfw = _cascade_safety_flags(d)
        if not has_redacted:
            non_redacted_indices.append(idx)
            if not has_nsfw:
                clean_indices.append(idx)

    eligible_indices = clean_indices if clean_indices else non_redacted_indices
    if not eligible_indices:
        return []

    meta_rows = []
    for idx in eligible_indices:
        d = cascades[idx]
        y_pol = float(d.y_polar.view(-1)[0].item())
        n_rep = int(round(float(torch.expm1(d.y_size.view(-1)[0]).item())))
        s_vals = d.edge_attr[:, 0] if d.edge_attr.numel() > 0 else torch.zeros(0)
        sup = int((s_vals > 0.1).sum().item())
        opp = int((s_vals < -0.1).sum().item())
        h_vals = d.edge_attr[:, 1] if (d.edge_attr.numel() > 0 and d.edge_attr.size(1) >= 2) else torch.zeros(0)
        unc = int((h_vals >= 0.5).sum().item())
        meta_rows.append(
            {
                "idx": idx,
                "cid": str(d.cascade_id),
                "y_polar": y_pol,
                "n_rep": n_rep,
                "sup": sup,
                "opp": opp,
                "unc": unc,
            }
        )

    meta_by_idx = {int(r["idx"]): r for r in meta_rows}
    if len(eligible_indices) <= count:
        return [
            int(r["idx"])
            for r in sorted(meta_rows, key=lambda r: (r["y_polar"], r["n_rep"], r["cid"]))
        ]

    by_polar = sorted(meta_rows, key=lambda r: (r["y_polar"], r["n_rep"], r["cid"]))
    by_size = sorted(meta_rows, key=lambda r: (-r["n_rep"], -r["y_polar"], r["cid"]))
    chosen: list[int] = []
    seen: set[int] = set()

    def _pick(candidates: Sequence[Mapping[str, Any]]) -> None:
        for row in candidates:
            if row["idx"] not in seen:
                chosen.append(int(row["idx"]))
                seen.add(int(row["idx"]))
                return

    # 1. Largest-reach debate and 2nd largest-reach thread
    _pick([r for r in by_size if r["sup"] >= 1 and r["opp"] >= 1] or by_size)
    _pick(by_size)

    # 2. Highest-polarization & Q90 conflict thread
    _pick([r for r in reversed(by_polar) if r["opp"] >= 3 and r["sup"] >= 1] or list(reversed(by_polar)))
    _pick(list(reversed(by_polar)))

    # 3. Lowest-polarization consensus thread
    _pick(by_polar)

    # 4. Quantile sweep filling remaining slots evenly across polarization distribution
    n_el = len(meta_rows)
    quantiles = (0.12, 0.25, 0.40, 0.55, 0.72, 0.85, 0.93, 0.33, 0.65)
    for q in quantiles:
        if len(chosen) >= count:
            break
        target_pos = min(n_el - 1, max(0, int(round(q * (n_el - 1)))))
        window = sorted(
            by_polar,
            key=lambda r: (
                abs(by_polar.index(r) - target_pos),
                0 if (r["sup"] >= 1 and r["opp"] >= 1) else 1,
                r["cid"],
            ),
        )
        _pick(window)

    while len(chosen) < count:
        _pick(by_polar)

    # Order cleanly from low -> high polarization for intuitive browsing
    chosen.sort(key=lambda idx_i: (meta_by_idx[idx_i]["y_polar"], meta_by_idx[idx_i]["n_rep"], meta_by_idx[idx_i]["cid"]))
    return chosen[:count]


class DemoBackend:
    """Thread-safe local inference, data, and paper-metrics backend for Demo Web MVP."""

    def __init__(
        self,
        *,
        data_dir: Path = Path("data/processed/vn"),
        ckpt_path: Path = Path("checkpoints/best_sbifagat_vn.pt"),
        ce_dir: Path = Path("checkpoints/dual_ce_vn"),
        paper_dir: Path = Path("results"),
        web_dir: Path = Path("web_demo"),
        preload: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.ckpt_path = Path(ckpt_path)
        self.ce_dir = Path(ce_dir)
        self.paper_dir = Path(paper_dir)
        self.web_dir = Path(web_dir)

        self._lock = threading.Lock()
        self._ce_lock = threading.Lock()
        self._gnn_model: SBiFAGAT | None = None
        self._gnn_ckpt_loaded = False
        self._ce_model: Any = None
        self._ce_tok: Any = None
        self._ce_load_attempted = False
        self._threads_ready = False
        self._is_synthetic_fallback = False
        self._thread_summaries: list[dict[str, Any]] = []
        self._threads_by_id: dict[str, dict[str, Any]] = {}
        self._sandbox_preset_cache: dict[tuple[str, ...], dict[str, Any]] = {}
        self._macro_cache: dict[str, Any] | None = None

        self._try_load_gnn_checkpoint()
        if preload:
            self._ensure_tier1_threads_loaded()

    @property
    def real_vn_data_available(self) -> bool:
        return self.data_dir.is_dir() and all(
            (self.data_dir / fname).is_file() for fname in _REQUIRED_VN_FILES
        )

    @property
    def live_cross_encoder_available(self) -> bool:
        return self.ce_dir.is_dir() and all(
            (self.ce_dir / fname).is_file() for fname in _REQUIRED_CE_FILES
        )

    def _try_load_gnn_checkpoint(self) -> None:
        if not self.ckpt_path.is_file():
            self._gnn_model = None
            self._gnn_ckpt_loaded = False
            return
        try:
            import torch

            payload = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
            cfg = payload.get("config", {})
            state_dict = payload["model_state_dict"]
            has_persist = (
                bool(cfg["use_polar_persistence_anchor"])
                if "use_polar_persistence_anchor" in cfg
                else ("head.persist_weight" in state_dict)
            )
            model = SBiFAGAT(
                in_dim=int(payload.get("in_dim", 770)),
                hidden_dim=int(cfg.get("hidden_dim", 64)),
                edge_dim=int(cfg.get("edge_dim", 2)),
                num_rumor_classes=int(payload.get("num_rumor_classes", 4)),
                eps=float(cfg.get("eps", 0.5)),
                dropout=float(cfg.get("dropout", 0.1)),
                use_cascade_stats=bool(cfg.get("use_cascade_stats", True)),
                predict_uncertainty=bool(cfg.get("predict_uncertainty", False)),
                use_polar_persistence_anchor=has_persist,
            )
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            self._gnn_model = model
            self._gnn_ckpt_loaded = True
        except Exception:
            self._gnn_model = None
            self._gnn_ckpt_loaded = False

    def _get_cached_cross_encoder(self) -> tuple[Any, Any] | None:
        """Load and cache `DualHeadCrossEncoder` once per backend instance."""
        with self._ce_lock:
            if self._ce_load_attempted:
                return (self._ce_model, self._ce_tok) if (self._ce_model is not None and self._ce_tok is not None) else None
            self._ce_load_attempted = True
            if not self.live_cross_encoder_available:
                return None
            try:
                import torch
                from src.nlp.dual_edge_scorer import DualHeadCrossEncoder

                ce_map_loc = "cuda" if torch.cuda.is_available() else "cpu"
                ce_model, ce_tok = DualHeadCrossEncoder.load_checkpoint(self.ce_dir, map_location=ce_map_loc)
                ce_model.eval()
                self._ce_model = ce_model
                self._ce_tok = ce_tok
                return (ce_model, ce_tok)
            except Exception:
                return None

    def get_health(self) -> dict[str, Any]:
        real_avail = self.real_vn_data_available
        live_ce = self.live_cross_encoder_available
        if live_ce:
            nlp_backend = "full_2d_cross_encoder"
        elif real_avail and self._gnn_ckpt_loaded:
            nlp_backend = "preextracted_full_2d_ce"
        else:
            nlp_backend = "heuristic_hash_sandbox"

        notes: list[str] = [VN_QUANTILE_CALIBRATION_NOTE]
        if real_avail and self._gnn_ckpt_loaded:
            notes.append(
                "Các bài Test VN thật sử dụng nhúng ViSoBERT SpanPool và cạnh 2D [s_uv, h_uv] "
                "từ DualHeadCrossEncoder đã trích xuất sẵn trong data/processed/vn/nlp_features.pt."
            )
        else:
            notes.append(
                "Không tìm thấy data/processed/vn/ hoặc checkpoint VN cục bộ; Tab 1 đang dùng 2 mẫu tổng hợp ngoại tuyến."
            )
        if not live_ce:
            notes.append(
                "Trọng số DualHeadCrossEncoder (dual_head_weights.pt) không có sẵn cục bộ: "
                "chế độ Sandbox dùng bộ chấm Heuristic để minh họa Path Contraction và định tuyến cây."
            )

        return {
            "status": "ok",
            "real_vn_data_available": real_avail,
            "gnn_checkpoint_loaded": self._gnn_ckpt_loaded,
            "cross_encoder_loaded": live_ce,
            "preextracted_2d_ce_available": bool(real_avail and self._gnn_ckpt_loaded),
            "nlp_backend": nlp_backend,
            "regime_thresholds": {
                "q50": VN_POLAR_Q50,
                "q75": VN_POLAR_Q75,
                "q90": VN_POLAR_Q90,
            },
            "notes": notes,
        }

    def _build_real_vn_test_cache(self) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
        import torch

        assert self._gnn_model is not None
        splits = load_topic_splits_json(self.data_dir / "topic_splits.json")
        nodes_tbl = pq.read_table(self.data_dir / "nodes.parquet")
        edges_tbl = pq.read_table(self.data_dir / "edges.parquet")
        test_topics = sorted(splits["test"])
        nt_test = nodes_tbl.filter(
            pc.is_in(nodes_tbl["topic_id"], value_set=pa.array(test_topics, type=pa.string()))
        )
        cids_test = pc.unique(nt_test["cascade_id"])
        et_test = edges_tbl.filter(pc.is_in(edges_tbl["cascade_id"], value_set=cids_test))

        nlp_pt = torch.load(self.data_dir / "nlp_features.pt", map_location="cpu", weights_only=False)
        dataset = EarlyCascadeDataset(
            self.data_dir,
            k_horizon=20,
            split="test",
            min_total_replies=25,
            polar_target_mode="future_only",
            nodes_table=nt_test,
            edges_table=et_test,
            nlp_features=nlp_pt,
            custom_topic_splits=splits,
            edge_source="auto",
        )
        selected_indices = _select_diverse_indices(dataset.data_list, count=10)
        selected_cids = {str(dataset.data_list[i].cascade_id) for i in selected_indices}

        node_meta_by_cascade: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in nt_test.to_pylist():
            cid = str(row["cascade_id"])
            if cid in selected_cids:
                node_meta_by_cascade[cid][str(row["node_id"])] = row

        edge_meta_by_cascade: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in et_test.to_pylist():
            cid = str(row["cascade_id"])
            if cid in selected_cids:
                edge_meta_by_cascade[cid].append(row)

        pt_stance = nlp_pt.get("edge_stance") or nlp_pt.get("teacher_edge_stance") or {}
        pt_hostility = nlp_pt.get("edge_hostility") or nlp_pt.get("teacher_edge_hostility") or {}
        del nlp_pt

        cached_ce = self._get_cached_cross_encoder()
        topic_enc = cached_ce[0].encoder if cached_ce is not None else None
        topic_tok = cached_ce[1] if cached_ce is not None else None
        topic_backend_choice: Literal["embedding", "lexical"] = "embedding" if cached_ce is not None else "lexical"

        summaries: list[dict[str, Any]] = []
        threads_by_id: dict[str, dict[str, Any]] = {}

        for idx_ds in selected_indices:
            d = dataset.data_list[idx_ds]
            cid = str(d.cascade_id)
            tid = str(d.topic_id)
            diag = self._gnn_model.predict_with_diagnostics(d, top_k_edges=5)
            gates = self._gnn_model.extract_edge_gates(d)

            gamma_td_list = gates["gamma_td"].detach().cpu().tolist()
            edge_attr_cpu = d.edge_attr.detach().cpu()
            edge_pairs = list(getattr(d, "edge_pairs", []))
            raw_meta = node_meta_by_cascade.get(cid, {})

            root_id = str(d.node_ids[0])
            root_meta = raw_meta.get(root_id, {})
            root_ts = root_meta.get("created_at_utc")

            # Compute exact BFS depth within kept V_K
            children_adj: dict[str, list[str]] = defaultdict(list)
            edge_info_by_dst: dict[str, dict[str, Any]] = {}
            behavior_counts = {lbl: 0 for lbl in _DEMO_2D_BEHAVIOR_KEYS}
            support_cnt = oppose_cnt = neutral_cnt = 0

            for e_i, (src_id, dst_id) in enumerate(edge_pairs):
                src_s, dst_s = str(src_id), str(dst_id)
                children_adj[src_s].append(dst_s)
                s_uv = float(edge_attr_cpu[e_i, 0].item()) if edge_attr_cpu.size(1) >= 1 else 0.0
                h_uv = float(edge_attr_cpu[e_i, 1].item()) if edge_attr_cpu.size(1) >= 2 else 0.0
                g_td = float(gamma_td_list[e_i]) if e_i < len(gamma_td_list) else 0.0
                stance_cls = _stance_class_from_score(s_uv)
                if stance_cls == "support":
                    support_cnt += 1
                elif stance_cls == "oppose":
                    oppose_cnt += 1
                else:
                    neutral_cnt += 1
                blbl = _behavior_from_2d(s_uv, h_uv)
                behavior_counts[blbl] = behavior_counts.get(blbl, 0) + 1
                edge_info_by_dst[dst_s] = {
                    "source_id": src_s,
                    "target_id": dst_s,
                    "s_uv": s_uv,
                    "h_uv": h_uv,
                    "gamma_td": g_td,
                    "stance_class": stance_cls,
                    "behavior_label": blbl,
                    "is_uncivil": h_uv >= 0.5,
                    "relinked": False,
                }

            depth_map: dict[str, int] = {root_id: 0}
            q: deque[str] = deque([root_id])
            while q:
                curr = q.popleft()
                for ch in children_adj.get(curr, []):
                    if ch not in depth_map:
                        depth_map[ch] = depth_map[curr] + 1
                        q.append(ch)

            tree_nodes: list[dict[str, Any]] = []
            node_text_by_id: dict[str, str] = {}
            authors_seen: set[str] = set()
            for n_idx, (nid_raw, n_txt_raw) in enumerate(zip(d.node_ids, d.node_texts)):
                nid = str(nid_raw)
                n_txt = str(n_txt_raw)
                node_text_by_id[nid] = n_txt
                n_meta = raw_meta.get(nid, {})
                is_root = n_idx == 0
                short_ah, _ = _safe_short_author_hash(
                    n_meta.get("user_id_hash"), n_idx, is_root=is_root
                )
                authors_seen.add(short_ah)
                dt_sec = 0.0
                n_ts = n_meta.get("created_at_utc")
                if not is_root and root_ts is not None and n_ts is not None:
                    try:
                        dt_sec = max(0.0, float((n_ts - root_ts).total_seconds()))
                    except Exception:
                        dt_sec = 0.0

                if is_root:
                    tree_nodes.append(
                        {
                            "node_id": nid,
                            "parent_id": None,
                            "is_root": True,
                            "author_hash": short_ah,
                            "depth": 0,
                            "time_delay_sec": 0.0,
                            "text": n_txt,
                            "stance_class": "root",
                            "behavior_label": "root",
                            "s_uv": 0.0,
                            "h_uv": 0.0,
                            "gamma_td": 0.0,
                            "is_uncivil": False,
                            "relinked": False,
                        }
                    )
                else:
                    e_inf = edge_info_by_dst.get(nid, {})
                    pid = str(e_inf.get("source_id") or n_meta.get("parent_id") or root_id)
                    tree_nodes.append(
                        {
                            "node_id": nid,
                            "parent_id": pid,
                            "is_root": False,
                            "author_hash": short_ah,
                            "depth": int(depth_map.get(nid, 1)),
                            "time_delay_sec": round(dt_sec, 1),
                            "text": n_txt,
                            "stance_class": str(e_inf.get("stance_class", "neutral")),
                            "behavior_label": str(e_inf.get("behavior_label", "neutral")),
                            "s_uv": round(float(e_inf.get("s_uv", 0.0)), 4),
                            "h_uv": round(float(e_inf.get("h_uv", 0.0)), 4),
                            "gamma_td": round(float(e_inf.get("gamma_td", 0.0)), 4),
                            "is_uncivil": bool(e_inf.get("is_uncivil", False)),
                            "relinked": False,
                        }
                    )

            tree_edges = [
                {
                    "source_id": str(src_id),
                    "target_id": str(dst_id),
                    "stance_class": edge_info_by_dst[str(dst_id)]["stance_class"],
                    "behavior_label": edge_info_by_dst[str(dst_id)]["behavior_label"],
                    "s_uv": round(float(edge_info_by_dst[str(dst_id)]["s_uv"]), 4),
                    "h_uv": round(float(edge_info_by_dst[str(dst_id)]["h_uv"]), 4),
                    "gamma_td": round(float(edge_info_by_dst[str(dst_id)]["gamma_td"]), 4),
                    "is_uncivil": bool(edge_info_by_dst[str(dst_id)]["is_uncivil"]),
                    "relinked": False,
                }
                for _, dst_id in edge_pairs
                for src_id in [edge_info_by_dst[str(dst_id)]["source_id"]]
            ]

            top_triggers: list[dict[str, Any]] = []
            for trig in diag["top_trigger_edges"]:
                pair = trig.get("pair")
                src_p = str(pair[0]) if isinstance(pair, (list, tuple)) and len(pair) == 2 else ""
                dst_p = str(pair[1]) if isinstance(pair, (list, tuple)) and len(pair) == 2 else ""
                s_v = float(trig.get("s_uv", 0.0))
                h_v = float(trig.get("h_uv", 0.0))
                top_triggers.append(
                    {
                        "edge_idx": int(trig.get("edge_idx", 0)),
                        "source_id": src_p,
                        "target_id": dst_p,
                        "importance": round(float(trig.get("importance", 0.0)), 4),
                        "gamma_td": round(float(trig.get("gamma_td", 0.0)), 4),
                        "s_uv": round(s_v, 4),
                        "h_uv": round(h_v, 4),
                        "stance_class": _stance_class_from_score(s_v),
                        "is_uncivil": h_v >= 0.5,
                        "reply_text_snippet": _clean_title_snippet(node_text_by_id.get(dst_p, ""), 110),
                    }
                )

            pred_polar = float(diag["y_polar_pred"].view(-1)[0].item())
            pred_replies = float(diag["predicted_total_replies"].view(-1)[0].item())
            raw_conf_t = diag.get("confidence_score")
            conf_score: float | None = (
                round(float(raw_conf_t.view(-1)[0].item()), 4)
                if raw_conf_t is not None
                else None
            )
            unc_avail = bool(diag.get("uncertainty_available", raw_conf_t is not None))
            act_replies = int(round(float(torch.expm1(d.y_size.view(-1)[0]).item())))
            act_polar = float(d.y_polar.view(-1)[0].item())
            early_polar = float(d.early_y_polar.view(-1)[0].item())
            has_future = act_replies > (len(tree_nodes) - 1)
            regime = _classify_calibrated_regime(pred_polar, behavior_counts)
            stats_vec = d.cascade_stats.view(-1).tolist()

            root_full_text = str(d.node_texts[0])
            title_snip = _clean_title_snippet(root_full_text, 120)
            user_graph = _build_thread_user_graph(tree_nodes, tree_edges)

            # Classify topic strictly on root + early replies in V_K (zero future leakage).
            # Reuse cached Transformer encoder (and NLI if locally cached) instead of dropping to lexical.
            topic_analysis = classify_post_topics(
                title_snip,
                root_full_text,
                reply_snippets=[n["text"] for n in tree_nodes[1:6]],
                lang="vi",
                backend=topic_backend_choice,
                node_encoder=topic_enc,
                node_tokenizer=topic_tok,
                allow_nli_download=False,
            )

            # Analyze full-thread trajectory across all real replies of the Test cascade
            full_nodes_dict = raw_meta or {str(n["node_id"]): n for n in tree_nodes}
            full_nodes_list = list(full_nodes_dict.values())
            full_edges_list = edge_meta_by_cascade.get(cid) or [
                {
                    "edge_id": f"{e['source_id']}->{e['target_id']}",
                    "source_id": e["source_id"],
                    "target_id": e["target_id"],
                    "s_uv": e["s_uv"],
                    "h_uv": e["h_uv"],
                }
                for e in tree_edges
            ]
            full_children_adj: dict[str, list[str]] = defaultdict(list)
            full_edge_scores: dict[str, dict[str, Any]] = {}
            for fe in full_edges_list:
                eid_f = str(fe.get("edge_id") or f"{fe['source_id']}->{fe['target_id']}")
                s_f = float(pt_stance.get(eid_f, fe.get("s_uv", 0.0)))
                h_f = float(pt_hostility.get(eid_f, fe.get("h_uv", 0.0)))
                full_children_adj[str(fe["source_id"])].append(str(fe["target_id"]))
                full_edge_scores[eid_f] = {"s_uv": s_f, "h_uv": h_f}

            full_depth_map: dict[str, int] = {root_id: 0}
            dq_full: deque[str] = deque([root_id])
            while dq_full:
                cur_f = dq_full.popleft()
                for ch_f in full_children_adj.get(cur_f, []):
                    if ch_f not in full_depth_map:
                        full_depth_map[ch_f] = full_depth_map[cur_f] + 1
                        dq_full.append(ch_f)

            imp_by_target = {str(t["target_id"]): float(t.get("importance", 0.0)) for t in top_triggers}
            root_meta_obj = raw_meta.get(root_id) or tree_nodes[0]
            large_thread_analysis = analyze_large_thread_dynamics(
                root_node=root_meta_obj,
                all_nodes=full_nodes_list,
                all_edges=full_edges_list,
                edge_scores=full_edge_scores,
                depth_map=full_depth_map,
                topic_analysis=topic_analysis,
                trigger_importance_by_target=imp_by_target,
                k_early_ref=20,
                lang="vi",
                scope="early_post_hoc",
            )

            summary = {
                "cascade_id": cid,
                "topic_id": tid,
                "title_snippet": title_snip,
                "primary_topic": topic_analysis.get("primary_topic", ""),
                "scale_tier": large_thread_analysis.get("scale_tier", "medium"),
                "k_observed": len(tree_nodes) - 1,
                "actual_total_replies": act_replies,
                "has_future_replies": has_future,
                "actual_y_polar": round(act_polar, 4),
                "actual_full_polarization_mode": "future_only",
                "early_y_polar": round(early_polar, 4),
                "predicted_total_replies": round(pred_replies, 1),
                "predicted_polarization_score": round(pred_polar, 4),
                "predicted_regime": regime,
                "confidence_score": conf_score,
                "uncertainty_available": unc_avail,
                "support_count": support_cnt,
                "oppose_count": oppose_cnt,
                "neutral_count": neutral_cnt,
                "author_count": len(authors_seen),
            }
            summaries.append(summary)

            threads_by_id[cid] = {
                "meta": {
                    "cascade_id": cid,
                    "topic_id": tid,
                    "root_title": title_snip,
                    "root_text": root_full_text,
                    "analysis_mode": "early",
                    "is_synthetic_fallback": False,
                    "checkpoint_loaded": True,
                    "cross_encoder_loaded": True,
                    "nlp_backend": "preextracted_full_2d_ce",
                    "author_count": len(authors_seen),
                },
                "forecast": {
                    "k_observed": len(tree_nodes) - 1,
                    "predicted_total_replies": round(pred_replies, 1),
                    "actual_total_replies": act_replies,
                    "has_future_replies": has_future,
                    "predicted_polarization_score": round(pred_polar, 4),
                    "actual_y_polar": round(act_polar, 4),
                    "actual_full_polarization_mode": "future_only",
                    "early_y_polar": round(early_polar, 4),
                    "predicted_regime": regime,
                    "confidence_score": conf_score,
                    "uncertainty_available": unc_avail,
                    "low_signal_warning": bool(diag["low_signal_warning"].view(-1)[0].item()),
                    "behavior_counts": behavior_counts,
                    "stance_counts": {
                        "support": support_cnt,
                        "oppose": oppose_cnt,
                        "neutral": neutral_cnt,
                    },
                    "early_burst_metrics": {
                        "elapsed_span_norm": stats_vec[0],
                        "mean_interarrival_norm": stats_vec[1],
                        "burst_acceleration": stats_vec[2],
                        "root_direct_ratio": stats_vec[3],
                        "max_depth_norm": stats_vec[4],
                        "unique_user_ratio": stats_vec[5],
                        "substantive_text_ratio": stats_vec[6],
                        "one_sided_hostility_index": stats_vec[7],
                    },
                    "regime_thresholds": {
                        "q50": VN_POLAR_Q50,
                        "q75": VN_POLAR_Q75,
                        "q90": VN_POLAR_Q90,
                    },
                },
                "tree": {
                    "nodes": tree_nodes,
                    "edges": tree_edges,
                },
                "user_graph": user_graph,
                "top_trigger_edges": top_triggers,
                "topic_analysis": topic_analysis,
                "large_thread_analysis": large_thread_analysis,
            }

        return summaries, threads_by_id

    def _build_synthetic_fallback_cache(self) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        presets = [
            ("vn_demo_01", "topic_demo_01", *_make_demo_offline_thread()),
            ("vn_demo_02", "topic_demo_02", *_make_synthetic_polarized_preset()),
        ]
        ce_ckpt = self.ce_dir if self.live_cross_encoder_available else None
        gnn_ckpt = self.ckpt_path if self.ckpt_path.is_file() else None

        summaries: list[dict[str, Any]] = []
        threads_by_id: dict[str, dict[str, Any]] = {}
        for cid, tid, post, comments in presets:
            res = predict_live_cascade(
                post,
                comments,
                k_horizon=20,
                cross_encoder_ckpt=ce_ckpt,
                gnn_ckpt=gnn_ckpt,
                lang="vi",
                junk_mode="contract",
                allow_untrained=True,
                is_demo=True,
            )
            payload = _format_live_result_payload(
                cascade_id=cid,
                topic_id=tid,
                post=post,
                res=res,
                is_synthetic_fallback=True,
            )
            sc = payload["forecast"]["stance_counts"]
            summaries.append(
                {
                    "cascade_id": cid,
                    "topic_id": tid,
                    "title_snippet": _clean_title_snippet(f"[Mẫu tổng hợp] {payload['meta']['root_title']}", 120),
                    "primary_topic": payload["topic_analysis"].get("primary_topic", ""),
                    "scale_tier": payload["large_thread_analysis"].get("scale_tier", "medium"),
                    "k_observed": payload["forecast"]["k_observed"],
                    "actual_total_replies": payload["forecast"]["actual_total_replies"],
                    "has_future_replies": payload["forecast"]["has_future_replies"],
                    "actual_y_polar": payload["forecast"]["actual_y_polar"],
                    "actual_full_polarization_mode": payload["forecast"]["actual_full_polarization_mode"],
                    "early_y_polar": payload["forecast"]["early_y_polar"],
                    "predicted_total_replies": payload["forecast"]["predicted_total_replies"],
                    "predicted_polarization_score": payload["forecast"]["predicted_polarization_score"],
                    "predicted_regime": payload["forecast"]["predicted_regime"],
                    "confidence_score": payload["forecast"]["confidence_score"],
                    "uncertainty_available": payload["forecast"]["uncertainty_available"],
                    "support_count": sc["support"],
                    "oppose_count": sc["oppose"],
                    "neutral_count": sc["neutral"],
                    "author_count": payload["meta"]["author_count"],
                }
            )
            threads_by_id[cid] = payload

        return summaries, threads_by_id

    def _ensure_tier1_threads_loaded(self) -> None:
        with self._lock:
            if self._threads_ready:
                return
            if self.real_vn_data_available and self._gnn_ckpt_loaded:
                try:
                    summaries, by_id = self._build_real_vn_test_cache()
                    self._is_synthetic_fallback = False
                except Exception:
                    summaries, by_id = self._build_synthetic_fallback_cache()
                    self._is_synthetic_fallback = True
            else:
                summaries, by_id = self._build_synthetic_fallback_cache()
                self._is_synthetic_fallback = True
            self._thread_summaries = summaries
            self._threads_by_id = by_id
            self._threads_ready = True

    def list_tier1_threads(self) -> dict[str, Any]:
        self._ensure_tier1_threads_loaded()
        return {
            "is_synthetic_fallback": self._is_synthetic_fallback,
            "count": len(self._thread_summaries),
            "threads": list(self._thread_summaries),
        }

    def get_tier1_thread(self, cascade_id: str) -> dict[str, Any]:
        self._ensure_tier1_threads_loaded()
        cid = str(cascade_id or "").strip()
        if not cid or cid not in self._threads_by_id:
            raise KeyError(f"Unknown cascade_id {cascade_id!r}")
        return self._threads_by_id[cid]

    def run_sandbox(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise ValueError("Payload JSON không hợp lệ.")
        mode = str(payload.get("mode") or "preset_junk_1").strip()
        junk_mode_raw = str(payload.get("junk_mode") or "contract").strip().lower()
        if junk_mode_raw not in {"contract", "prune"}:
            raise ValueError(f"junk_mode không hợp lệ: {junk_mode_raw!r}")
        junk_mode: Literal["contract", "prune"] = "contract" if junk_mode_raw == "contract" else "prune"

        default_amode = "full" if mode == "preset_large_1" else "early"
        amode_raw = str(payload.get("analysis_mode") or default_amode).strip().lower()
        if amode_raw not in {"early", "full"}:
            raise ValueError(f"analysis_mode không hợp lệ: {amode_raw!r}")
        analysis_mode: Literal["early", "full"] = "full" if amode_raw == "full" else "early"

        raw_k = payload.get("k_horizon")
        if raw_k is None:
            k_horizon = 20
        elif isinstance(raw_k, bool):
            raise ValueError(f"k_horizon không hợp lệ: {raw_k!r}")
        else:
            k_horizon = int(raw_k)
            if k_horizon < 1 or k_horizon > 500:
                raise ValueError(f"k_horizon phải nằm trong [1, 500], nhận {k_horizon!r}")

        raw_max_full = payload.get("max_full_comments")
        if raw_max_full is None:
            max_full_comments = 500
        elif isinstance(raw_max_full, bool):
            raise ValueError(f"max_full_comments không hợp lệ: {raw_max_full!r}")
        else:
            max_full_comments = int(raw_max_full)
            if max_full_comments < 20 or max_full_comments > 500:
                raise ValueError(
                    f"max_full_comments phải nằm trong [20, 500], nhận {max_full_comments!r}"
                )

        custom_topics = payload.get("custom_topics")
        topics_cache_key = json.dumps(custom_topics, ensure_ascii=False) if custom_topics is not None else ""
        lang_raw = str(payload.get("lang") or "vi").strip().lower()
        lang_val = "en" if lang_raw == "en" else "vi"

        base_ts = 1763400000.0
        if mode in {"preset_junk_1", "preset_junk_2", "preset_large_1"}:
            cache_key = (
                mode,
                junk_mode,
                analysis_mode,
                str(k_horizon),
                str(max_full_comments),
                topics_cache_key,
                lang_val,
            )
            with self._lock:
                cached = self._sandbox_preset_cache.get(cache_key)
            if cached is not None:
                return dict(cached)
            post, comments = build_sandbox_preset(mode)
            is_demo = True
        elif mode == "reddit_url":
            reddit_url = str(payload.get("reddit_url") or "").strip()
            post_id = _extract_post_id_from_url(reddit_url)
            post, comments = _fetch_reddit_thread(post_id)
            is_demo = False
        elif mode == "custom":
            post_id = "sandbox_custom"
            title = str(payload.get("title") or "").strip()
            root_text = str(payload.get("root_text") or "").strip()
            if not title and not root_text:
                raise ValueError("Vui lòng nhập tiêu đề hoặc nội dung bài gốc.")
            post = {
                "id": post_id,
                "title": title or _clean_title_snippet(root_text, 100),
                "selftext": root_text,
                "subreddit": "sandbox",
                "author": "op_sandbox",
                "created_utc": base_ts,
            }
            raw_comments = payload.get("comments")
            if isinstance(raw_comments, list) and raw_comments:
                comments = []
                for idx, c in enumerate(raw_comments, start=1):
                    if not isinstance(c, Mapping):
                        continue
                    cid = str(c.get("id") or f"c{idx}").removeprefix("t1_")
                    raw_p = str(c.get("parent_id") or f"t3_{post_id}").strip()
                    if raw_p.lower() in {"root", post_id, f"t3_{post_id}"}:
                        pid = f"t3_{post_id}"
                    elif raw_p.startswith(("t1_", "t3_")):
                        pid = raw_p
                    else:
                        pid = f"t1_{raw_p}"
                    body = str(c.get("text") or c.get("body") or "")
                    author = c.get("author_raw") if c.get("author_raw") is not None else c.get("author")
                    author_str = str(author) if author is not None else (
                        None if body.strip().lower() in {"[deleted]", "[removed]"} else f"custom_u_{idx}"
                    )
                    is_bot = bool(c.get("author_is_bot")) or (
                        isinstance(author_str, str) and author_str.strip().lower() == "automoderator"
                    )
                    raw_delay = c.get("time_delay_sec")
                    if raw_delay is None:
                        delay = float(idx * 45.0)
                    else:
                        if isinstance(raw_delay, bool):
                            raise ValueError(f"Giá trị time_delay_sec không hợp lệ: {raw_delay!r}")
                        try:
                            delay = float(raw_delay)
                        except (TypeError, ValueError) as exc:
                            raise ValueError(
                                f"Giá trị time_delay_sec không hợp lệ: {raw_delay!r}"
                            ) from exc
                        if not math.isfinite(delay) or delay < 0.0 or delay > 3650.0 * 86400.0:
                            raise ValueError(
                                f"Giá trị time_delay_sec không hợp lệ ({raw_delay!r}): "
                                "phải hữu hạn và nằm trong [0, 315360000] giây."
                            )
                    comments.append(
                        {
                            "fullname": f"t1_{cid}",
                            "id": cid,
                            "link_id": f"t3_{post_id}",
                            "parent_id": pid,
                            "body": body,
                            "text": body,
                            "created_utc": base_ts + delay,
                            "author": author_str,
                            "author_raw": author_str,
                            "author_is_bot": is_bot,
                            "author_hash": hash_author_sha256(author_str),
                        }
                    )
            else:
                raw_lines = str(payload.get("raw_indented_text") or "")
                comments = parse_sandbox_raw_text(raw_lines, post_id=post_id, base_ts=base_ts)
            if not comments:
                raise ValueError("Cần ít nhất 1 bình luận để dựng cây hội thoại.")
            is_demo = True
        else:
            raise ValueError(f"Chế độ sandbox không hợp lệ: {mode!r}")

        ce_ckpt = self.ce_dir if self.live_cross_encoder_available else None
        cached_ce = self._get_cached_cross_encoder()
        gnn_ckpt = self.ckpt_path if self.ckpt_path.is_file() else None
        res = predict_live_cascade(
            post,
            comments,
            k_horizon=k_horizon,
            analysis_mode=analysis_mode,
            max_full_comments=max_full_comments,
            custom_topics=custom_topics,
            cross_encoder_ckpt=ce_ckpt,
            preloaded_ce=cached_ce,
            gnn_ckpt=gnn_ckpt,
            lang=lang_val,
            junk_mode=junk_mode,
            allow_untrained=(mode != "reddit_url"),
            is_demo=is_demo,
        )
        cid = f"vn_{str(post.get('id') or 'sandbox').removeprefix('t3_')}"
        out = _format_live_result_payload(
            cascade_id=cid,
            topic_id=f"sandbox_{mode}",
            post=post,
            res=res,
            is_synthetic_fallback=not bool(res.get("cross_encoder_loaded", False)),
        )
        out["sandbox_mode"] = mode
        if mode in {"preset_junk_1", "preset_junk_2", "preset_large_1"}:
            cache_key = (
                mode,
                junk_mode,
                analysis_mode,
                str(k_horizon),
                str(max_full_comments),
                topics_cache_key,
                lang_val,
            )
            with self._lock:
                self._sandbox_preset_cache[cache_key] = out
        return out

    def classify_topic(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise ValueError("Payload JSON không hợp lệ.")
        title = str(payload.get("title") or "").strip()
        root_text = str(payload.get("root_text") or payload.get("body") or "").strip()
        if not title and not root_text:
            raise ValueError("Cần cung cấp tiêu đề hoặc nội dung bài đăng để phân loại chủ đề.")
        raw_snips = payload.get("reply_snippets")
        snips = [str(s) for s in raw_snips[:10]] if isinstance(raw_snips, list) else []
        custom_topics = payload.get("custom_topics")
        lang_raw = str(payload.get("lang") or "vi").strip().lower()
        lang_lit: Literal["vi", "en"] = "en" if lang_raw == "en" else "vi"
        backend_raw = str(payload.get("topic_backend") or "auto").strip().lower()
        if backend_raw not in {"auto", "nli", "embedding", "lexical"}:
            raise ValueError(f"topic_backend không hợp lệ: {backend_raw!r}")
        backend_lit: Literal["auto", "nli", "embedding", "lexical"] = backend_raw  # type: ignore[assignment]
        cached_ce = self._get_cached_cross_encoder()
        allow_nli_dl = bool(payload.get("allow_nli_download", False))
        return classify_post_topics(
            title,
            root_text,
            reply_snippets=snips,
            custom_topics=custom_topics,
            lang=lang_lit,
            backend=backend_lit,
            node_encoder=cached_ce[0].encoder if cached_ce is not None else None,
            node_tokenizer=cached_ce[1] if cached_ce is not None else None,
            allow_nli_download=allow_nli_dl,
        )

    def get_tier2_macro(self) -> dict[str, Any]:
        with self._lock:
            if self._macro_cache is not None:
                return self._macro_cache

        def _read_json(name: str) -> dict[str, Any]:
            path = self.paper_dir / name
            if not path.is_file():
                raise FileNotFoundError(f"Missing paper metric file: {path}")
            return json.loads(path.read_text(encoding="utf-8"))

        macro_vn = _read_json("results_macro_network_vn.json")
        macro_pheme = _read_json("results_macro_network_pheme.json")
        link_vn = _read_json("results_signed_link_vn.json")
        link_pheme = _read_json("results_signed_link_pheme.json")

        def _round_opt(val: Any) -> float | None:
            if val is None or isinstance(val, bool):
                return None
            try:
                fv = float(val)
            except (TypeError, ValueError):
                return None
            return round(fv, 4) if math.isfinite(fv) else None

        def _format_link_rows(link_json: Mapping[str, Any]) -> list[dict[str, Any]]:
            stats = link_json.get("summary_stats") or {}
            rows: list[dict[str, Any]] = []
            for model_name, metrics in stats.items():
                row: dict[str, Any] = {"model": model_name}
                for key in (
                    "roc_auc",
                    "pr_auc",
                    "macro_f1",
                    "trans_roc_auc",
                    "trans_pr_auc",
                    "trans_macro_f1",
                    "unseen_pair_trans_roc_auc",
                    "unseen_roc_auc",
                    "unseen_pr_auc",
                    "unseen_macro_f1",
                ):
                    pair = metrics.get(key)
                    if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                        row[key] = {"mean": _round_opt(pair[0]), "std": _round_opt(pair[1])}
                    else:
                        row[key] = {"mean": None, "std": None}
                rows.append(row)
            return rows

        payload = {
            "corpora": {
                "vn": {
                    "label": "r/vozforums (VN)",
                    "figure_url": "/assets/fig5_vn.svg",
                    "macro": macro_vn,
                    "signed_link": {
                        "rows": _format_link_rows(link_vn),
                        "split_counts": (link_vn.get("extra") or {}).get("split_counts") or {},
                        "p_values": link_vn.get("p_values") or {},
                        "adjusted_p_values": link_vn.get("adjusted_p_values") or {},
                        "seeds": link_vn.get("seeds") or [],
                    },
                },
                "pheme": {
                    "label": "PHEME (EN)",
                    "figure_url": "/assets/fig5_pheme.svg",
                    "macro": macro_pheme,
                    "signed_link": {
                        "rows": _format_link_rows(link_pheme),
                        "split_counts": (link_pheme.get("extra") or {}).get("split_counts") or {},
                        "p_values": link_pheme.get("p_values") or {},
                        "adjusted_p_values": link_pheme.get("adjusted_p_values") or {},
                        "seeds": link_pheme.get("seeds") or [],
                    },
                },
            },
            "caveats": [
                (
                    "Tập kiểm tra xuyên thời gian PHEME (Transductive) chỉ có n = 212 cạnh (trong đó 166 cạnh là cặp mới, "
                    "21,7% là cặp (u, v) lặp lại giữa train và test); mức Trans. ROC-AUC 0.9106 của SBiFAGAT-Link "
                    "(SiGAT = 0.9178) phản ánh độ khớp trên các nút đã xuất hiện theo sự kiện chứ không phải tổng quát hóa "
                    "người dùng mới (Unseen ROC-AUC = 0.6985; BiEdgeGATv2-Link dẫn đầu 0.7217)."
                ),
                (
                    "Trên đồ thị r/vozforums (VN), PriorPairSignPersistence dẫn đầu Trans. ROC-AUC (0.6296) và SGCN dẫn đầu "
                    "nhóm GNN (0.6101), trong khi SBiFAGAT-Link dẫn đầu ROC-AUC tổng thể (0.5946), PR-AUC (0.8486) và "
                    "Unseen ROC-AUC (0.5884) với Trans. ROC-AUC = 0.5954; sai khác so với SDGNN chưa đạt ý nghĩa thống kê "
                    "sau hiệu chỉnh Holm-Bonferroni (p >= 0.0617)."
                ),
                (
                    "Cả hai mạng lưới đều phản cân bằng cấu trúc mạnh (anti-balanced): VN z = -15.17 (3.535/5.143 tam giác "
                    "thuần dấu âm (---)), PHEME z = -4.30 (111/212 tam giác (---)), nhưng đạt Cân bằng yếu Davis cao "
                    "(VN 94.05%, PHEME 87.74%) — cho thấy cấu trúc đa phe tranh cãi dây chuyền (k >= 3)."
                ),
                (
                    "Kiểm định hoán vị Null-Graph cho thấy Gómez Q_signed trên VN (0.1988 vs Null 0.1977, z = +0.28, "
                    "p = 0.333) không vượt ngẫu nhiên kèm RWC âm (-0.0925), trong khi PHEME (10.364 nút WCC, bậc <= 3.09) "
                    "có Q_signed = 0.4000 thấp hơn Null (0.4716, z = -39.72, p = 1.00) với RWC = 0.3181 phản ánh tách cụm sự kiện."
                ),
            ],
        }
        with self._lock:
            self._macro_cache = payload
        return payload


def make_handler(backend: DemoBackend):
    """Build a hardened HTTP request handler backed by `DemoBackend`."""
    web_root = backend.web_dir.resolve()
    paper_root = backend.paper_dir.resolve()

    static_allowlist: dict[str, tuple[Path, Path, str]] = {
        "/": (backend.web_dir / "index.html", web_root, "text/html; charset=utf-8"),
        "/styles.css": (backend.web_dir / "styles.css", web_root, "text/css; charset=utf-8"),
        "/app.js": (backend.web_dir / "app.js", web_root, "application/javascript; charset=utf-8"),
        "/assets/fig5_vn.svg": (
            backend.paper_dir / "figures" / "vn" / "fig5_macro_signed_network.svg",
            paper_root,
            "image/svg+xml; charset=utf-8",
        ),
        "/assets/fig5_pheme.svg": (
            backend.paper_dir / "figures" / "pheme" / "fig5_macro_signed_network.svg",
            paper_root,
            "image/svg+xml; charset=utf-8",
        ),
    }

    class DemoRequestHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # pragma: no cover
            pass

        def _send_security_headers(self) -> None:
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'",
            )

        def _send_bytes(self, status: int, content_type: str, raw: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self._send_security_headers()
            self.end_headers()
            self.wfile.write(raw)

        def _send_json(self, status: int, body: Mapping[str, Any]) -> None:
            raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self._send_bytes(status, "application/json; charset=utf-8", raw)

        def _is_traversal_attempt(self, raw_path: str, clean_path: str) -> bool:
            low_raw = raw_path.lower()
            if ".." in raw_path or ".." in clean_path or "\\" in raw_path or "%2e" in low_raw or "\x00" in raw_path:
                return True
            return any(
                clean_path.startswith(prefix)
                for prefix in (
                    "/paper",
                    "/results",
                    "/archive",
                    "/src",
                    "/data",
                    "/checkpoints",
                    "/scripts",
                    "/.env",
                    "/.git",
                )
            )

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path

            if self._is_traversal_attempt(self.path, path):
                self._send_json(HTTPStatus.FORBIDDEN, {"error": "Forbidden path"})
                return

            if path in static_allowlist:
                target_path, allowed_root, content_type = static_allowlist[path]
                resolved = target_path.resolve()
                if not resolved.is_relative_to(allowed_root):
                    self._send_json(HTTPStatus.FORBIDDEN, {"error": "Forbidden path"})
                    return
                if not resolved.is_file():
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Static asset missing: {path}"})
                    return
                self._send_bytes(HTTPStatus.OK, content_type, resolved.read_bytes())
                return

            if path == "/api/health":
                self._send_json(HTTPStatus.OK, backend.get_health())
                return

            if path == "/api/tier1/threads":
                self._send_json(HTTPStatus.OK, backend.list_tier1_threads())
                return

            if path == "/api/tier1/thread":
                qs = parse_qs(parsed.query)
                cid = (qs.get("id") or [""])[0]
                if not cid:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Missing query parameter 'id'"})
                    return
                try:
                    payload = backend.get_tier1_thread(cid)
                except KeyError as exc:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                    return
                self._send_json(HTTPStatus.OK, payload)
                return

            if path == "/api/tier2/macro":
                try:
                    payload = backend.get_tier2_macro()
                except FileNotFoundError as exc:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                    return
                self._send_json(HTTPStatus.OK, payload)
                return

            if path.startswith("/assets/"):
                self._send_json(HTTPStatus.FORBIDDEN, {"error": "Asset not in allowlist"})
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if self._is_traversal_attempt(self.path, parsed.path):
                self._send_json(HTTPStatus.FORBIDDEN, {"error": "Forbidden path"})
                return
            if parsed.path not in {"/api/tier1/sandbox", "/api/tier1/classify_topic"}:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
                return

            try:
                length = int(self.headers.get("Content-Length") or "0")
                if length < 0 or length > _MAX_POST_BYTES:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Payload size out of bounds"})
                    return
                raw = self.rfile.read(length) if length > 0 else b"{}"
                body = json.loads(raw.decode("utf-8"))
                if parsed.path == "/api/tier1/classify_topic":
                    result = backend.classify_topic(body)
                else:
                    result = backend.run_sandbox(body)
            except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_GATEWAY, {"error": f"Không thể xử lý yêu cầu: {exc}"})
                return

            self._send_json(HTTPStatus.OK, result)

    return DemoRequestHandler


def create_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    data_dir: Path = Path("data/processed/vn"),
    ckpt_path: Path = Path("checkpoints/best_sbifagat_vn.pt"),
    ce_dir: Path = Path("checkpoints/dual_ce_vn"),
    paper_dir: Path = Path("results"),
    web_dir: Path = Path("web_demo"),
    preload: bool = False,
) -> ThreadingHTTPServer:
    """Create a strictly loopback-only HTTP server for the SBi-FAGAT Demo Web MVP."""
    clean_host = str(host or "").strip()
    if clean_host not in ALLOWED_LOCAL_HOSTS:
        raise ValueError(
            f"Refusing non-loopback bind host {host!r}; only {sorted(ALLOWED_LOCAL_HOSTS)} are permitted."
        )
    backend = DemoBackend(
        data_dir=Path(data_dir),
        ckpt_path=Path(ckpt_path),
        ce_dir=Path(ce_dir),
        paper_dir=Path(paper_dir),
        web_dir=Path(web_dir),
        preload=preload,
    )
    handler_cls = make_handler(backend)
    return ThreadingHTTPServer((clean_host, int(port)), handler_cls)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Offline-first local Defense Demo Web MVP for SBi-FAGAT.")
    parser.add_argument("--host", default="127.0.0.1", help="Loopback host only (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on (default: 8765)")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/vn"))
    parser.add_argument("--ckpt-path", type=Path, default=Path("checkpoints/best_sbifagat_vn.pt"))
    parser.add_argument("--ce-dir", type=Path, default=Path("checkpoints/dual_ce_vn"))
    parser.add_argument("--paper-dir", type=Path, default=Path("results"))
    parser.add_argument("--web-dir", type=Path, default=Path("web_demo"))
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser tab")
    args = parser.parse_args()

    server = create_server(
        host=args.host,
        port=args.port,
        data_dir=args.data_dir,
        ckpt_path=args.ckpt_path,
        ce_dir=args.ce_dir,
        paper_dir=args.paper_dir,
        web_dir=args.web_dir,
        preload=True,
    )
    actual_port = server.server_address[1]
    url = f"http://{args.host}:{actual_port}/"
    print(f"[SBi-FAGAT Demo] Running locally at {url}", flush=True)
    print("[SBi-FAGAT Demo] Press Ctrl+C to stop.", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[SBi-FAGAT Demo] Shutting down cleanly.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
