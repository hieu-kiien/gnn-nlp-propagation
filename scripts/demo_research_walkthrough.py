"""5-second CPU educational walkthrough explaining SBi-FAGAT without requiring GPU or datasets.

Demonstrates step-by-step on a 13-node Vietnamese debate tree (root + K = 12 replies):
  1. How replies form a bidirectional signed graph (Top-Down + Bottom-Up) with 2D e_uv = [s_uv, h_uv].
  2. Why StandardGCN smooths opposing stances together (over-smoothing).
  3. How SBi-FAGAT's signed gate gamma_uv in (-1, +1) flips negative on opposing/hostile edges
     while staying positive on agreeing/calm edges to keep Pro and Con camps separated.
  4. How early target labels (y_size, y_polar, Bimodality Coefficient BC) avoid data leakage.
"""

from __future__ import annotations

import math
import sys

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from src.data.pyg_dataset import (
    CascadeData,
    TOTAL_NODE_FEATURE_DIM,
    _compute_full_tree_polarization,
    compute_bimodality_coefficient,
    compute_signed_assortativity,
)
from src.models.baselines import StandardGCN
from src.models.sbi_fagat import SBiFAGAT
from src.utils.seed import seed_everything


def _make_demo_debate_graphs() -> tuple[CascadeData, CascadeData, list[str], list[str], list[int], list[int]]:
    """Build (1) a split 13-node Vietnamese debate (K=12) and (2) an agreement cascade."""
    seed_everything(42)

    topic_axis = F.normalize(torch.randn(768), dim=0)
    stance_axis = F.normalize(torch.randn(768), dim=0)

    def _make_node_vec(camp_sign: float, step: int) -> torch.Tensor:
        noise = 0.06 * F.normalize(torch.randn(768), dim=0)
        sem = F.normalize(0.45 * topic_axis + camp_sign * 0.85 * stance_axis + noise, dim=0)
        t_norm = min(1.0, 0.03 * step)
        d_norm = 0.0 if step == 0 else (0.05 if step <= 4 else 0.10)
        return torch.cat([sem, torch.tensor([t_norm, d_norm], dtype=torch.float32)], dim=0)

    node_specs = [
        ("c0 (Gốc): Đề xuất thu phí xe máy vào nội đô để giảm ùn tắc.", +1.0),
        ("r1 (->c0): Hoàn toàn ủng hộ! Đô thị lớn nào cũng cần hạn chế xe cá nhân.", +1.0),
        ("r2 (->c0): Phản đối kịch liệt! Thu phí kiểu này chỉ khổ người lao động.", -1.0),
        ("r3 (->r1): Lý thuyết suông! Xe buýt chưa đủ thì cấm với thu cái gì?", -1.0),
        ("r4 (->r2): Chưa chắc, không mạnh tay xe máy thì tắc mãi không xong.", +1.0),
        ("r5 (->c0): Đồng ý nếu miễn phí vé metro tháng đầu cho công nhân.", +1.0),
        ("r6 (->r3): Chuẩn bác ơi, hạ tầng chưa xong đã đòi tận thu.", -1.0),
        ("r7 (->r1): Chuẩn, giảm khói bụi nội đô rất đáng làm.", +1.0),
        ("r8 (->c0): Quá vô lý, phương tiện công cộng chưa đáp ứng nổi!", -1.0),
        ("r9 (->r2): Đúng vậy, thu tiền xong rồi lại đâu vào đấy thôi.", -1.0),
        ("r10 (->r5): Nhất trí 100%, cứ có lộ trình trợ giá là hợp lý.", +1.0),
        ("r11 (->r4): Nói thì dễ, thử đứng đợi xe buýt giữa trưa nắng xem sao!", -1.0),
        ("r12 (->r6): Nghĩ kỹ đi, chờ hoàn hảo hết mới làm thì 20 năm nữa vẫn tắc.", +1.0),
    ]
    labels = [spec[0] for spec in node_specs]
    camp_signs = [spec[1] for spec in node_specs]

    # Edges (u -> v) with 2D edge features [s_uv (relative to u), h_uv (hostility)]
    edge_defs = [
        (0, 1, +0.90, 0.04, "c0 -> r1  (Ủng hộ ôn hòa)"),
        (0, 2, -0.92, 0.86, "c0 -> r2  (Phản đối gay gắt)"),
        (1, 3, -0.88, 0.80, "r1 -> r3  (Phản bác phe ủng hộ)"),
        (2, 4, -0.84, 0.62, "r2 -> r4  (Phản bác phe phản đối)"),
        (0, 5, +0.85, 0.08, "c0 -> r5  (Ủng hộ có điều kiện)"),
        (3, 6, +0.86, 0.68, "r3 -> r6  (Hùa theo chỉ trích)"),
        (1, 7, +0.91, 0.05, "r1 -> r7  (Đồng tình với r1)"),
        (0, 8, -0.89, 0.79, "c0 -> r8  (Phản đối gay gắt)"),
        (2, 9, +0.88, 0.58, "r2 -> r9  (Hùa theo phản đối)"),
        (5, 10, +0.92, 0.04, "r5 -> r10 (Đồng tình r5)"),
        (4, 11, -0.87, 0.77, "r4 -> r11 (Công kích lại r4)"),
        (6, 12, -0.85, 0.70, "r6 -> r12 (Phản bác lại r6)"),
    ]

    x_split = torch.stack([_make_node_vec(s, idx) for idx, s in enumerate(camp_signs)], dim=0)
    src = [e[0] for e in edge_defs]
    dst = [e[1] for e in edge_defs]
    edge_attr_split = torch.tensor([[e[2], e[3]] for e in edge_defs], dtype=torch.float32)
    edge_names = [e[4] for e in edge_defs]

    edge_index_td = torch.tensor([src, dst], dtype=torch.long)
    edge_index_bu = torch.tensor([dst, src], dtype=torch.long)

    root_aligned_stances = camp_signs[1:]  # 6 positive (+1), 6 negative (-1)
    y_polar_split = _compute_full_tree_polarization(root_aligned_stances)
    bc_split = compute_bimodality_coefficient(root_aligned_stances)

    split_graph = CascadeData(
        x=x_split,
        edge_index=edge_index_td,
        edge_index_td=edge_index_td,
        edge_index_bu=edge_index_bu,
        edge_attr=edge_attr_split,
        edge_stance=edge_attr_split[:, 0:1],
        root_index=torch.tensor([0], dtype=torch.long),
        y_size=torch.tensor([math.log1p(len(node_specs) - 1)], dtype=torch.float32),
        y_polar=torch.tensor([y_polar_split], dtype=torch.float32),
        y_bimodality=torch.tensor([bc_split], dtype=torch.float32),
        y_rumor=torch.tensor([2], dtype=torch.long),
    )

    x_agree = torch.stack([_make_node_vec(+1.0, idx) for idx in range(len(node_specs))], dim=0)
    edge_attr_agree = torch.tensor([[+0.90, 0.05]] * len(edge_defs), dtype=torch.float32)
    agree_graph = CascadeData(
        x=x_agree,
        edge_index=edge_index_td,
        edge_index_td=edge_index_td,
        edge_index_bu=edge_index_bu,
        edge_attr=edge_attr_agree,
        edge_stance=edge_attr_agree[:, 0:1],
        root_index=torch.tensor([0], dtype=torch.long),
        y_size=torch.tensor([math.log1p(len(node_specs) - 1)], dtype=torch.float32),
        y_polar=torch.tensor([0.0], dtype=torch.float32),
        y_bimodality=torch.tensor([0.0], dtype=torch.float32),
        y_rumor=torch.tensor([0], dtype=torch.long),
    )

    pro_nodes = [i for i, s in enumerate(camp_signs) if i > 0 and s > 0]
    con_nodes = [i for i, s in enumerate(camp_signs) if i > 0 and s < 0]
    return split_graph, agree_graph, labels, edge_names, pro_nodes, con_nodes


def _mean_camp_cosine(node_emb: torch.Tensor, pro_idx: list[int], con_idx: list[int]) -> float:
    pro_center = F.normalize(node_emb[pro_idx].mean(dim=0, keepdim=True), dim=-1)
    con_center = F.normalize(node_emb[con_idx].mean(dim=0, keepdim=True), dim=-1)
    return float(F.cosine_similarity(pro_center, con_center).item())


def run_walkthrough() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("=" * 88)
    print("[MINH HỌA CƠ CHẾ HỌC THUẬT - DỮ LIỆU GIẢ LẬP ĐỂ GIẢI THÍCH TOÁN HỌC]")
    print("[TUYỆT ĐỐI KHÔNG DÙNG ĐẦU RA NÀY LÀM SỐ LIỆU BÁO CÁO KHOA HỌC]")
    print("=" * 88)

    split_g, agree_g, node_labels, edge_names, pro_nodes, con_nodes = _make_demo_debate_graphs()

    print("\n[Bước 1] Cây thảo luận tiếng Việt mẫu (N = 13 nút, K = 12 phản hồi sớm):")
    for txt in node_labels[:7]:
        print(f"  - {txt}")
    print(f"  ... ({len(node_labels) - 7} bình luận kế tiếp tạo thành 2 phe cân bằng 6 Ủng hộ vs 6 Phản đối)")

    src_list = split_g.edge_index_td[0].tolist()
    dst_list = split_g.edge_index_td[1].tolist()
    edge_pairs_str = [(f"n{u}", f"n{v}") for u, v in zip(src_list, dst_list)]
    signs_by_idx = [+1.0 if (i == 0 or i in pro_nodes) else -1.0 for i in range(split_g.num_nodes)]
    node_stance_map = {f"n{i}": s for i, s in enumerate(signs_by_idx)}
    assort_r = compute_signed_assortativity(edge_pairs_str, node_stance_map, root_id="n0")

    print(f"\n  -> Mục tiêu cuối kỳ y_polar (phương sai + cân bằng 2 phe) = {float(split_g.y_polar.item()):.3f}")
    print(f"  -> Kiểm định tách phe độc lập Sarle Bimodality (BC > 0.555 là phân đôi) = {float(split_g.y_bimodality.item()):.3f}")
    print(f"  -> Hệ số đồng thuận/xung đột cấu trúc Signed Assortativity r = {assort_r:+.3f}")
    print("  -> Chú ý Zero Data Leakage: data.x chỉ chứa [vector 768-d, t_norm, d_norm], không hề chứa y_polar hay y_size!")

    print("\n[Bước 2] So sánh GCN truyền thống (Over-Smoothing) vs. SBi-FAGAT (Cổng dấu gamma_uv):")
    raw_cos = _mean_camp_cosine(split_g.x, pro_nodes, con_nodes)
    print(f"  - Cosine ban đầu giữa 2 phe {pro_nodes} & {con_nodes}: {raw_cos:+.4f} (đối lập rõ)")

    seed_everything(42)
    gcn = StandardGCN(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=64, num_rumor_classes=4)
    fagat = SBiFAGAT(in_dim=TOTAL_NODE_FEATURE_DIM, hidden_dim=64, edge_dim=2, num_rumor_classes=4)

    # Train both StandardGCN and SBiFAGAT on the exact same pure polarization loss (zero extra supervision)
    contrast_batch = Batch.from_data_list([split_g, agree_g])
    for _name, model in (("StandardGCN", gcn), ("SBiFAGAT", fagat)):
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()
        for _ in range(25):
            opt.zero_grad()
            out = model(contrast_batch)
            loss = F.mse_loss(out["y_polar_pred"], contrast_batch.y_polar)
            loss.backward()
            opt.step()
        model.eval()

    with torch.no_grad():
        gcn_out = gcn(split_g)
        fagat_out = fagat(split_g)
        gates = fagat.extract_edge_gates(split_g)

    gcn_cos = _mean_camp_cosine(gcn_out["node_emb"], pro_nodes, con_nodes)
    fagat_cos = _mean_camp_cosine(fagat_out["node_emb"], pro_nodes, con_nodes)

    print(f"  - StandardGCN (chỉ cộng dương hàng xóm): Cosine(Ủng hộ, Phản đối) = {gcn_cos:+.4f}  (2 phe bị kéo chập lại -> san phẳng)")
    print(f"  - SBi-FAGAT (lan truyền có dấu 2 chiều):  Cosine(Ủng hộ, Phản đối) = {fagat_cos:+.4f}  (giữ khoảng cách giữa 2 phe)")
    print(
        f"  - Dự đoán y_polar (Đáp án = {float(split_g.y_polar.item()):.3f}): "
        f"StandardGCN={float(gcn_out['y_polar_pred'].item()):.3f} | "
        f"SBi-FAGAT={float(fagat_out['y_polar_pred'].item()):.3f}"
    )

    print("\n[Bước 3] Giải mã cổng tần số có dấu gamma_uv trên 6 cạnh đầu tiên (Interpretability):")
    gamma_td = gates["gamma_td"].tolist()
    gamma_bu = gates["gamma_bu"].tolist()
    e_vals = split_g.edge_attr.tolist()
    for name, (s_uv, h_uv), g_td, g_bu in list(zip(edge_names, e_vals, gamma_td, gamma_bu))[:6]:
        kind = "TẦN SỐ THẤP (+ giữ đồng thuận)" if g_td >= 0 else "TẦN SỐ CAO (- đẩy 2 phe tách xa)"
        print(
            f"  - {name:34s} | s={s_uv:+.2f}, h={h_uv:.2f} | "
            f"gamma_TD={g_td:+.3f}, gamma_BU={g_bu:+.3f} -> {kind}"
        )

    print("\n[Hoàn tất] Chạy xong demo cơ chế trong < 5 giây!")


if __name__ == "__main__":
    run_walkthrough()
