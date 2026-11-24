"use strict";

(() => {
  const SVG_NS = "http://www.w3.org/2000/svg";

  const STANCE_COLORS = {
    root: { fill: "#0F172A", stroke: "#0F172A", text: "#FFFFFF", edge: "#475569", label: "Bài gốc" },
    support: { fill: "#ECFDF5", stroke: "#059669", text: "#065F46", edge: "#059669", label: "Tán thành" },
    oppose: { fill: "#FEF2F2", stroke: "#DC2626", text: "#991B1B", edge: "#DC2626", label: "Phản đối" },
    neutral: { fill: "#F8FAFC", stroke: "#64748B", text: "#334155", edge: "#64748B", label: "Trung lập" },
  };

  const BEHAVIOR_VI = {
    root: "Bài gốc khởi nguồn",
    support: "Tán thành / Ủng hộ",
    neutral: "Trung lập / Hỏi thông tin",
    constructive_oppose: "Phản biện ôn hòa",
    irrational_fallacy: "Ngụy biện / Bẻ lái",
    hostile_conflict: "Xung đột / Gay gắt cao",
  };

  const state = {
    activeTab: "tier1",
    tier1Mode: "real", // "real" | "sandbox"
    viewMode: "tree", // "tree" | "user_graph"
    showProfanityBadge: false,
    threadsIndex: null,
    currentThreadId: null,
    currentPayload: null,
    selectedNodeId: null,
    selectedUserId: null,
    tier2Data: null,
    tier2Corpus: "vn",
    sortMetric: null,
    sortDir: "desc",
  };

  // Safe DOM builders using textContent exclusively (zero dynamic HTML strings)
  const $ = (id) => document.getElementById(id);

  function clearEl(el) {
    while (el.firstChild) {
      el.removeChild(el.firstChild);
    }
  }

  function el(tag, attrs = {}, text = null) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v === null || v === undefined) continue;
      if (k === "className") node.className = v;
      else if (k === "dataset") {
        for (const [dk, dv] of Object.entries(v)) node.dataset[dk] = dv;
      } else {
        node.setAttribute(k, String(v));
      }
    }
    if (text !== null && text !== undefined) {
      node.textContent = String(text);
    }
    return node;
  }

  function svgEl(tag, attrs = {}) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v !== null && v !== undefined) {
        node.setAttribute(k, String(v));
      }
    }
    return node;
  }

  function announce(msg) {
    const region = $("live-status");
    if (region) region.textContent = msg;
  }

  function showError(msg) {
    const box = $("live-error");
    if (!box) return;
    if (!msg) {
      box.hidden = true;
      box.textContent = "";
    } else {
      box.textContent = msg;
      box.hidden = false;
    }
  }

  function fmtNum(val, digits = 4) {
    if (val === null || val === undefined || Number.isNaN(Number(val))) return "—";
    return Number(val).toFixed(digits);
  }

  function fmtSigned(val, digits = 3) {
    if (val === null || val === undefined || Number.isNaN(Number(val))) return "—";
    const num = Number(val);
    return `${num >= 0 ? "+" : ""}${num.toFixed(digits)}`;
  }

  function fmtPct(val, digits = 1) {
    if (val === null || val === undefined || Number.isNaN(Number(val))) return "—";
    return `${(Number(val) * 100).toFixed(digits)}%`;
  }

  function fmtInt(val) {
    if (val === null || val === undefined || Number.isNaN(Number(val))) return "—";
    return Math.round(Number(val)).toLocaleString("vi-VN");
  }

  function truncText(str, maxLen) {
    const s = String(str || "").replace(/\s+/g, " ").trim();
    if (s.length <= maxLen) return s;
    return s.slice(0, Math.max(0, maxLen - 1)).trimEnd() + "…";
  }

  async function apiGet(url) {
    const resp = await fetch(url, { headers: { Accept: "application/json" } });
    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data.error || `HTTP ${resp.status}`);
    }
    return data;
  }

  async function apiPost(url, payload) {
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data.error || `HTTP ${resp.status}`);
    }
    return data;
  }

  // =========================================================================
  // Accessible Tab Management (ARIA + Roving Tabindex + beforematch)
  // =========================================================================
  function initTabs() {
    const tabs = [$("tab-tier1"), $("tab-tier2")];
    const panels = {
      tier1: $("panel-tier1"),
      tier2: $("panel-tier2"),
    };

    function activateTab(tabKey, moveFocus = false) {
      state.activeTab = tabKey;
      const isT1 = tabKey === "tier1";
      tabs[0].setAttribute("aria-selected", String(isT1));
      tabs[0].setAttribute("tabindex", isT1 ? "0" : "-1");
      tabs[1].setAttribute("aria-selected", String(!isT1));
      tabs[1].setAttribute("tabindex", !isT1 ? "0" : "-1");

      if (isT1) {
        panels.tier1.removeAttribute("hidden");
        panels.tier2.setAttribute("hidden", "until-found");
      } else {
        panels.tier2.removeAttribute("hidden");
        panels.tier1.setAttribute("hidden", "until-found");
        if (!state.tier2Data) {
          loadTier2Macro().catch((err) => showError(err.message));
        }
      }

      if (moveFocus) {
        (isT1 ? tabs[0] : tabs[1]).focus();
      }
    }

    tabs[0].addEventListener("click", () => activateTab("tier1", false));
    tabs[1].addEventListener("click", () => activateTab("tier2", false));

    const tablist = document.querySelector('[role="tablist"]');
    tablist.addEventListener("keydown", (e) => {
      const idx = document.activeElement === tabs[1] ? 1 : 0;
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        e.preventDefault();
        const nextIdx = (idx + 1) % 2;
        activateTab(nextIdx === 0 ? "tier1" : "tier2", true);
      } else if (e.key === "Home") {
        e.preventDefault();
        activateTab("tier1", true);
      } else if (e.key === "End") {
        e.preventDefault();
        activateTab("tier2", true);
      }
    });

    panels.tier1.addEventListener("beforematch", () => activateTab("tier1", false));
    panels.tier2.addEventListener("beforematch", () => activateTab("tier2", false));
  }

  // =========================================================================
  // Tab 1 Rendering: Provenance, KPIs, Contraction Strip, SVG Tree/UserGraph
  // =========================================================================
  function renderHonestyBanner(payload) {
    const meta = payload.meta || {};
    const banner = $("honesty-banner");
    const textEl = $("honesty-text");
    clearEl(textEl);

    const ceLoaded = Boolean(meta.cross_encoder_loaded);
    const isSynth = Boolean(meta.is_synthetic_fallback);
    const ckptLoaded = Boolean(meta.checkpoint_loaded);

    if (ceLoaded && !isSynth && state.tier1Mode === "real") {
      banner.dataset.tone = "verified";
      textEl.appendChild(
        el(
          "strong",
          {},
          "Xác thực dữ liệu thật (Tập Test VN r/vozforums): "
        )
      );
      textEl.appendChild(
        document.createTextNode(
          "Sử dụng nhúng ViSoBERT SpanPool và đặc trưng cạnh 2D [s_uv, h_uv] từ Student DualHeadCrossEncoder trích xuất sẵn trên GPU + trọng số SBi-FAGAT đã huấn luyện (tách chủ đề độc lập, zero Test leakage)."
        )
      );
    } else {
      banner.dataset.tone = "warning";
      const prefix = isSynth && state.tier1Mode === "real"
        ? "CHẾ ĐỘ MẪU TỔNG HỢP NGOẠI TUYẾN (Thiếu dữ liệu VN cục bộ): "
        : "CẢNH BÁO TRUNG THỰC KHOA HỌC (cross_encoder_loaded = false): ";
      textEl.appendChild(el("strong", {}, prefix));
      const detail = ckptLoaded
        ? "Trọng số DualHeadCrossEncoder không có mặt cục bộ, nên điểm cạnh [s_uv, h_uv] trong Sandbox được chấm bằng quy tắc Heuristic chỉ để minh họa cơ chế Path Contraction và định tuyến cây; trọng số GNN đầu cuối vẫn nạp từ best_sbifagat_vn.pt."
        : "Cả trọng số DualHeadCrossEncoder lẫn checkpoint GNN đều không có mặt cục bộ (hoặc đang chạy giả lập CI); mọi chỉ số chỉ phục vụ kiểm chứng cấu trúc đường ống và Path Contraction, không phải dự báo khoa học thực nghiệm.";
      textEl.appendChild(document.createTextNode(detail));
    }

    $("pill-nlp-backend").textContent = `nlp: ${meta.nlp_backend || "—"}`;
    $("pill-cascade-id").textContent = `id: ${meta.cascade_id || "—"}`;
  }

  function renderTier1KPIs(payload) {
    const fc = payload.forecast || {};
    const meta = payload.meta || {};
    $("kpi-pred-replies").textContent = `~${Number(fc.predicted_total_replies || 0).toFixed(1)}`;
    $("kpi-actual-replies").textContent = fmtInt(fc.actual_total_replies);

    $("kpi-pred-polar").textContent = fmtNum(fc.predicted_polarization_score, 4);
    $("kpi-actual-polar").textContent = fc.has_future_replies
      ? fmtNum(fc.actual_y_polar, 4)
      : `${fmtNum(fc.actual_y_polar, 4)} (quan sát)`;

    $("kpi-regime").textContent = fc.predicted_regime || "—";
    $("kpi-confidence").textContent = fmtPct(fc.confidence_score, 1);
    $("kpi-authors").textContent = fmtInt(meta.author_count);

    const sc = fc.stance_counts || { support: 0, oppose: 0, neutral: 0 };
    const total = Math.max(1, (sc.support || 0) + (sc.neutral || 0) + (sc.oppose || 0));
    $("bar-support").style.width = `${((sc.support || 0) / total) * 100}%`;
    $("bar-neutral").style.width = `${((sc.neutral || 0) / total) * 100}%`;
    $("bar-oppose").style.width = `${((sc.oppose || 0) / total) * 100}%`;

    $("kpi-k-obs").textContent = `K=${fc.k_observed ?? 0}`;
    $("kpi-stance-counts").textContent =
      `Tán thành: ${sc.support || 0} · Trung lập: ${sc.neutral || 0} · Phản đối: ${sc.oppose || 0}`;

    $("root-title-display").textContent = meta.root_title || "—";
    $("root-text-display").textContent = meta.root_text || "—";
  }

  function renderContractionStrip(payload) {
    const strip = $("contraction-strip");
    const stats = payload.junk_filter_stats;
    if (state.tier1Mode !== "sandbox" || !stats) {
      strip.hidden = true;
      return;
    }
    strip.hidden = false;
    $("contraction-mode-tag").textContent = `mode: ${stats.mode || "contract"}`;
    $("cs-raw").textContent = fmtInt(stats.raw_comments);
    $("cs-kept").textContent = fmtInt(stats.kept_comments);
    $("cs-dropped").textContent = fmtInt(stats.dropped_leaves);
    $("cs-contracted").textContent = fmtInt(stats.contracted_nodes);
    $("cs-relinked").textContent = fmtInt(stats.relinked_edges);

    const pills = $("preserved-stances-pills");
    clearEl(pills);
    const shortList = payload.preserved_short_stances || [];
    if (!shortList.length) {
      pills.appendChild(el("span", { className: "muted-note" }, "Không có bình luận <= 3 từ trong mẫu này."));
    } else {
      for (const txt of shortList) {
        pills.appendChild(el("span", { className: "stance-chip" }, `"${txt}"`));
      }
    }
  }

  function renderTriggerEdges(payload) {
    const list = $("trigger-edges-list");
    clearEl(list);
    const triggers = payload.top_trigger_edges || [];
    if (!triggers.length) {
      list.appendChild(el("li", { className: "muted-note" }, "Chưa có cạnh kích hoạt."));
      return;
    }

    for (const tr of triggers) {
      const li = el("li");
      const btn = el("button", { type: "button", className: "trigger-btn" });
      const top = el("div", { className: "trigger-btn-top" });
      const stInfo = STANCE_COLORS[tr.stance_class] || STANCE_COLORS.neutral;

      top.appendChild(
        el("span", { className: `badge-tag badge-${tr.stance_class || "neutral"}` }, stInfo.label)
      );
      top.appendChild(
        el(
          "span",
          { className: "tabular" },
          `γ=${fmtSigned(tr.gamma_td, 3)} · |γ|=${fmtNum(tr.importance, 3)}`
        )
      );
      btn.appendChild(top);
      btn.appendChild(
        el("div", { className: "trigger-snip" }, tr.reply_text_snippet || tr.target_id)
      );
      btn.addEventListener("click", () => {
        if (state.viewMode !== "tree") {
          setViewMode("tree");
        }
        selectTreeNode(tr.target_id);
      });
      li.appendChild(btn);
      list.appendChild(li);
    }
  }

  function renderTreeNodeInspector(node) {
    const box = $("node-inspector-body");
    clearEl(box);
    if (!node) {
      box.appendChild(
        el("p", { className: "muted-note" }, "Nhấn vào một nút trên sơ đồ để xem chi tiết.")
      );
      return;
    }

    const tags = el("div", { className: "meta-pills" });
    const stInfo = STANCE_COLORS[node.stance_class] || STANCE_COLORS.neutral;
    tags.appendChild(el("span", { className: `badge-tag badge-${node.stance_class}` }, stInfo.label));
    if (node.relinked) {
      tags.appendChild(el("span", { className: "badge-tag badge-relink" }, "Nối nhánh (Contracted)"));
    }
    if (state.showProfanityBadge && node.is_uncivil) {
      tags.appendChild(el("span", { className: "badge-tag badge-uncivil" }, "Công kích / Thô tục"));
    }
    box.appendChild(tags);

    box.appendChild(el("blockquote", { className: "detail-quote" }, node.text || "(Trống)"));

    const kv = el("div", { className: "detail-kv-grid" });
    const entries = [
      ["Tác giả (SHA-256)", node.author_hash || "—"],
      ["Độ sâu / Độ trễ", `d=${node.depth} · +${Math.round(node.time_delay_sec || 0)}s`],
      ["Lập trường s_uv", node.is_root ? "Bài gốc (+1.0)" : fmtSigned(node.s_uv, 4)],
      ["Gay gắt h_uv", node.is_root ? "—" : fmtNum(node.h_uv, 4)],
      ["Cổng SBi-FAGAT γ_uv", node.is_root ? "—" : (node.gamma_td !== null && node.gamma_td !== undefined ? fmtSigned(node.gamma_td, 4) : "— (Top 5 bên dưới)")],
      ["Nhóm diễn ngôn", BEHAVIOR_VI[node.behavior_label] || node.behavior_label || "—"],
    ];
    for (const [k, v] of entries) {
      const cell = el("div", { className: "kv-cell" });
      cell.appendChild(el("div", { className: "kv-k" }, k));
      cell.appendChild(el("div", { className: "kv-v tabular" }, v));
      kv.appendChild(cell);
    }
    box.appendChild(kv);
  }

  function renderUserInspector(userNode) {
    const box = $("node-inspector-body");
    clearEl(box);
    if (!userNode) {
      box.appendChild(el("p", { className: "muted-note" }, "Nhấn vào một nút người dùng để xem chi tiết."));
      return;
    }

    const tags = el("div", { className: "meta-pills" });
    const campBadgeClass = userNode.camp === 0 ? "badge-support" : "badge-oppose";
    const campLabel = userNode.camp === 0 ? "Phe 0 · Gốc / Đồng thuận" : "Phe 1 · Đối lập / Phản biện";
    tags.appendChild(el("span", { className: `badge-tag ${campBadgeClass}` }, campLabel));
    if (userNode.is_op) {
      tags.appendChild(el("span", { className: "badge-tag badge-root" }, "Tác giả bài gốc (OP)"));
    }
    if (state.showProfanityBadge && userNode.is_uncivil) {
      tags.appendChild(el("span", { className: "badge-tag badge-uncivil" }, "Có phát ngôn công kích"));
    }
    box.appendChild(tags);

    const kv = el("div", { className: "detail-kv-grid" });
    const entries = [
      ["Định danh ẩn danh", userNode.user_id],
      ["Số lượt tham gia", `${userNode.reply_count} nút`],
      ["Cân bằng lập trường", fmtSigned(userNode.stance_balance, 4)],
      ["Phân cụm trong thread", `Camp ${userNode.camp}`],
    ];
    for (const [k, v] of entries) {
      const cell = el("div", { className: "kv-cell" });
      cell.appendChild(el("div", { className: "kv-k" }, k));
      cell.appendChild(el("div", { className: "kv-v tabular" }, v));
      kv.appendChild(cell);
    }
    box.appendChild(kv);

    const nodes = (state.currentPayload?.tree?.nodes || []).filter(
      (n) => n.author_hash === userNode.user_id
    );
    if (nodes.length) {
      box.appendChild(el("div", { className: "field-label" }, "Các bình luận trong cửa sổ quan sát:"));
      for (const n of nodes) {
        const stInfo = STANCE_COLORS[n.stance_class] || STANCE_COLORS.neutral;
        const item = el("div", { className: "detail-quote" }, `[${stInfo.label}] ${n.text}`);
        box.appendChild(item);
      }
    }
  }

  function selectTreeNode(nodeId) {
    state.selectedNodeId = nodeId;
    const svg = $("graph-svg");
    svg.querySelectorAll(".svg-node-group").forEach((g) => {
      g.classList.toggle("is-selected", g.dataset.nodeId === nodeId);
    });
    const node = (state.currentPayload?.tree?.nodes || []).find((n) => n.node_id === nodeId);
    renderTreeNodeInspector(node || null);
  }

  function selectUserNode(userId) {
    state.selectedUserId = userId;
    const svg = $("graph-svg");
    svg.querySelectorAll(".svg-node-group").forEach((g) => {
      g.classList.toggle("is-selected", g.dataset.userId === userId);
    });
    const userNode = (state.currentPayload?.user_graph?.nodes || []).find((u) => u.user_id === userId);
    renderUserInspector(userNode || null);
  }

  // =========================================================================
  // Pure Vanilla SVG Renderers (Tree Layout + Thread User Graph)
  // =========================================================================
  function renderTreeSvg(payload) {
    const svg = $("graph-svg");
    clearEl(svg);
    const tree = payload.tree || { nodes: [], edges: [] };
    const nodes = tree.nodes || [];
    const edges = tree.edges || [];
    if (!nodes.length) return;

    const nodeMap = new Map(nodes.map((n) => [n.node_id, n]));
    const childrenMap = new Map(nodes.map((n) => [n.node_id, []]));
    for (const n of nodes) {
      if (n.parent_id && childrenMap.has(n.parent_id)) {
        childrenMap.get(n.parent_id).push(n.node_id);
      }
    }
    for (const arr of childrenMap.values()) {
      arr.sort((a, b) => (nodeMap.get(a).time_delay_sec || 0) - (nodeMap.get(b).time_delay_sec || 0));
    }

    const root = nodes.find((n) => n.is_root) || nodes[0];
    const widthSlots = new Map();
    const visited = new Set();

    function computeSlots(nid) {
      if (visited.has(nid)) return 1;
      visited.add(nid);
      const ch = childrenMap.get(nid) || [];
      if (!ch.length) {
        widthSlots.set(nid, 1);
        return 1;
      }
      let sum = 0;
      for (const c of ch) sum += computeSlots(c);
      const w = Math.max(1, sum);
      widthSlots.set(nid, w);
      return w;
    }

    const totalSlots = computeSlots(root.node_id);
    const NODE_W = 176;
    const NODE_H = 58;
    const SLOT_W = 194;
    const LEVEL_H = 104;
    const PAD_X = 28;
    const PAD_Y = 24;

    const pos = new Map();
    let maxDepth = 0;
    const placed = new Set();

    function assignPos(nid, startSlot, depth) {
      if (placed.has(nid)) return;
      placed.add(nid);
      maxDepth = Math.max(maxDepth, depth);
      const mySlots = widthSlots.get(nid) || 1;
      const centerSlot = startSlot + mySlots / 2;
      const x = PAD_X + centerSlot * SLOT_W - NODE_W / 2;
      const y = PAD_Y + depth * LEVEL_H;
      pos.set(nid, { x, y, cx: x + NODE_W / 2, cy: y + NODE_H / 2 });

      let cursor = startSlot;
      for (const cid of childrenMap.get(nid) || []) {
        const cSlots = widthSlots.get(cid) || 1;
        assignPos(cid, cursor, depth + 1);
        cursor += cSlots;
      }
    }

    assignPos(root.node_id, 0, 0);

    const svgW = Math.max(880, totalSlots * SLOT_W + PAD_X * 2);
    const svgH = Math.max(440, (maxDepth + 1) * LEVEL_H + PAD_Y * 2);
    svg.setAttribute("viewBox", `0 0 ${svgW} ${svgH}`);
    svg.style.minWidth = `${Math.min(svgW, 1100)}px`;
    svg.setAttribute(
      "aria-label",
      `Sơ đồ cây thảo luận gồm ${nodes.length} nút với ${edges.length} cạnh`
    );

    const edgeLayer = svgEl("g", { class: "edges-layer" });
    const nodeLayer = svgEl("g", { class: "nodes-layer" });
    svg.appendChild(edgeLayer);
    svg.appendChild(nodeLayer);

    for (const e of edges) {
      const p1 = pos.get(e.source_id);
      const p2 = pos.get(e.target_id);
      if (!p1 || !p2) continue;

      const x1 = p1.cx;
      const y1 = p1.y + NODE_H;
      const x2 = p2.cx;
      const y2 = p2.y;
      const midY = (y1 + y2) / 2;
      const d = `M ${x1} ${y1} C ${x1} ${midY}, ${x2} ${midY}, ${x2} ${y2}`;

      const colorInfo = STANCE_COLORS[e.stance_class] || STANCE_COLORS.neutral;
      const gammaAbs = Math.abs(Number(e.gamma_td || 0));
      const strokeW = (1.8 + 1.8 * Math.min(1, gammaAbs)).toFixed(2);

      const pathAttrs = {
        d,
        fill: "none",
        stroke: e.relinked ? "#D97706" : colorInfo.edge,
        "stroke-width": e.relinked ? "2.6" : strokeW,
        "stroke-opacity": e.relinked ? "0.95" : "0.78",
      };
      if (e.relinked) {
        pathAttrs["stroke-dasharray"] = "6 4";
      }
      edgeLayer.appendChild(svgEl("path", pathAttrs));

      if (e.relinked) {
        const mx = (x1 + x2) / 2;
        const my = midY;
        const tagGroup = svgEl("g");
        tagGroup.appendChild(
          svgEl("rect", {
            x: mx - 34,
            y: my - 10,
            width: 68,
            height: 20,
            rx: 999,
            fill: "#FFFBEB",
            stroke: "#D97706",
            "stroke-width": 1.2,
          })
        );
        const tagTxt = svgEl("text", {
          x: mx,
          y: my + 3.5,
          "text-anchor": "middle",
          "font-size": "10",
          "font-weight": "700",
          fill: "#92400E",
        });
        tagTxt.textContent = "Nối nhánh";
        tagGroup.appendChild(tagTxt);
        edgeLayer.appendChild(tagGroup);
      }
    }

    for (const n of nodes) {
      const p = pos.get(n.node_id);
      if (!p) continue;
      const stInfo = STANCE_COLORS[n.stance_class] || STANCE_COLORS.neutral;

      const g = svgEl("g", {
        class: `svg-node-group${state.selectedNodeId === n.node_id ? " is-selected" : ""}`,
        tabindex: "0",
        role: "button",
        "aria-label": `${stInfo.label}: ${n.author_hash}, ${truncText(n.text, 60)}`,
        "data-node-id": n.node_id,
      });

      const rectAttrs = {
        class: "node-box",
        x: p.x,
        y: p.y,
        width: NODE_W,
        height: NODE_H,
        rx: 8,
        fill: stInfo.fill,
        stroke: n.relinked ? "#D97706" : stInfo.stroke,
        "stroke-width": n.is_root ? 2.2 : 1.6,
      };
      if (n.relinked) {
        rectAttrs["stroke-dasharray"] = "5 3";
      }
      g.appendChild(svgEl("rect", rectAttrs));

      const headerTxt = svgEl("text", {
        x: p.x + 10,
        y: p.y + 18,
        "font-size": "10.5",
        "font-weight": "700",
        "font-family": "var(--font-mono)",
        fill: stInfo.text,
      });
      headerTxt.textContent = n.is_root
        ? `${n.author_hash} · GỐC`
        : `${n.author_hash} · +${Math.round(n.time_delay_sec || 0)}s`;
      g.appendChild(headerTxt);

      const bodyTxt = svgEl("text", {
        x: p.x + 10,
        y: p.y + 40,
        "font-size": "11.5",
        "font-weight": "500",
        fill: stInfo.text,
      });
      bodyTxt.textContent = truncText(n.text, 23);
      g.appendChild(bodyTxt);

      // Optional Profanity / Hostility Badge (strictly OFF by default!)
      if (state.showProfanityBadge && n.is_uncivil) {
        g.appendChild(
          svgEl("rect", {
            x: p.x + NODE_W - 62,
            y: p.y - 9,
            width: 58,
            height: 17,
            rx: 999,
            fill: "#DC2626",
          })
        );
        const badTxt = svgEl("text", {
          x: p.x + NODE_W - 33,
          y: p.y + 3,
          "text-anchor": "middle",
          "font-size": "9.5",
          "font-weight": "700",
          fill: "#FFFFFF",
        });
        badTxt.textContent = "Công kích";
        g.appendChild(badTxt);
      }

      const activate = () => selectTreeNode(n.node_id);
      g.addEventListener("click", activate);
      g.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter" || ev.key === " ") {
          ev.preventDefault();
          activate();
        }
      });

      nodeLayer.appendChild(g);
    }

    if (!state.selectedNodeId || !nodeMap.has(state.selectedNodeId)) {
      selectTreeNode(root.node_id);
    } else {
      selectTreeNode(state.selectedNodeId);
    }
  }

  function renderUserGraphSvg(payload) {
    const svg = $("graph-svg");
    clearEl(svg);
    const ug = payload.user_graph || { nodes: [], edges: [], camp_counts: { camp0: 0, camp1: 0 } };
    const uNodes = ug.nodes || [];
    const uEdges = ug.edges || [];

    const svgW = 900;
    const camp0 = uNodes.filter((u) => u.camp === 0);
    const camp1 = uNodes.filter((u) => u.camp === 1);
    const maxCampLen = Math.max(camp0.length, camp1.length, 1);
    const svgH = Math.max(460, 120 + maxCampLen * 62);

    svg.setAttribute("viewBox", `0 0 ${svgW} ${svgH}`);
    svg.style.minWidth = "720px";
    svg.setAttribute(
      "aria-label",
      `Đồ thị tương tác người dùng trong thread: ${camp0.length} người phe 0, ${camp1.length} người phe 1`
    );

    const defs = svgEl("defs");
    for (const [cls, col] of [
      ["support", "#059669"],
      ["oppose", "#DC2626"],
      ["neutral", "#64748B"],
    ]) {
      const marker = svgEl("marker", {
        id: `arrow-${cls}`,
        viewBox: "0 0 10 10",
        refX: "8",
        refY: "5",
        markerWidth: "6",
        markerHeight: "6",
        orient: "auto-start-reverse",
      });
      marker.appendChild(svgEl("path", { d: "M 0 1 L 10 5 L 0 9 z", fill: col }));
      defs.appendChild(marker);
    }
    svg.appendChild(defs);

    // Camp divider & labels
    const bgLayer = svgEl("g");
    bgLayer.appendChild(
      svgEl("line", {
        x1: svgW / 2,
        y1: 24,
        x2: svgW / 2,
        y2: svgH - 20,
        stroke: "#CBD5E1",
        "stroke-width": 1.5,
        "stroke-dasharray": "5 5",
      })
    );

    const lTitle = svgEl("text", {
      x: 225,
      y: 34,
      "text-anchor": "middle",
      "font-size": "13",
      "font-weight": "700",
      fill: "#065F46",
    });
    lTitle.textContent = `Phe 0 · Gốc / Đồng thuận (${camp0.length} người)`;
    bgLayer.appendChild(lTitle);

    const rTitle = svgEl("text", {
      x: 675,
      y: 34,
      "text-anchor": "middle",
      "font-size": "13",
      "font-weight": "700",
      fill: "#991B1B",
    });
    rTitle.textContent = `Phe 1 · Đối lập / Phản biện (${camp1.length} người)`;
    bgLayer.appendChild(rTitle);
    svg.appendChild(bgLayer);

    const pos = new Map();
    function layoutCamp(arr, baseCenterX, bendDir) {
      const topY = 78;
      const botY = svgH - 52;
      const span = Math.max(1, botY - topY);
      arr.forEach((u, i) => {
        const frac = arr.length <= 1 ? 0.5 : i / (arr.length - 1);
        const y = topY + frac * span;
        const arc = Math.sin(frac * Math.PI) * 90 * bendDir;
        const x = baseCenterX + arc;
        const r = 16 + Math.min(11, (u.reply_count || 1) * 2.4);
        pos.set(u.user_id, { x, y, r });
      });
    }

    layoutCamp(camp0, 250, -1);
    layoutCamp(camp1, 650, 1);

    const edgeLayer = svgEl("g");
    const nodeLayer = svgEl("g");
    svg.appendChild(edgeLayer);
    svg.appendChild(nodeLayer);

    for (const e of uEdges) {
      const p1 = pos.get(e.source);
      const p2 = pos.get(e.target);
      if (!p1 || !p2) continue;

      const dx = p2.x - p1.x;
      const dy = p2.y - p1.y;
      const dist = Math.hypot(dx, dy) || 1;
      const ux = dx / dist;
      const uy = dy / dist;

      const sx = p1.x + ux * (p1.r + 2);
      const sy = p1.y + uy * (p1.r + 2);
      const ex = p2.x - ux * (p2.r + 5);
      const ey = p2.y - uy * (p2.r + 5);

      const curveOffset = Math.min(55, Math.max(22, dist * 0.16));
      const mx = (sx + ex) / 2 - uy * curveOffset;
      const my = (sy + ey) / 2 + ux * curveOffset;

      const stInfo = STANCE_COLORS[e.stance_class] || STANCE_COLORS.neutral;
      const strokeW = (1.6 + 1.5 * Math.log1p(e.count || 1)).toFixed(2);
      edgeLayer.appendChild(
        svgEl("path", {
          d: `M ${sx} ${sy} Q ${mx} ${my} ${ex} ${ey}`,
          fill: "none",
          stroke: stInfo.edge,
          "stroke-width": strokeW,
          "stroke-opacity": "0.78",
          "marker-end": `url(#arrow-${e.stance_class || "neutral"})`,
        })
      );
    }

    for (const u of uNodes) {
      const p = pos.get(u.user_id);
      if (!p) continue;
      const fill = u.camp === 0 ? "#ECFDF5" : "#FEF2F2";
      const stroke = u.is_op ? "#0F172A" : (u.camp === 0 ? "#059669" : "#DC2626");
      const textCol = u.camp === 0 ? "#065F46" : "#991B1B";

      const g = svgEl("g", {
        class: `svg-node-group${state.selectedUserId === u.user_id ? " is-selected" : ""}`,
        tabindex: "0",
        role: "button",
        "aria-label": `${u.label}, Phe ${u.camp}, ${u.reply_count} bình luận`,
        "data-user-id": u.user_id,
      });

      g.appendChild(
        svgEl("circle", {
          class: "node-box",
          cx: p.x,
          cy: p.y,
          r: p.r,
          fill,
          stroke,
          "stroke-width": u.is_op ? 2.8 : 1.8,
        })
      );

      const cntTxt = svgEl("text", {
        x: p.x,
        y: p.y + 4,
        "text-anchor": "middle",
        "font-size": "11.5",
        "font-weight": "700",
        "font-family": "var(--font-mono)",
        fill: textCol,
      });
      cntTxt.textContent = String(u.reply_count || 1);
      g.appendChild(cntTxt);

      const lblTxt = svgEl("text", {
        x: p.x,
        y: p.y + p.r + 14,
        "text-anchor": "middle",
        "font-size": "10.5",
        "font-weight": "600",
        "font-family": "var(--font-mono)",
        fill: "#1E293B",
      });
      lblTxt.textContent = u.label;
      g.appendChild(lblTxt);

      if (state.showProfanityBadge && u.is_uncivil) {
        g.appendChild(
          svgEl("circle", {
            cx: p.x + p.r * 0.72,
            cy: p.y - p.r * 0.72,
            r: 7,
            fill: "#DC2626",
            stroke: "#FFFFFF",
            "stroke-width": 1.5,
          })
        );
      }

      const activate = () => selectUserNode(u.user_id);
      g.addEventListener("click", activate);
      g.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter" || ev.key === " ") {
          ev.preventDefault();
          activate();
        }
      });

      nodeLayer.appendChild(g);
    }

    if (uNodes.length) {
      const hasSel = uNodes.some((u) => u.user_id === state.selectedUserId);
      selectUserNode(hasSel ? state.selectedUserId : uNodes[0].user_id);
    }
  }

  function renderCurrentGraph() {
    if (!state.currentPayload) return;
    if (state.viewMode === "tree") {
      renderTreeSvg(state.currentPayload);
    } else {
      renderUserGraphSvg(state.currentPayload);
    }
  }

  function renderPayload(payload) {
    state.currentPayload = payload;
    state.selectedNodeId = null;
    state.selectedUserId = null;
    renderHonestyBanner(payload);
    renderTier1KPIs(payload);
    renderContractionStrip(payload);
    renderTriggerEdges(payload);
    renderCurrentGraph();
  }

  function setViewMode(mode) {
    state.viewMode = mode;
    $("view-tree-btn").setAttribute("aria-pressed", String(mode === "tree"));
    $("view-user-btn").setAttribute("aria-pressed", String(mode === "user_graph"));
    renderCurrentGraph();
  }

  // =========================================================================
  // Data Loaders (Tier 1 & Sandbox)
  // =========================================================================
  async function loadRealThreads() {
    const idx = await apiGet("/api/tier1/threads");
    state.threadsIndex = idx;

    const realBtn = $("mode-real-btn");
    realBtn.textContent = idx.is_synthetic_fallback
      ? "Mẫu Tổng hợp Offline (CI Fallback)"
      : "Bài Test VN Thật (Offline GPU)";

    const select = $("thread-select");
    clearEl(select);
    $("thread-count-label").textContent = String(idx.count || 0);

    for (const item of idx.threads || []) {
      const label =
        `${item.cascade_id} | y_polar=${fmtNum(item.actual_y_polar, 3)} | ` +
        `N=${item.actual_total_replies} (dự báo ~${Math.round(item.predicted_total_replies)}) — ` +
        `${truncText(item.title_snippet, 68)}`;
      select.appendChild(el("option", { value: item.cascade_id }, label));
    }

    if ((idx.threads || []).length > 0) {
      const firstId = idx.threads[0].cascade_id;
      select.value = firstId;
      await loadThreadDetail(firstId);
    }
  }

  async function loadThreadDetail(cascadeId) {
    showError("");
    state.currentThreadId = cascadeId;
    const payload = await apiGet(`/api/tier1/thread?id=${encodeURIComponent(cascadeId)}`);
    renderPayload(payload);
    announce(`Đã nạp chuỗi thảo luận ${cascadeId}.`);
  }

  function getSelectedJunkMode() {
    const checked = document.querySelector('input[name="junk_mode"]:checked');
    return checked ? checked.value : "contract";
  }

  async function runSandboxRequest(reqBody) {
    showError("");
    const submitBtn = $("btn-run-custom");
    submitBtn.disabled = true;
    try {
      const payload = await apiPost("/api/tier1/sandbox", reqBody);
      renderPayload(payload);
      const stats = payload.junk_filter_stats || {};
      announce(
        `Đã chạy Sandbox: giữ ${stats.kept_comments || 0} bình luận, thu gọn ${stats.contracted_nodes || 0} nút giữa, nối ${stats.relinked_edges || 0} cạnh.`
      );
    } catch (err) {
      showError(err.message);
    } finally {
      submitBtn.disabled = false;
    }
  }

  function switchTier1Mode(mode) {
    state.tier1Mode = mode;
    const isReal = mode === "real";
    $("mode-real-btn").setAttribute("aria-pressed", String(isReal));
    $("mode-sandbox-btn").setAttribute("aria-pressed", String(!isReal));
    $("real-selector-block").hidden = !isReal;
    $("sandbox-controls-block").hidden = isReal;

    if (isReal) {
      if (state.currentThreadId) {
        loadThreadDetail(state.currentThreadId).catch((e) => showError(e.message));
      } else {
        loadRealThreads().catch((e) => showError(e.message));
      }
    } else {
      runSandboxRequest({ mode: "preset_junk_1", junk_mode: getSelectedJunkMode() });
    }
  }

  // =========================================================================
  // Tab 2: Macro Network & 6-Model Signed Link Table
  // =========================================================================
  async function loadTier2Macro() {
    const data = await apiGet("/api/tier2/macro");
    state.tier2Data = data;

    const caveatsList = $("tier2-caveats-list");
    clearEl(caveatsList);
    for (const c of data.caveats || []) {
      caveatsList.appendChild(el("li", {}, c));
    }
    renderTier2Corpus(state.tier2Corpus);
  }

  function renderTier2Corpus(corpusKey) {
    state.tier2Corpus = corpusKey;
    $("corpus-vn-btn").setAttribute("aria-pressed", String(corpusKey === "vn"));
    $("corpus-pheme-btn").setAttribute("aria-pressed", String(corpusKey === "pheme"));

    if (!state.tier2Data) return;
    const corp = state.tier2Data.corpora[corpusKey];
    if (!corp) return;

    const m = corp.macro || {};
    const sb = m.structural_balance || {};
    const pol = m.polarization || {};

    $("tier2-source-pill").textContent = `edge_label_source: ${m.edge_label_source || "teacher"}`;
    $("tier2-wcc-pill").textContent =
      `WCC: ${fmtInt(m.largest_signed_wcc_nodes)} nút (${fmtPct(m.largest_signed_wcc_ratio, 1)})`;

    $("t2-q-signed").textContent = fmtNum(pol.signed_modularity, 4);
    $("t2-q-sub").textContent = `Q+: ${fmtNum(pol.q_pos, 4)} · Q-: ${fmtNum(pol.q_neg, 4)}`;

    $("t2-rwc").textContent = fmtNum(pol.rwc, 4);
    const camps = m.camp_sizes || [0, 0];
    $("t2-camps").textContent = `Phe 0: ${fmtInt(camps[0])} · Phe 1: ${fmtInt(camps[1])}`;

    $("t2-z-score").textContent = fmtSigned(sb.balance_z_score, 2);
    $("t2-triads").textContent =
      `Cân bằng: ${fmtPct(sb.balanced_ratio, 1)} vs Null: ${fmtPct(sb.null_balanced_mean, 1)} (${fmtInt(sb.num_triangles)} tam giác)`;

    const withinNeg = Number(pol.within_camp_neg_rate || 0);
    const crossNeg = Number(pol.cross_camp_neg_rate || 0);
    const deltaPp = (crossNeg - withinNeg) * 100;
    $("t2-cross-delta").textContent = `+${deltaPp.toFixed(1)} pp`;
    $("t2-neg-rates").textContent = `Liên phe: ${fmtPct(crossNeg, 1)} · Trong phe: ${fmtPct(withinNeg, 1)}`;

    $("fig5-title").textContent = `Hình 5 · ${corp.label}: Cấu trúc 2 Phe & Tam giác Có dấu`;
    $("fig5-img").src = corp.figure_url;
    $("dialog-fig-img").src = corp.figure_url;
    $("fig5-caption").textContent =
      `Tập ${corp.label}: ${fmtInt(m.num_users)} người dùng, ${fmtInt(m.num_interactions)} tương tác, ${fmtInt(m.signed_strong_pairs)} cặp có dấu mạnh.`;

    renderSignedLinkTable(corp.signed_link);
  }

  function renderSignedLinkTable(linkObj) {
    const tbody = $("signed-link-tbody");
    clearEl(tbody);
    const rows = [...(linkObj?.rows || [])];
    const metricKeys = [
      "roc_auc",
      "pr_auc",
      "macro_f1",
      "trans_roc_auc",
      "trans_pr_auc",
      "trans_macro_f1",
    ];

    const bestPerCol = {};
    for (const k of metricKeys) {
      bestPerCol[k] = Math.max(...rows.map((r) => Number(r[k]?.mean || 0)));
    }

    if (state.sortMetric) {
      const m = state.sortMetric;
      const mult = state.sortDir === "asc" ? 1 : -1;
      rows.sort((a, b) => mult * (Number(a[m]?.mean || 0) - Number(b[m]?.mean || 0)));
    }

    document.querySelectorAll(".sortable-th").forEach((th) => {
      const k = th.dataset.metric;
      if (k === state.sortMetric) {
        th.setAttribute("aria-sort", state.sortDir === "asc" ? "ascending" : "descending");
      } else {
        th.setAttribute("aria-sort", "none");
      }
    });

    for (const r of rows) {
      const tr = el("tr", { className: r.model === "SBiFAGAT-Link" ? "row-highlight" : "" });
      tr.appendChild(el("th", { scope: "row" }, r.model));
      for (const k of metricKeys) {
        const cellObj = r[k] || { mean: 0, std: 0 };
        const isBest = Math.abs(Number(cellObj.mean) - bestPerCol[k]) < 1e-6;
        const td = el(
          "td",
          { className: `tabular${isBest ? " cell-best" : ""}` },
          `${Number(cellObj.mean).toFixed(4)} ± ${Number(cellObj.std).toFixed(4)}`
        );
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }

    const sp = linkObj?.split_counts || {};
    $("signed-link-splits-note").textContent =
      `Phân hoạch thời gian: Train=${fmtInt(sp.train_edges)} cạnh | Val=${fmtInt(sp.val_edges)} | ` +
      `Test=${fmtInt(sp.test_edges)} (Transductive n=${fmtInt(sp.test_transductive_edges)} cạnh).`;
  }

  // =========================================================================
  // Initialization & Event Wiring
  // =========================================================================
  function initControls() {
    $("mode-real-btn").addEventListener("click", () => switchTier1Mode("real"));
    $("mode-sandbox-btn").addEventListener("click", () => switchTier1Mode("sandbox"));

    $("thread-select").addEventListener("change", (e) => {
      loadThreadDetail(e.target.value).catch((err) => showError(err.message));
    });

    $("view-tree-btn").addEventListener("click", () => setViewMode("tree"));
    $("view-user-btn").addEventListener("click", () => setViewMode("user_graph"));

    $("toggle-profanity-badge").addEventListener("change", (e) => {
      state.showProfanityBadge = Boolean(e.target.checked);
      renderCurrentGraph();
      if (state.viewMode === "tree" && state.selectedNodeId) {
        selectTreeNode(state.selectedNodeId);
      } else if (state.viewMode === "user_graph" && state.selectedUserId) {
        selectUserNode(state.selectedUserId);
      }
    });

    $("btn-preset-1").addEventListener("click", () => {
      runSandboxRequest({ mode: "preset_junk_1", junk_mode: getSelectedJunkMode() });
    });
    $("btn-preset-2").addEventListener("click", () => {
      runSandboxRequest({ mode: "preset_junk_2", junk_mode: getSelectedJunkMode() });
    });

    document.querySelectorAll('input[name="junk_mode"]').forEach((rad) => {
      rad.addEventListener("change", () => {
        if (state.tier1Mode !== "sandbox") return;
        const lastMode = state.currentPayload?.sandbox_mode || "preset_junk_1";
        if (lastMode === "preset_junk_1" || lastMode === "preset_junk_2") {
          runSandboxRequest({ mode: lastMode, junk_mode: getSelectedJunkMode() });
        }
      });
    });

    const rawLinesEl = $("sb-raw-lines");
    const urlInputEl = $("sb-reddit-url");
    const titleEl = $("sb-title");

    rawLinesEl.addEventListener("input", () => rawLinesEl.removeAttribute("aria-invalid"));
    urlInputEl.addEventListener("input", () => urlInputEl.removeAttribute("aria-invalid"));
    titleEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.isComposing || e.keyCode === 229)) e.preventDefault();
    });
    urlInputEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.isComposing || e.keyCode === 229)) e.preventDefault();
    });

    $("sandbox-form").addEventListener("submit", (e) => {
      e.preventDefault();
      const urlVal = urlInputEl.value.trim();
      const junkMode = getSelectedJunkMode();
      if (urlVal) {
        runSandboxRequest({ mode: "reddit_url", reddit_url: urlVal, junk_mode: junkMode });
        return;
      }
      const rawLines = rawLinesEl.value.trim();
      if (!rawLines) {
        rawLinesEl.setAttribute("aria-invalid", "true");
        showError("Vui lòng nhập ít nhất 1 bình luận hoặc dán URL Reddit.");
        rawLinesEl.focus();
        return;
      }
      runSandboxRequest({
        mode: "custom",
        title: titleEl.value.trim(),
        root_text: $("sb-root-text").value.trim(),
        raw_indented_text: rawLines,
        junk_mode: junkMode,
      });
    });

    $("corpus-vn-btn").addEventListener("click", () => renderTier2Corpus("vn"));
    $("corpus-pheme-btn").addEventListener("click", () => renderTier2Corpus("pheme"));

    document.querySelectorAll(".sort-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const m = btn.dataset.metric;
        if (state.sortMetric === m) {
          state.sortDir = state.sortDir === "desc" ? "asc" : "desc";
        } else {
          state.sortMetric = m;
          state.sortDir = "desc";
        }
        if (state.tier2Data) {
          renderSignedLinkTable(state.tier2Data.corpora[state.tier2Corpus]?.signed_link);
        }
      });
    });

    // Native <dialog> modal with Safari closedBy pointer fallback
    const dialog = $("figure-dialog");
    $("btn-zoom-fig5").addEventListener("click", () => dialog.showModal());
    $("btn-close-dialog").addEventListener("click", () => dialog.close("dismiss"));
    dialog.addEventListener("close", () => {
      dialog.returnValue = "";
    });
    if (!("closedBy" in HTMLDialogElement.prototype)) {
      let pressedOut = false;
      const isOutside = (ev) => {
        const r = dialog.getBoundingClientRect();
        return ev.clientX < r.left || ev.clientX > r.right || ev.clientY < r.top || ev.clientY > r.bottom;
      };
      dialog.addEventListener("pointerdown", (ev) => {
        pressedOut = isOutside(ev);
      });
      dialog.addEventListener("pointerup", (ev) => {
        if (pressedOut && isOutside(ev)) dialog.close("backdrop");
        pressedOut = false;
      });
    }
  }

  async function boot() {
    initTabs();
    initControls();
    try {
      const health = await apiGet("/api/health");
      const dot = $("health-dot");
      const txt = $("health-text");
      if (health.real_vn_data_available && health.gnn_checkpoint_loaded) {
        dot.dataset.ready = "true";
        txt.textContent = "Sẵn sàng · Tập Test VN + Checkpoint thật";
      } else {
        dot.dataset.ready = "warn";
        txt.textContent = "Ngoại tuyến · Chế độ Mẫu tổng hợp";
      }
      await loadRealThreads();
    } catch (err) {
      showError(`Không thể kết nối máy chủ cục bộ: ${err.message}`);
      $("health-text").textContent = "Lỗi kết nối";
    }
  }

  boot();
})();
