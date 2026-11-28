"use strict";

(() => {
  const SVG_NS = "http://www.w3.org/2000/svg";

  const STANCE_COLORS = {
    root: { fill: "#0F172A", stroke: "#0F172A", text: "#FFFFFF", edge: "#475569", label: "Bài gốc" },
    support: { fill: "#ECFDF5", stroke: "#059669", text: "#065F46", edge: "#059669", label: "Tán thành" },
    oppose: { fill: "#FEF2F2", stroke: "#DC2626", text: "#991B1B", edge: "#DC2626", label: "Phản đối" },
    neutral: { fill: "#F8FAFC", stroke: "#64748B", text: "#334155", edge: "#64748B", label: "Trung lập" },
    unscored: { fill: "#F1F5F9", stroke: "#94A3B8", text: "#475569", edge: "#CBD5E1", label: "Chưa chấm" },
  };

  const BEHAVIOR_VI = {
    root: "Bài gốc khởi nguồn",
    support: "Tán thành / Ủng hộ",
    neutral: "Trung lập / Hỏi thông tin",
    constructive_oppose: "Phản biện ôn hòa",
    hostile_conflict: "Xung đột / Gay gắt cao",
    unscored: "Chưa chấm NLP (chọn Quét sâu 100% hoặc xem mẫu phân tầng)",
  };

  const MAX_SVG_NODES = 120;

  const STAGE_LABELS_VI = {
    queued: "Đang đưa vào hàng đợi...",
    building_graph: "Đang dựng cấu trúc cây bình luận...",
    fetching_comments: "Đang tải bình luận từ bài viết...",
    filtering_junk: "Đang dọn bình luận rác...",
    scoring_edges: "AI đang đọc và chấm lập trường...",
    running_gnn: "Đang tổng hợp mạng và dự báo...",
    building_summary: "Đang trích xuất 2 luồng ý kiến...",
    completed: "Hoàn tất!",
  };

  const state = {
    activeTab: "tier1",
    tier1Mode: "real", // "real" | "sandbox"
    viewMode: "tree", // "tree" | "user_graph"
    graphZoom: 1.0,
    modalZoom: 1.0,
    graphBaseSize: { w: 900, h: 460 },
    graphRootXRatio: 0.5,
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
    activeScanJobId: null,
    pendingScanJobId: null,
    scanRunSeq: 0,
    prefilledNewsMeta: false,
    activeBranchId: null,
    megaPage: 1,
    megaTotalPages: 1,
    megaUploadedJson: null,
    megaUploadedTxt: null,
    megaPollTimer: null,
    megaSearchDebounceTimer: null,
    megaCommentsReqSeq: 0,
  };

  function detectUnsupportedSocialUrl(rawText) {
    const raw = String(rawText || "").trim();
    if (!raw || /\s/.test(raw)) return null;
    if (!/^(https?:\/\/)?[a-z0-9.-]+\.[a-z]{2,}(\/|$)/i.test(raw)) return null;
    try {
      const u = new URL(/^https?:\/\//i.test(raw) ? raw : `https://${raw}`);
      const h = (u.hostname || "").toLowerCase();
      if (
        h === "facebook.com" ||
        h.endsWith(".facebook.com") ||
        h === "fb.com" ||
        h.endsWith(".fb.com") ||
        h === "fb.watch" ||
        h.endsWith(".fb.watch")
      ) {
        return "Facebook";
      }
      if (
        h === "threads.net" ||
        h.endsWith(".threads.net") ||
        h === "threads.com" ||
        h.endsWith(".threads.com")
      ) {
        return "Threads";
      }
    } catch (_err) {
      return null;
    }
    return null;
  }

  function looksLikeSingleUrl(rawText) {
    const raw = String(rawText || "").trim();
    return /^https?:\/\/\S+$/i.test(raw);
  }

  function polarTierInfo(val, thresholds = null, stanceCounts = null) {
    const sc = stanceCounts || {};
    const sup = Number(sc.support || 0);
    const opp = Number(sc.oppose || 0);
    const neu = Number(sc.neutral || 0);
    const scoredN = sup + opp + neu;
    const q50 = Number(thresholds?.q50 ?? 0.0928);
    const q75 = Number(thresholds?.q75 ?? 0.1202);
    const q90 = Number(thresholds?.q90 ?? 0.1504);
    const hasVal = val !== null && val !== undefined && !Number.isNaN(Number(val));
    const v = hasVal ? Number(val) : 0;

    let rankShort = "Chưa rõ";
    if (hasVal) {
      if (v >= q90) rankShort = "Top 10% cao nhất VN";
      else if (v >= q75) rankShort = "Top 25% khá cao ở VN";
      else if (v >= q50) rankShort = "Mức trung bình ở VN";
      else rankShort = "Nhóm 50% thấp nhất VN";
    }

    if (scoredN >= 5 && opp / scoredN >= 0.60 && sup / scoredN <= 0.15) {
      const oppPct = Math.round((opp / scoredN) * 100);
      const supPct = Math.round((sup / scoredN) * 100);
      let detail = `Chỉ số chia rẽ thực đo: ${v.toFixed(3)} (${rankShort}).`;
      if (v >= q90) {
        detail = `Chỉ số chia rẽ thực đo: ${v.toFixed(3)} (Top 10% VN — bức xúc tập trung 1 chiều, ít ý kiến đối trọng).`;
      } else if (v >= q75) {
        detail = `Chỉ số chia rẽ thực đo: ${v.toFixed(3)} (mức khá cao — phần lớn bình luận cùng phản biện).`;
      } else if (hasVal) {
        detail = `Chỉ số chia rẽ thực đo: ${v.toFixed(3)} (thấp vì đa số đứng cùng một phía, không chia đôi 50/50).`;
      }
      return {
        label: "Phản biện 1 chiều (áp đảo)",
        badgeClass: "badge-relink",
        color: "#D97706",
        bigText: `${oppPct}%`,
        unitText: `phản biện · ${supPct}% ủng hộ`,
        oneSidedNote: detail,
        rankShort,
      };
    }
    if (scoredN >= 5 && sup / scoredN >= 0.60 && opp / scoredN <= 0.15) {
      const supPct = Math.round((sup / scoredN) * 100);
      const oppPct = Math.round((opp / scoredN) * 100);
      return {
        label: "Đồng thuận ủng hộ",
        badgeClass: "badge-support",
        color: "#059669",
        bigText: `${supPct}%`,
        unitText: `ủng hộ · ${oppPct}% phản biện`,
        oneSidedNote: `Chỉ số chia rẽ thực đo: ${v.toFixed(3)} (đa số tán thành quan điểm bài viết).`,
        rankShort,
      };
    }
    if (!hasVal) {
      return {
        label: "Chưa rõ",
        badgeClass: "badge-neutral",
        color: "#64748B",
        bigText: "—",
        unitText: "chưa có số liệu",
        oneSidedNote: "",
        rankShort,
      };
    }
    if (v >= q90) {
      return {
        label: "Chia rẽ rất cao (Top 10%)",
        badgeClass: "badge-oppose",
        color: "#DC2626",
        bigText: v.toFixed(3),
        unitText: `${rankShort} (mốc Top 10% = ${q90.toFixed(2)})`,
        oneSidedNote: "",
        rankShort,
      };
    }
    if (v >= q75) {
      return {
        label: "Tranh cãi khá cao (Top 25%)",
        badgeClass: "badge-relink",
        color: "#D97706",
        bigText: v.toFixed(3),
        unitText: `${rankShort} (mốc Top 10% = ${q90.toFixed(2)})`,
        oneSidedNote: "",
        rankShort,
      };
    }
    if (v >= q50) {
      return {
        label: "Tranh luận vừa",
        badgeClass: "badge-neutral",
        color: "#1D4ED8",
        bigText: v.toFixed(3),
        unitText: `${rankShort} (mốc Top 10% = ${q90.toFixed(2)})`,
        oneSidedNote: "",
        rankShort,
      };
    }
    return {
      label: "Ít chia rẽ (Đồng thuận)",
      badgeClass: "badge-support",
      color: "#059669",
      bigText: v.toFixed(3),
      unitText: `${rankShort} (mốc Top 10% = ${q90.toFixed(2)})`,
      oneSidedNote: "",
      rankShort,
    };
  }

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
      if (typeof box.scrollIntoView === "function") {
        box.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
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

  function fmtHumanDelay(sec) {
    const s = Math.max(0, Math.round(Number(sec || 0)));
    if (s < 60) return "vừa xong";
    if (s < 3600) return `${Math.round(s / 60)} phút`;
    if (s < 86400) return `${Math.round(s / 3600)} giờ`;
    return `${Math.round(s / 86400)} ngày`;
  }

  function buildUserDisplayMap(uNodes) {
    const map = new Map();
    let idx = 1;
    for (const u of uNodes || []) {
      if (u.is_op) {
        map.set(u.user_id, "Tác giả bài");
      } else {
        map.set(u.user_id, `Người #${idx++}`);
      }
    }
    return map;
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
      const err = new Error(data.error || `HTTP ${resp.status}`);
      err.status = resp.status;
      err.payload = data;
      throw err;
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
    } else if (ceLoaded && !isSynth && state.tier1Mode === "sandbox") {
      banner.dataset.tone = "verified";
      textEl.appendChild(
        el(
          "strong",
          {},
          "Sandbox Suy luận Cục bộ Thật (cross_encoder_loaded = true): "
        )
      );
      textEl.appendChild(
        document.createTextNode(
          "Đang chạy trực tiếp Student DualHeadCrossEncoder + encoder ViSoBERT cục bộ và trọng số GNN best_sbifagat_vn.pt trên ngữ cảnh Sandbox/tùy chỉnh."
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

    if ($("pill-nlp-backend")) $("pill-nlp-backend").textContent = `nlp: ${meta.nlp_backend || "—"}`;
    if ($("pill-cascade-id")) $("pill-cascade-id").textContent = `id: ${meta.cascade_id || "—"}`;
  }

  function renderTier1KPIs(payload) {
    const fc = payload.forecast || {};
    const meta = payload.meta || {};
    const largeInfo = payload.large_thread_analysis || {};
    const isNewsPortal = meta.domain_type === "news_portal";

    const srcBadge = $("result-source-badge");
    if (srcBadge) {
      srcBadge.hidden = false;
      if (meta.source_platform === "vnexpress") {
        srcBadge.textContent = "Nguồn: VnExpress (Cây trả lời thật)";
      } else if (meta.source_platform === "dantri") {
        srcBadge.textContent = "Nguồn: Báo Dân Trí (Cây trả lời thật)";
      } else if (meta.source_platform === "tuoitre") {
        srcBadge.textContent = "Nguồn: Báo Tuổi Trẻ (Cây trả lời thật)";
      } else if (payload.scan_job_id) {
        srcBadge.textContent = "Quét đầy đủ bài viết";
      } else if (state.tier1Mode === "sandbox") {
        srcBadge.textContent = "Dữ liệu bạn vừa nhập";
      } else {
        srcBadge.textContent = "Mẫu kiểm chứng thật (Reddit VN)";
      }
    }

    const domainBadge = $("domain-shift-badge");
    if (domainBadge) {
      if (isNewsPortal) {
        domainBadge.hidden = false;
        domainBadge.textContent = "Ngoài miền học (Báo chí · Mô hình học trên Reddit VN)";
      } else {
        domainBadge.hidden = true;
      }
    }

    const topo = payload.input_topology || {};
    const hasRealTs = topo.has_real_timestamps !== false;
    const predReplies = Number(fc.predicted_total_replies || 0);
    const actReplies = Number(fc.actual_total_replies || 0);
    const kObsVal = Number(fc.k_observed ?? actReplies);
    const gnnKVal = Number(fc.gnn_k_horizon ?? Math.min(20, kObsVal));
    $("kpi-actual-replies").textContent = fmtInt(actReplies);

    const fcLbl = $("verdict-forecast-label");
    const predLbl = $("kpi-pred-label");
    const actLbl = $("kpi-actual-label");
    const proofBadge = $("forecast-proof-badge");
    const honestyNote = $("forecast-honesty-note");
    const explainNote = $("forecast-explain-note");
    const effK = Math.min(gnnKVal, Math.max(1, kObsVal));
    if (!hasRealTs) {
      if (fcLbl) fcLbl.textContent = "Phân tích nội dung (bài dán tay)";
      if (predLbl) predLbl.textContent = "Dự báo tăng trưởng";
      if (actLbl) actLbl.textContent = "Đã nhập";
      $("kpi-pred-replies").textContent = "—";
      if (proofBadge) {
        proofBadge.textContent = "Chưa có giờ đăng thật";
        proofBadge.className = "badge-tag badge-neutral";
      }
      if (honestyNote) honestyNote.hidden = false;
      if (explainNote) explainNote.hidden = true;
    } else {
      const hasFutureCheck = Boolean(fc.has_future_replies && actReplies > effK);
      if (fcLbl) {
        fcLbl.textContent = hasFutureCheck
          ? `Kiểm chứng Dự báo sớm (che từ BL #${effK + 1})`
          : `Dự báo sớm bài mới (từ ${effK} BL đầu)`;
      }
      if (predLbl) {
        predLbl.textContent = hasFutureCheck ? `SBi-FAGAT dự báo (${effK} BL đầu)` : "AI dự báo tổng quy mô";
      }
      if (actLbl) {
        if (hasFutureCheck) {
          actLbl.textContent = payload.scan_job_id ? "Thực tế đã thu thập" : "Thực tế sau cùng";
        } else {
          actLbl.textContent = "Hiện tại mới có";
        }
      }
      $("kpi-pred-replies").textContent = `~${predReplies.toFixed(0)}`;
      if (honestyNote) honestyNote.hidden = true;
      if (explainNote) {
        explainNote.hidden = false;
        if (hasFutureCheck) {
          explainNote.textContent = isNewsPortal
            ? `Mô hình chỉ nhìn ${effK} bình luận đầu tiên để ước lượng quy mô (dữ liệu huấn luyện từ Reddit VN nên trên báo chí chỉ mang tính tham khảo).`
            : `Mô hình chỉ nhìn ${effK} bình luận sớm nhất (che hoàn toàn phần sau) để dự báo tổng quy mô và đối chứng với thực tế.`;
        } else {
          explainNote.textContent = `Bài viết mới có ${effK} bình luận. AI ước lượng tổng bình luận dự kiến khi lan truyền tiếp (chuẩn tối ưu là đủ 20 BL đầu).`;
        }
      }
      if (proofBadge) {
        if (hasFutureCheck && actReplies > 0) {
          const diff = Math.round(predReplies - actReplies);
          const absDiff = Math.abs(diff);
          const relErr = absDiff / Math.max(1, actReplies);
          if (absDiff <= 10 || relErr <= 0.25) {
            proofBadge.textContent = `Khớp sát (lệch ${absDiff} BL)`;
            proofBadge.className = "badge-tag badge-support";
          } else if (isNewsPortal) {
            proofBadge.textContent = "Ước tính tham khảo (báo chí khác nhịp Reddit)";
            proofBadge.className = "badge-tag badge-neutral";
          } else {
            proofBadge.textContent = diff < 0 ? `Thấp hơn ${absDiff} BL` : `Cao hơn ${absDiff} BL`;
            proofBadge.className = "badge-tag badge-neutral";
          }
        } else if (kObsVal < 20) {
          proofBadge.textContent = `Bài mới (${fmtInt(kObsVal)}/20 BL đầu · tham khảo)`;
          proofBadge.className = "badge-tag badge-neutral";
        } else {
          proofBadge.textContent = `Đang xem ${fmtInt(kObsVal)} bình luận`;
          proofBadge.className = "badge-tag badge-neutral";
        }
      }
    }

    const topoPill = $("topology-status-pill");
    if (topoPill) {
      if (topo.source_label_vi) {
        topoPill.textContent = topo.source_label_vi;
        topoPill.className = `pill ${topo.mode === "flat_star" ? "badge-relink" : "badge-support"}`;
        topoPill.hidden = false;
      } else {
        topoPill.hidden = true;
      }
    }
    renderGnnStructuralProof(payload);

    const sc = fc.stance_counts || { support: 0, oppose: 0, neutral: 0, unscored: 0 };
    const isFullView = meta.analysis_mode === "full" && largeInfo.full_y_polar !== undefined && largeInfo.full_y_polar !== null;
    const displayPolar = isFullView
      ? Number(largeInfo.full_y_polar)
      : Number(fc.predicted_polarization_score ?? fc.early_y_polar ?? 0);
    const thresholds = fc.regime_thresholds || { q50: 0.0928, q75: 0.1202, q90: 0.1504 };
    const tier = polarTierInfo(displayPolar, thresholds, sc);
    const polarTitleEl = $("verdict-polar-label");
    if (polarTitleEl) {
      polarTitleEl.textContent = isFullView
        ? "Kết luận toàn bài: Mức độ chia rẽ"
        : `Dự báo sớm (${effK} BL đầu): Mức chia rẽ`;
    }

    const q90Ref = Number(thresholds.q90 ?? 0.1504);
    const meterScaleMax = Math.max(0.30, q90Ref * 2.0);
    const meterPct = Math.max(4, Math.min(100, Math.round((displayPolar / meterScaleMax) * 100)));
    const lvlEl = $("verdict-polar-level");
    if (lvlEl) {
      lvlEl.textContent = tier.label;
      lvlEl.className = `badge-tag ${tier.badgeClass}`;
    }
    if ($("verdict-polar-100")) $("verdict-polar-100").textContent = tier.bigText || fmtNum(displayPolar, 3);
    if ($("verdict-polar-unit")) {
      $("verdict-polar-unit").textContent = tier.unitText || `mốc Top 10% = ${Number(q90Ref).toFixed(2)}`;
    }
    const polarSubEl = $("verdict-polar-sub");
    if (polarSubEl) {
      if (tier.oneSidedNote) {
        polarSubEl.textContent = tier.oneSidedNote;
      } else if (isFullView && fc.predicted_polarization_score !== undefined) {
        polarSubEl.textContent = `Dự báo sớm từ ${effK} BL đầu: ${fmtNum(fc.predicted_polarization_score, 3)}`;
      } else if (fc.has_future_replies && fc.actual_y_polar !== null && fc.actual_y_polar !== undefined) {
        polarSubEl.textContent = `Thực tế hậu kỳ: ${fmtNum(fc.actual_y_polar, 3)}`;
      } else {
        polarSubEl.textContent = "Đo độ cân bằng đối đầu 2 phía";
      }
    }
    const meterEl = $("verdict-polar-meter");
    if (meterEl) {
      meterEl.style.setProperty("--polar-pct", `${meterPct}%`);
      meterEl.style.setProperty("--polar-color", tier.color);
    }

    $("kpi-pred-polar").textContent = fmtNum(fc.predicted_polarization_score, 3);
    $("kpi-early-polar").textContent = fmtNum(fc.early_y_polar, 3);
    $("kpi-actual-polar").textContent =
      fc.has_future_replies && fc.actual_y_polar !== null && fc.actual_y_polar !== undefined
        ? fmtNum(fc.actual_y_polar, 3)
        : "Chưa có (bài mới)";

    $("kpi-regime").textContent = fc.predicted_regime || "—";
    $("kpi-confidence").textContent =
      fc.confidence_score !== null && fc.confidence_score !== undefined && fc.uncertainty_available !== false
        ? fmtPct(fc.confidence_score, 1)
        : "Chưa bật · N/A (disabled)";
    $("kpi-authors").textContent = fmtInt(meta.author_count);

    const scoredTotal = Math.max(1, (sc.support || 0) + (sc.neutral || 0) + (sc.oppose || 0));
    $("bar-support").style.width = `${((sc.support || 0) / scoredTotal) * 100}%`;
    $("bar-neutral").style.width = `${((sc.neutral || 0) / scoredTotal) * 100}%`;
    $("bar-oppose").style.width = `${((sc.oppose || 0) / scoredTotal) * 100}%`;

    $("kpi-k-obs").textContent =
      meta.analysis_mode === "full" && kObsVal !== gnnKVal
        ? `GNN K=${gnnKVal} · Hiển thị N=${kObsVal}`
        : `GNN K=${gnnKVal}`;
    const supPctStr = `${Math.round(((sc.support || 0) / scoredTotal) * 100)}%`;
    const neuPctStr = `${Math.round(((sc.neutral || 0) / scoredTotal) * 100)}%`;
    const oppPctStr = `${Math.round(((sc.oppose || 0) / scoredTotal) * 100)}%`;
    const unscoredPart = sc.unscored ? ` · Chưa chấm: ${fmtInt(sc.unscored)}` : "";
    $("kpi-stance-counts").textContent = isNewsPortal
      ? `Ủng hộ: ${sc.support || 0} (${supPctStr}) · Trung lập: ${sc.neutral || 0} (${neuPctStr}) · Phản biện: ${sc.oppose || 0} (${oppPctStr})${unscoredPart}`
      : `Tán thành: ${sc.support || 0} (${supPctStr}) · Trung lập: ${sc.neutral || 0} (${neuPctStr}) · Phản đối: ${sc.oppose || 0} (${oppPctStr})${unscoredPart}`;
    const scopeEl = $("verdict-stance-scope");
    if (scopeEl) {
      const scoredCntNum = (sc.support || 0) + (sc.neutral || 0) + (sc.oppose || 0);
      if (state.tier1Mode === "real") {
        scopeEl.textContent = `Tính trong ${fmtInt(scoredCntNum)} bình luận đầu (mẫu có ${fmtInt(actReplies)} BL — xem đủ bên dưới)`;
      } else if (sc.unscored) {
        scopeEl.textContent = `Đã chấm ${fmtInt(scoredCntNum)} bình luận phân tầng (chọn Quét sâu 100% để chấm hết)`;
      } else {
        scopeEl.textContent = `Tính trên ${fmtInt(scoredCntNum)} bình luận đã chấm (so với bình luận cha trực tiếp)`;
      }
    }

    const cleanTitle = String(meta.root_title || "—").trim();
    let cleanBody = String(meta.root_text || "").trim();
    if (cleanBody && cleanTitle !== "—" && cleanBody.startsWith(cleanTitle)) {
      cleanBody = cleanBody.slice(cleanTitle.length).trim();
    }
    $("root-title-display").textContent = cleanTitle;
    const bodyEl = $("root-text-display");
    if (bodyEl) {
      if (!cleanBody || cleanBody === cleanTitle) {
        bodyEl.hidden = true;
        bodyEl.textContent = "";
      } else {
        bodyEl.hidden = false;
        bodyEl.textContent = cleanBody;
      }
    }
  }

  function renderGnnStructuralProof(payload) {
    const drawer = $("gnn-proof-drawer");
    const body = $("gnn-proof-body");
    if (!drawer || !body) return;
    const probe = payload?.gnn_structural_probe;
    const topo = payload?.input_topology || {};
    if (!probe || !probe.available) {
      drawer.hidden = true;
      return;
    }
    drawer.hidden = false;
    clearEl(body);

    const kWin = fmtInt(probe.k_window_evaluated || 20);
    let lead = "";
    if (probe.has_multi_hop) {
      lead =
        `Kiểm chứng trong cửa sổ sớm ${kWin} bình luận đầu: mô hình GNN đang xét ` +
        `${fmtInt(probe.multi_hop_edges)}/${fmtInt(probe.total_gnn_edges)} cạnh đối đáp giữa bình luận với nhau (không phải phẳng 1 tầng).`;
    } else if (Number(topo.multi_hop_edges || 0) > 0) {
      lead =
        `${kWin} bình luận đầu tiên theo thời gian đều phản hồi trực tiếp bài gốc (` +
        `${fmtInt(topo.multi_hop_edges)} đối đáp tầng 2+ xuất hiện ở các bình luận đến sau), ` +
        `nên phép thử duỗi phẳng cửa sổ đầu chưa tạo độ lệch.`;
    } else {
      lead = `Cửa sổ ${kWin} bình luận đầu chỉ có phản hồi trực tiếp về bài gốc (1 tầng).`;
    }
    body.appendChild(el("p", { className: "muted-note" }, lead));

    const grid = el("div", { className: "gnn-proof-grid" });
    const cards = [
      ["Cây thật (SBi-FAGAT)", fmtNum(probe.base_polar, 3), `dự báo ~${fmtNum(probe.base_replies, 0)} BL`],
      ["Khi duỗi phẳng về gốc", fmtNum(probe.flat_polar, 3), `lệch ${fmtSigned(probe.delta_polar_structure, 3)}`],
      ["Khi xóa nhãn tán/phản đối", fmtNum(probe.zero_stance_polar, 3), `lệch ${fmtSigned(probe.delta_polar_stance, 3)}`],
    ];
    for (const [k, v, sub] of cards) {
      const cell = el("div", { className: "kv-cell" });
      cell.appendChild(el("div", { className: "kv-k" }, k));
      cell.appendChild(el("div", { className: "kv-v tabular" }, `${v} (${sub})`));
      grid.appendChild(cell);
    }
    body.appendChild(grid);
  }

  function renderContractionStrip(payload) {
    const strip = $("contraction-strip");
    const stats = payload.junk_filter_stats;
    if (state.tier1Mode !== "sandbox" || !stats) {
      strip.hidden = true;
      return;
    }
    const dropped = Number(stats.dropped_leaves || 0);
    const contracted = Number(stats.contracted_nodes || 0);
    const relinked = Number(stats.relinked_edges || 0);
    const totalActivity = dropped + contracted + relinked;
    if (totalActivity === 0) {
      strip.hidden = true;
      return;
    }
    strip.hidden = false;
    $("contraction-mode-tag").textContent = stats.mode === "prune" ? "cắt nhánh" : "nối nhánh";
    $("cs-raw").textContent = fmtInt(stats.raw_comments);
    $("cs-kept").textContent = fmtInt(stats.kept_comments);
    $("cs-dropped").textContent = fmtInt(dropped);
    $("cs-contracted").textContent = fmtInt(contracted);
    $("cs-relinked").textContent = fmtInt(relinked);
    const plainSum = $("contraction-plain-summary");
    if (plainSum) {
      const totalRemoved = dropped + contracted;
      if (stats.mode === "prune") {
        plainSum.textContent = `Đã xóa ${fmtInt(totalRemoved)} bình luận rác cùng nhánh phụ thuộc bên dưới.`;
      } else {
        plainSum.textContent =
          `Đã dọn ${fmtInt(totalRemoved)} bình luận rác ` +
          `và tự động nối lại ${fmtInt(relinked)} nhánh trả lời bên dưới.`;
      }
    }

    const pills = $("preserved-stances-pills");
    clearEl(pills);
    const shortList = payload.preserved_short_stances || [];
    if (!shortList.length) {
      pills.appendChild(el("span", { className: "muted-note" }, "Không có câu ngắn (<= 3 từ) trong mẫu này."));
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
      list.appendChild(el("li", { className: "muted-note" }, "Chưa có bình luận nổi bật."));
      return;
    }

    triggers.forEach((tr, idx) => {
      const li = el("li");
      const btn = el("button", { type: "button", className: "trigger-btn" });
      const top = el("div", { className: "trigger-btn-top" });
      const stInfo = STANCE_COLORS[tr.stance_class] || STANCE_COLORS.neutral;

      top.appendChild(
        el("span", { className: `badge-tag badge-${tr.stance_class || "neutral"}` }, stInfo.label)
      );
      top.appendChild(
        el("span", {}, idx < 2 ? `#${idx + 1} · Tác động mạnh` : `#${idx + 1} · Đáng chú ý`)
      );
      btn.appendChild(top);
      btn.appendChild(
        el("div", { className: "trigger-snip" }, tr.reply_text_snippet || tr.target_id)
      );
      btn.addEventListener("click", () => {
        if (state.viewMode !== "tree") {
          setViewMode("tree");
        }
        const match = (state.currentPayload?.tree?.nodes || []).find((n) => n.node_id === tr.target_id);
        if (match) {
          selectTreeNode(tr.target_id);
        } else {
          state.selectedNodeId = tr.target_id;
          renderTreeNodeInspector({
            node_id: tr.target_id,
            parent_id: tr.source_id,
            author_hash: "Ngoài vùng đang thu gọn",
            depth: 1,
            time_delay_sec: 0,
            text: tr.reply_text_snippet || tr.target_id,
            stance_class: tr.stance_class || "neutral",
            behavior_label: tr.h_uv >= 0.5 ? "hostile_conflict" : (tr.stance_class || "neutral"),
            s_uv: tr.s_uv ?? 0,
            h_uv: tr.h_uv ?? 0,
            gamma_td: tr.gamma_td,
            is_uncivil: Boolean(tr.is_uncivil),
            relinked: false,
          });
        }
      });
      li.appendChild(btn);
      list.appendChild(li);
    });
  }

  function renderTreeNodeInspector(node) {
    const box = $("node-inspector-body");
    clearEl(box);
    if (!node) {
      box.appendChild(
        el("p", { className: "muted-note" }, "Nhấn vào một ô trên sơ đồ để đọc chi tiết.")
      );
      return;
    }

    const tags = el("div", { className: "meta-pills" });
    const stInfo = STANCE_COLORS[node.stance_class] || STANCE_COLORS.neutral;
    tags.appendChild(el("span", { className: `badge-tag badge-${node.stance_class}` }, stInfo.label));
    if (!node.is_root) {
      tags.appendChild(el("span", { className: "pill" }, fmtHumanDelay(node.time_delay_sec)));
    }
    if (node.relinked) {
      tags.appendChild(el("span", { className: "badge-tag badge-relink" }, "Nối nhánh"));
    }
    if (state.showProfanityBadge && node.is_uncivil) {
      tags.appendChild(el("span", { className: "badge-tag badge-uncivil" }, "Công kích"));
    }
    box.appendChild(tags);

    box.appendChild(el("blockquote", { className: "detail-quote" }, node.text || "(Trống)"));
    box.appendChild(
      el("div", { className: "muted-note" }, `Sắc thái: ${BEHAVIOR_VI[node.behavior_label] || "Bình thường"}`)
    );

    const techDetails = el("details", { className: "mini-tech-details" });
    techDetails.appendChild(el("summary", {}, "Thông số kỹ thuật"));
    const kv = el("div", { className: "detail-kv-grid" });
    const entries = [
      ["Mã giả danh", node.author_hash || "—"],
      ["Tầng / Độ trễ", `Tầng ${node.depth} · +${Math.round(node.time_delay_sec || 0)}s`],
      ["Điểm lập trường", node.is_root ? "Bài gốc (+1.0)" : fmtSigned(node.s_uv, 3)],
      ["Độ gay gắt", node.is_root ? "—" : fmtNum(node.h_uv, 3)],
      ["Trọng số cổng γ", node.is_root ? "—" : (node.gamma_td !== null && node.gamma_td !== undefined ? fmtSigned(node.gamma_td, 3) : "—")],
    ];
    for (const [k, v] of entries) {
      const cell = el("div", { className: "kv-cell" });
      cell.appendChild(el("div", { className: "kv-k" }, k));
      cell.appendChild(el("div", { className: "kv-v tabular" }, v));
      kv.appendChild(cell);
    }
    techDetails.appendChild(kv);
    box.appendChild(techDetails);
  }

  function renderUserInspector(userNode) {
    const box = $("node-inspector-body");
    clearEl(box);
    if (!userNode) {
      box.appendChild(el("p", { className: "muted-note" }, "Nhấn vào một vòng tròn để xem các bình luận."));
      return;
    }

    const uMap = buildUserDisplayMap(state.currentPayload?.user_graph?.nodes || []);
    const displayName = uMap.get(userNode.user_id) || "Người dùng";

    const tags = el("div", { className: "meta-pills" });
    const campBadgeClass = userNode.camp === 0 ? "badge-support" : (userNode.camp === 1 ? "badge-oppose" : "badge-neutral");
    const campLabel = userNode.camp === 0 ? "Phe Đồng thuận" : (userNode.camp === 1 ? "Phe Phản biện" : "Trung lập");
    tags.appendChild(el("span", { className: `badge-tag ${campBadgeClass}` }, campLabel));
    tags.appendChild(el("span", { className: "pill" }, `${displayName} · ${userNode.reply_count} lượt`));
    if (state.showProfanityBadge && userNode.is_uncivil) {
      tags.appendChild(el("span", { className: "badge-tag badge-uncivil" }, "Có công kích"));
    }
    box.appendChild(tags);

    const sampleNodes = Array.isArray(userNode.sample_comments) && userNode.sample_comments.length
      ? userNode.sample_comments
      : (state.currentPayload?.tree?.nodes || []).filter((n) => n.author_hash === userNode.user_id);
    if (sampleNodes.length) {
      for (const n of sampleNodes) {
        const stInfo = STANCE_COLORS[n.stance_class] || STANCE_COLORS.neutral;
        box.appendChild(el("div", { className: "detail-quote" }, `[${stInfo.label}] ${n.text}`));
      }
    } else {
      box.appendChild(el("p", { className: "muted-note" }, "Xem thêm bình luận của người dùng này trong danh sách bên dưới."));
    }

    const techDetails = el("details", { className: "mini-tech-details" });
    techDetails.appendChild(el("summary", {}, "Thông số kỹ thuật"));
    const kv = el("div", { className: "detail-kv-grid" });
    const entries = [
      ["Mã giả danh", userNode.user_id],
      ["Số bình luận", `${userNode.reply_count} bình luận`],
      ["Cân bằng", fmtSigned(userNode.stance_balance, 3)],
      ["Nhóm cụm", campLabel],
    ];
    for (const [k, v] of entries) {
      const cell = el("div", { className: "kv-cell" });
      cell.appendChild(el("div", { className: "kv-k" }, k));
      cell.appendChild(el("div", { className: "kv-v tabular" }, v));
      kv.appendChild(cell);
    }
    techDetails.appendChild(kv);
    box.appendChild(techDetails);
  }

  function updateZoomIndicators() {
    if ($("btn-graph-zoom-reset")) {
      $("btn-graph-zoom-reset").textContent = `${Math.round(state.graphZoom * 100)}%`;
    }
    if ($("btn-modal-zoom-reset")) {
      $("btn-modal-zoom-reset").textContent = `${Math.round(state.modalZoom * 100)}%`;
    }
  }

  function applySvgZoomSizing(svg, viewport, baseW, baseH, zoomScale, rootRatio = 0.5, resetScroll = false) {
    if (!svg) return;
    const z = Math.max(0.25, Math.min(2.5, Number(zoomScale || 1.0)));
    const scaledW = Math.round(baseW * z);
    const scaledH = Math.round(baseH * z);
    svg.style.minWidth = "0";
    svg.style.width = `${scaledW}px`;
    svg.style.height = `${scaledH}px`;
    if (viewport && resetScroll) {
      const targetLeft = Math.max(0, Math.round(scaledW * rootRatio - viewport.clientWidth / 2));
      viewport.scrollLeft = targetLeft;
      viewport.scrollTop = 0;
    }
  }

  function formatModalNodePreview(item, isUser = false) {
    const box = $("modal-node-preview");
    if (!box) return;
    if (!item) {
      box.textContent = "Nhấn vào bất kỳ ô bình luận nào để đọc nội dung đầy đủ (kéo chuột để trượt, giữ Ctrl + lăn chuột để thu phóng).";
      return;
    }
    if (isUser) {
      const campLabel = item.camp === 0 ? "Phe Đồng thuận" : (item.camp === 1 ? "Phe Phản biện" : "Trung lập");
      const firstCmt = Array.isArray(item.sample_comments) && item.sample_comments.length ? item.sample_comments[0].text : "";
      box.textContent = `[${campLabel} · ${item.reply_count} lượt] ${firstCmt || item.label}`;
      return;
    }
    const stInfo = STANCE_COLORS[item.stance_class] || STANCE_COLORS.neutral;
    box.textContent = item.is_root
      ? `[Bài gốc] ${item.text}`
      : `[${stInfo.label} · Tầng ${item.depth} · ${fmtHumanDelay(item.time_delay_sec)}] ${item.text}`;
  }

  function selectTreeNode(nodeId) {
    state.selectedNodeId = nodeId;
    document.querySelectorAll(".graph-svg .svg-node-group").forEach((g) => {
      g.classList.toggle("is-selected", g.dataset.nodeId === nodeId);
    });
    const node = (state.currentPayload?.tree?.nodes || []).find((n) => n.node_id === nodeId);
    renderTreeNodeInspector(node || null);
    formatModalNodePreview(node || null, false);
  }

  function selectUserNode(userId) {
    state.selectedUserId = userId;
    document.querySelectorAll(".graph-svg .svg-node-group").forEach((g) => {
      g.classList.toggle("is-selected", g.dataset.userId === userId);
    });
    const userNode = (state.currentPayload?.user_graph?.nodes || []).find((u) => u.user_id === userId);
    renderUserInspector(userNode || null);
    formatModalNodePreview(userNode || null, true);
  }

  // =========================================================================
  // Pure Vanilla SVG Renderers (Tree Layout + Thread User Graph)
  // =========================================================================
  function renderTreeSvg(
    payload,
    svg = $("graph-svg"),
    viewport = document.querySelector("#panel-tier1 .graph-stage-card .svg-viewport"),
    zoomScale = state.graphZoom,
    resetScroll = false
  ) {
    if (!svg) return;
    clearEl(svg);
    const tree = payload.tree || { nodes: [], edges: [] };
    const rawNodes = tree.nodes || [];
    const rawEdges = tree.edges || [];
    if (!rawNodes.length) return;

    // Hard browser-freeze guard: BFS from root up to MAX_SVG_NODES so all kept nodes stay connected.
    let nodes = rawNodes;
    let edges = rawEdges;
    if (rawNodes.length > MAX_SVG_NODES) {
      const rootCand = rawNodes.find((n) => n.is_root) || rawNodes[0];
      const byId = new Map(rawNodes.map((n) => [n.node_id, n]));
      const chAdj = new Map(rawNodes.map((n) => [n.node_id, []]));
      for (const n of rawNodes) {
        if (n.parent_id && chAdj.has(n.parent_id)) {
          chAdj.get(n.parent_id).push(n.node_id);
        }
      }
      const bfsNodes = [];
      const seenBfs = new Set([rootCand.node_id]);
      const queue = [rootCand.node_id];
      while (queue.length && bfsNodes.length < MAX_SVG_NODES) {
        const curId = queue.shift();
        const curNode = byId.get(curId);
        if (curNode) bfsNodes.push(curNode);
        for (const chId of chAdj.get(curId) || []) {
          if (!seenBfs.has(chId) && bfsNodes.length + queue.length < MAX_SVG_NODES * 2) {
            seenBfs.add(chId);
            queue.push(chId);
          }
        }
      }
      nodes = bfsNodes;
      const keepSet = new Set(nodes.map((n) => n.node_id));
      edges = rawEdges.filter((e) => keepSet.has(e.source_id) && keepSet.has(e.target_id));
    }

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
    const rootPos = pos.get(root.node_id);
    const rootRatio = rootPos ? rootPos.cx / Math.max(1, svgW) : 0.5;
    state.graphBaseSize = { w: svgW, h: svgH };
    state.graphRootXRatio = rootRatio;

    svg.setAttribute("viewBox", `0 0 ${svgW} ${svgH}`);
    applySvgZoomSizing(svg, viewport, svgW, svgH, zoomScale, rootRatio, resetScroll);
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
        "font-family": "var(--font-sans)",
        fill: stInfo.text,
      });
      headerTxt.textContent = n.is_root
        ? "Bài gốc"
        : `${stInfo.label} · ${fmtHumanDelay(n.time_delay_sec)}`;
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

  function renderUserGraphSvg(
    payload,
    svg = $("graph-svg"),
    viewport = document.querySelector("#panel-tier1 .graph-stage-card .svg-viewport"),
    zoomScale = state.graphZoom,
    resetScroll = false
  ) {
    if (!svg) return;
    clearEl(svg);
    const ug = payload.user_graph || { nodes: [], edges: [], camp_counts: { camp0: 0, camp1: 0, camp2: 0 } };
    const uNodes = ug.nodes || [];
    const uEdges = ug.edges || [];

    const svgW = 900;
    const camp0 = uNodes.filter((u) => u.camp === 0);
    const camp1 = uNodes.filter((u) => u.camp === 1);
    const camp2 = uNodes.filter((u) => u.camp === 2);
    const hasNeutral = camp2.length > 0;
    const maxCampLen = Math.max(camp0.length, camp1.length, camp2.length, 1);
    const svgH = Math.max(460, 120 + maxCampLen * 62);

    state.graphBaseSize = { w: svgW, h: svgH };
    state.graphRootXRatio = 0.5;
    svg.setAttribute("viewBox", `0 0 ${svgW} ${svgH}`);
    applySvgZoomSizing(svg, viewport, svgW, svgH, zoomScale, 0.5, resetScroll);
    svg.setAttribute(
      "aria-label",
      `Đồ thị tương tác người dùng: ${camp0.length} đồng thuận, ${camp2.length} trung lập, ${camp1.length} phản biện`
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

    const bgLayer = svgEl("g");
    const c0X = hasNeutral ? 165 : 225;
    const c2X = 450;
    const c1X = hasNeutral ? 735 : 675;
    if (hasNeutral) {
      for (const divX of [305, 595]) {
        bgLayer.appendChild(
          svgEl("line", {
            x1: divX,
            y1: 24,
            x2: divX,
            y2: svgH - 20,
            stroke: "#CBD5E1",
            "stroke-width": 1.4,
            "stroke-dasharray": "5 5",
          })
        );
      }
    } else {
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
    }

    const mkTitle = (xPos, fillCol, textVal) => {
      const t = svgEl("text", {
        x: xPos,
        y: 34,
        "text-anchor": "middle",
        "font-size": "13",
        "font-weight": "700",
        fill: fillCol,
      });
      t.textContent = textVal;
      bgLayer.appendChild(t);
    };
    mkTitle(c0X, "#065F46", `Đồng thuận / Gốc (${camp0.length})`);
    if (hasNeutral) {
      mkTitle(c2X, "#475569", `Trung lập (${camp2.length})`);
    }
    mkTitle(c1X, "#991B1B", `Phản biện / Đối lập (${camp1.length})`);
    svg.appendChild(bgLayer);

    const pos = new Map();
    function layoutCamp(arr, baseCenterX, bendDir, bendSpan) {
      const topY = 78;
      const botY = svgH - 52;
      const span = Math.max(1, botY - topY);
      arr.forEach((u, i) => {
        const frac = arr.length <= 1 ? 0.5 : i / (arr.length - 1);
        const y = topY + frac * span;
        const arc = Math.sin(frac * Math.PI) * bendSpan * bendDir;
        const x = baseCenterX + arc;
        const r = 16 + Math.min(11, (u.reply_count || 1) * 2.4);
        pos.set(u.user_id, { x, y, r });
      });
    }

    layoutCamp(camp0, c0X, -1, hasNeutral ? 45 : 90);
    if (hasNeutral) layoutCamp(camp2, c2X, 0, 0);
    layoutCamp(camp1, c1X, 1, hasNeutral ? 45 : 90);

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

    const uMap = buildUserDisplayMap(uNodes);
    for (const u of uNodes) {
      const p = pos.get(u.user_id);
      if (!p) continue;
      const fill = u.camp === 0 ? "#ECFDF5" : (u.camp === 1 ? "#FEF2F2" : "#F8FAFC");
      const stroke = u.is_op ? "#0F172A" : (u.camp === 0 ? "#059669" : (u.camp === 1 ? "#DC2626" : "#64748B"));
      const textCol = u.camp === 0 ? "#065F46" : (u.camp === 1 ? "#991B1B" : "#334155");
      const campDesc = u.camp === 0 ? "Phe Đồng thuận" : (u.camp === 1 ? "Phe Phản biện" : "Trung lập");
      const friendlyName = uMap.get(u.user_id) || "Người dùng";

      const g = svgEl("g", {
        class: `svg-node-group${state.selectedUserId === u.user_id ? " is-selected" : ""}`,
        tabindex: "0",
        role: "button",
        "aria-label": `${friendlyName}, ${campDesc}, ${u.reply_count} bình luận`,
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
        "font-size": "11",
        "font-weight": "600",
        "font-family": "var(--font-sans)",
        fill: "#1E293B",
      });
      lblTxt.textContent = friendlyName;
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

  function renderCurrentGraph(resetScroll = false) {
    if (!state.currentPayload) return;
    const inlineVp = document.querySelector("#panel-tier1 .graph-stage-card .svg-viewport");
    if (state.viewMode === "tree") {
      renderTreeSvg(state.currentPayload, $("graph-svg"), inlineVp, state.graphZoom, resetScroll);
    } else {
      renderUserGraphSvg(state.currentPayload, $("graph-svg"), inlineVp, state.graphZoom, resetScroll);
    }
    const modalDlg = $("graph-fullscreen-dialog");
    if (modalDlg && modalDlg.open) {
      const modalVp = $("modal-svg-viewport");
      if (state.viewMode === "tree") {
        renderTreeSvg(state.currentPayload, $("modal-graph-svg"), modalVp, state.modalZoom, resetScroll);
      } else {
        renderUserGraphSvg(state.currentPayload, $("modal-graph-svg"), modalVp, state.modalZoom, resetScroll);
      }
    }
    updateZoomIndicators();
  }

  function setGraphZoom(nextZoom, isModal = false, resetScroll = false) {
    const clamped = Math.max(0.25, Math.min(2.5, Math.round(Number(nextZoom) * 100) / 100));
    if (isModal) {
      state.modalZoom = clamped;
      applySvgZoomSizing(
        $("modal-graph-svg"),
        $("modal-svg-viewport"),
        state.graphBaseSize.w,
        state.graphBaseSize.h,
        clamped,
        state.graphRootXRatio,
        resetScroll
      );
    } else {
      state.graphZoom = clamped;
      applySvgZoomSizing(
        $("graph-svg"),
        document.querySelector("#panel-tier1 .graph-stage-card .svg-viewport"),
        state.graphBaseSize.w,
        state.graphBaseSize.h,
        clamped,
        state.graphRootXRatio,
        resetScroll
      );
    }
    updateZoomIndicators();
  }

  function fitGraphZoom(isModal = false) {
    const vp = isModal
      ? $("modal-svg-viewport")
      : document.querySelector("#panel-tier1 .graph-stage-card .svg-viewport");
    const availW = vp ? Math.max(320, vp.clientWidth - 28) : 900;
    const fitRatio = Math.max(0.25, Math.min(1.0, availW / Math.max(1, state.graphBaseSize.w)));
    setGraphZoom(fitRatio, isModal, true);
  }

  function wirePanAndWheel(viewport, isModal = false) {
    if (!viewport) return;
    let dragging = false;
    let dragged = false;
    let startX = 0;
    let startY = 0;
    let origLeft = 0;
    let origTop = 0;

    viewport.addEventListener("pointerdown", (ev) => {
      if (ev.button !== 0) return;
      if (ev.target.closest && ev.target.closest(".svg-node-group")) return;
      dragging = true;
      dragged = false;
      startX = ev.clientX;
      startY = ev.clientY;
      origLeft = viewport.scrollLeft;
      origTop = viewport.scrollTop;
      viewport.classList.add("is-panning");
    });
    viewport.addEventListener("pointermove", (ev) => {
      if (!dragging) return;
      const dx = ev.clientX - startX;
      const dy = ev.clientY - startY;
      if (Math.abs(dx) > 4 || Math.abs(dy) > 4) dragged = true;
      viewport.scrollLeft = origLeft - dx;
      viewport.scrollTop = origTop - dy;
    });
    const endPan = () => {
      if (!dragging) return;
      dragging = false;
      viewport.classList.remove("is-panning");
      if (dragged) {
        // swallow accidental click right after pan
        const suppress = (ce) => { ce.stopPropagation(); };
        viewport.addEventListener("click", suppress, { capture: true, once: true });
      }
    };
    viewport.addEventListener("pointerup", endPan);
    viewport.addEventListener("pointercancel", endPan);
    viewport.addEventListener("wheel", (ev) => {
      if (!ev.ctrlKey && !ev.metaKey) return;
      ev.preventDefault();
      const cur = isModal ? state.modalZoom : state.graphZoom;
      const delta = ev.deltaY < 0 ? 0.15 : -0.15;
      setGraphZoom(cur + delta, isModal, false);
    }, { passive: false });
  }

  function createNode(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function createSvgNode(tag, attrs) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    if (attrs) {
      Object.entries(attrs).forEach(([key, val]) => {
        if (val !== undefined && val !== null) node.setAttribute(key, String(val));
      });
    }
    return node;
  }

  function clearNode(node) {
    if (!node) return;
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function renderTopicAnalysis(payload) {
    const card = $("ai-understanding-card");
    const topicInfo = payload?.topic_analysis;
    if (!card || !topicInfo) return;

    const pill = $("topic-backend-pill");
    if (pill) {
      const backend = topicInfo.topic_backend || "lexical_ngram_fallback";
      const loaded = Boolean(topicInfo.transformer_topic_loaded);
      pill.textContent = loaded ? `${backend} (Transformer)` : `${backend} (Fallback)`;
      pill.classList.toggle("is-real", loaded);
    }

    const primaryLabel = $("topic-primary-label");
    if (primaryLabel) {
      primaryLabel.textContent = topicInfo.primary_topic || "Chưa xác định";
    }

    const primaryConf = $("topic-primary-conf");
    if (primaryConf) {
      const conf = Number(topicInfo.primary_confidence ?? 0);
      primaryConf.textContent = `${Math.round(conf * 100)}%`;
    }

    const secondaryLine = $("topic-secondary-line");
    if (secondaryLine) {
      const secs = Array.isArray(topicInfo.secondary_topics) ? topicInfo.secondary_topics : [];
      if (secs.length === 0) {
        secondaryLine.textContent = topicInfo.is_custom_taxonomy
          ? "Phân loại theo danh sách chủ đề tùy chỉnh."
          : "Không có chủ đề phụ nổi bật.";
      } else {
        const names = secs.map((s) => {
          if (typeof s === "string") return s;
          const lbl = s.label || s.topic || s.id || "";
          const sc = s.score ?? s.confidence;
          return sc !== undefined ? `${lbl} (${Math.round(Number(sc) * 100)}%)` : lbl;
        });
        secondaryLine.textContent = `Chủ đề liên quan: ${names.join(", ")}`;
      }
    }

    const barsList = $("topic-bars-list");
    if (barsList) {
      clearNode(barsList);
      const topics = Array.isArray(topicInfo.topics) ? topicInfo.topics.slice(0, 6) : [];
      topics.forEach((item) => {
        const score = Math.max(0, Math.min(1, Number(item.score ?? item.confidence ?? 0)));
        const labelText = item.label || item.topic || item.id || "Chủ đề";
        const row = createNode("li", "topic-bar-row");
        const head = createNode("div", "topic-bar-head");
        head.appendChild(createNode("span", "topic-bar-name", labelText));
        head.appendChild(createNode("span", "topic-bar-pct tabular-nums", `${Math.round(score * 100)}%`));
        const prog = document.createElement("progress");
        prog.className = "topic-progress";
        prog.max = 100;
        prog.value = Math.round(score * 100);
        prog.setAttribute("aria-label", `${labelText}: ${Math.round(score * 100)}%`);
        row.appendChild(head);
        row.appendChild(prog);
        barsList.appendChild(row);
      });
    }
  }

  function renderTrajectorySvg(largeInfo) {
    const svg = $("trajectory-svg");
    const caption = $("trajectory-caption");
    if (!svg) return;
    clearNode(svg);

    const points = Array.isArray(largeInfo?.polarization_trajectory)
      ? largeInfo.polarization_trajectory
      : [];
    if (points.length === 0) {
      const empty = createSvgNode("text", {
        x: 210,
        y: 80,
        "text-anchor": "middle",
        fill: "#64748b",
        "font-size": 12,
      });
      empty.textContent = "Chưa đủ mốc bình luận để vẽ diễn biến chia rẽ.";
      svg.appendChild(empty);
      if (caption) caption.textContent = "Chưa có dữ liệu diễn biến.";
      return;
    }

    const W = 420;
    const H = 160;
    const padL = 44;
    const padR = 16;
    const padT = 16;
    const padB = 28;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

    const rawYVals = points.map((p) => Math.max(0, Math.min(1, Number(p.y_polar ?? p.polar ?? 0))));
    const rawMaxY = Math.max(0.16, ...rawYVals);
    const yCeil = Math.min(1.0, Math.max(0.25, Math.ceil((rawMaxY * 1.25) / 0.05) * 0.05));
    const yTicks = [0, yCeil * 0.5, yCeil];

    yTicks.forEach((val) => {
      const ratio = val / Math.max(1e-6, yCeil);
      const y = padT + (1 - ratio) * plotH;
      svg.appendChild(
        createSvgNode("line", {
          x1: padL,
          y1: y,
          x2: W - padR,
          y2: y,
          stroke: "#e2e8f0",
          "stroke-width": 1,
          "stroke-dasharray": val === 0 ? "" : "3 3",
        })
      );
      const lbl = createSvgNode("text", {
        x: padL - 6,
        y: y + 4,
        "text-anchor": "end",
        fill: "#64748b",
        "font-size": 10,
      });
      lbl.textContent = val.toFixed(2);
      svg.appendChild(lbl);
    });

    // Subtle VN Q90 reference line if within visible Y range
    if (0.1504 <= yCeil) {
      const q90Y = padT + (1 - 0.1504 / yCeil) * plotH;
      svg.appendChild(
        createSvgNode("line", {
          x1: padL,
          y1: q90Y,
          x2: W - padR,
          y2: q90Y,
          stroke: "#fca5a5",
          "stroke-width": 1,
          "stroke-dasharray": "4 4",
        })
      );
    }

    const kVals = points.map((p, idx) => Number(p.k ?? p.milestone_k ?? p.n_replies ?? idx + 1));
    const minK = Math.min(...kVals);
    const maxK = Math.max(minK + 1, ...kVals);
    const useLogScale = maxK >= 500;
    const tx = (kVal) => (useLogScale ? Math.log1p(Math.max(0, kVal)) : kVal);
    const minT = tx(minK);
    const maxT = Math.max(minT + 1e-6, tx(maxK));
    const coords = points.map((p, idx) => {
      const kVal = kVals[idx];
      const yVal = rawYVals[idx];
      const x =
        points.length === 1
          ? padL + plotW / 2
          : padL + ((tx(kVal) - minT) / Math.max(1e-6, maxT - minT)) * plotW;
      const y = padT + (1 - Math.min(1, yVal / yCeil)) * plotH;
      return { x, y, kVal, yVal, label: `${kVal} BL` };
    });

    if (coords.length >= 2) {
      const d = coords
        .map((c, idx) => `${idx === 0 ? "M" : "L"} ${c.x.toFixed(1)} ${c.y.toFixed(1)}`)
        .join(" ");
      svg.appendChild(
        createSvgNode("path", {
          d,
          fill: "none",
          stroke: "#2563eb",
          "stroke-width": 2.4,
          "stroke-linecap": "round",
          "stroke-linejoin": "round",
        })
      );
    }

    const labelStep = coords.length > 7 ? Math.ceil(coords.length / 6) : 1;
    let lastLblX = -999;
    coords.forEach((c, idx) => {
      const dot = createSvgNode("circle", {
        cx: c.x.toFixed(1),
        cy: c.y.toFixed(1),
        r: 4.2,
        fill: "#1d4ed8",
        stroke: "#ffffff",
        "stroke-width": 1.5,
      });
      const title = createSvgNode("title");
      title.textContent = `Mốc ${c.label}: điểm chia rẽ ${c.yVal.toFixed(3)} (tổng ${maxK} BL)`;
      dot.appendChild(title);
      svg.appendChild(dot);

      const isEdgeTick = idx === 0 || idx === coords.length - 1;
      if ((isEdgeTick || idx % labelStep === 0) && (c.x - lastLblX >= 42 || isEdgeTick)) {
        if (!isEdgeTick || c.x - lastLblX >= 32) {
          lastLblX = c.x;
          const xLbl = createSvgNode("text", {
            x: c.x.toFixed(1),
            y: H - 8,
            "text-anchor": "middle",
            fill: "#475569",
            "font-size": 10,
          });
          xLbl.textContent = c.kVal >= 1000 ? `${(c.kVal / 1000).toFixed(c.kVal % 1000 === 0 ? 0 : 1)}k BL` : `${c.kVal} BL`;
          svg.appendChild(xLbl);
        }
      }
    });

    if (caption) {
      const first = coords[0];
      const last = coords[coords.length - 1];
      if (coords.length === 1) {
        caption.textContent = `Điểm chia rẽ tại mốc ${first.kVal} bình luận: ${first.yVal.toFixed(3)} (vạch đỏ đứt là mức Q90 = 0.150).`;
      } else {
        const diff = last.yVal - first.yVal;
        const dirWord =
          diff >= 0.05 ? "tăng mạnh" : diff >= 0.025 ? "tăng nhẹ" : diff <= -0.05 ? "giảm mạnh" : diff <= -0.025 ? "giảm nhẹ" : "khá ổn định";
        const scaleNote = useLogScale ? " · trục ngang theo thang log" : "";
        caption.textContent =
          `Mức chia rẽ ${dirWord}: từ ${first.yVal.toFixed(3)} (lúc ${first.kVal} BL) ` +
          `đến ${last.yVal.toFixed(3)} (lúc ${last.kVal} BL)${scaleNote}.`;
      }
    }
  }

  function jumpToCommentsWithStance(stanceVal) {
    const sel = $("mega-stance-filter");
    const sec = $("mega-comments-section");
    if (sel) {
      sel.value = stanceVal;
    }
    state.megaPage = 1;
    if (state.activeScanJobId || state.tier1Mode === "real") {
      loadMegaCommentsPage(1).catch(() => {});
    } else if (Array.isArray(state.currentPayload?.comment_rows)) {
      renderClientCommentsPage(1);
    }
    if (sec && !sec.hidden) {
      sec.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  function appendArgumentGroup(container, titleText, items, emptyText, openByDefault, tone = "neutral", jumpInfo = null) {
    const buildList = () => {
      if (items.length === 0) {
        return createNode("p", "arguments-empty", emptyText);
      }
      const ul = createNode("ul", "arguments-list");
      items.slice(0, 3).forEach((entry) => {
        const li = createNode("li", "arguments-item");
        if (typeof entry === "string") {
          li.textContent = entry;
        } else {
          const text = entry.text || entry.snippet || entry.summary || "";
          const replyHint = entry.child_replies ? ` (${entry.child_replies} phản hồi)` : "";
          li.textContent = `${text}${replyHint}`;
        }
        ul.appendChild(li);
      });
      return ul;
    };

    if (tone === "flash") {
      if (items.length === 0) return;
      const details = createNode("details", "arguments-details");
      details.dataset.tone = tone;
      const summary = createNode("summary", "arguments-summary", `${titleText} (${items.length})`);
      details.appendChild(summary);
      details.appendChild(buildList());
      container.appendChild(details);
      return;
    }

    const panel = createNode("div", "arg-panel");
    panel.dataset.tone = tone;
    panel.appendChild(createNode("h3", "arg-panel-title", titleText));
    panel.appendChild(buildList());
    if (jumpInfo && jumpInfo.count > 0) {
      const jBtn = createNode(
        "button",
        "arg-jump-btn",
        `Xem toàn bộ ${fmtInt(jumpInfo.count)} bình luận phe này ↓`
      );
      jBtn.type = "button";
      jBtn.addEventListener("click", () => jumpToCommentsWithStance(jumpInfo.stance));
      panel.appendChild(jBtn);
    }
    container.appendChild(panel);
  }

  function renderLargePostUnderstanding(payload) {
    const largeInfo = payload?.large_thread_analysis;
    if (!largeInfo) return;

    const scopePill = $("large-scope-pill");
    if (scopePill) {
      scopePill.hidden = false;
      if (largeInfo.scope === "full_analysis") {
        scopePill.textContent = "Đọc toàn chuỗi";
      } else if (largeInfo.scope === "early_post_hoc") {
        scopePill.textContent = "Đối chứng toàn bài";
      } else {
        scopePill.textContent = "Cửa sổ sớm";
      }
    }

    const scaleBadge = $("large-scale-badge");
    if (scaleBadge) {
      scaleBadge.hidden = false;
      scaleBadge.textContent = `${largeInfo.scale_label_vi || largeInfo.scale_tier || "Chuỗi"} · ${fmtInt(largeInfo.total_replies_analyzed ?? 0)} BL`;
    }

    const archBadge = $("large-archetype-badge");
    if (archBadge) {
      archBadge.hidden = false;
      archBadge.textContent = largeInfo.post_archetype_label_vi || largeInfo.post_archetype || "Thảo luận";
    }

    const trendBadge = $("large-trend-badge");
    if (trendBadge) {
      trendBadge.hidden = false;
      const delta = Number(largeInfo.polar_delta ?? 0);
      const sign = delta >= 0 ? "+" : "";
      trendBadge.textContent = `${largeInfo.evolution_trend_label_vi || largeInfo.evolution_trend || "Ổn định"} (${sign}${delta.toFixed(3)})`;
    }

    const oodBadge = $("ood-scale-badge");
    const fc = payload.forecast || {};
    if (oodBadge) {
      const isOod = Boolean(fc.ood_scale_warning || payload.mega_scan_summary?.ood_scale_warning);
      oodBadge.hidden = !isOod;
      if (isOod) {
        const refMax = fc.training_max_replies_ref || payload.mega_scan_summary?.training_max_replies_ref || 345;
        oodBadge.textContent = `Bài quy mô lớn (> ${refMax} BL trong tập học)`;
      }
    }

    const covBadge = $("mega-coverage-badge");
    const ms = payload.mega_scan_summary || {};
    if (covBadge) {
      const isPartial = Boolean(ms.is_stratified_subsampled && ms.total_edges && ms.scored_edges);
      covBadge.hidden = !isPartial;
      if (isPartial) {
        const pct = Math.round(Number(ms.coverage_ratio ?? 1) * 100);
        const ci = Array.isArray(ms.bootstrap_ci_95) ? ` · CI 95% [${ms.bootstrap_ci_95[0].toFixed(3)}, ${ms.bootstrap_ci_95[1].toFixed(3)}]` : "";
        covBadge.textContent = `Lấy mẫu phân tầng ${fmtInt(ms.scored_edges)}/${fmtInt(ms.total_edges)} (${pct}%)${ci}`;
      }
    }

    const sem = largeInfo.semantic_understanding || {};
    const summaryEl = $("large-summary-text");
    if (summaryEl) {
      const txt = sem.summary_vi || "";
      summaryEl.hidden = !txt;
      summaryEl.textContent = txt;
    }

    renderTrajectorySvg(largeInfo);

    const cols = $("arguments-columns");
    if (cols) {
      clearNode(cols);
      const proItems = Array.isArray(sem.top_pro_arguments)
        ? sem.top_pro_arguments
        : Array.isArray(sem.pro_arguments)
          ? sem.pro_arguments
          : [];
      const antiItems = Array.isArray(sem.top_anti_arguments)
        ? sem.top_anti_arguments
        : Array.isArray(sem.anti_arguments)
          ? sem.anti_arguments
          : [];
      const flashItems = Array.isArray(sem.flashpoint_comments)
        ? sem.flashpoint_comments
        : Array.isArray(sem.flashpoints)
          ? sem.flashpoints
          : [];
      const sc = payload?.forecast?.stance_counts || {};
      const isNews = payload?.meta?.domain_type === "news_portal";
      appendArgumentGroup(
        cols,
        isNews ? "Luồng Ủng hộ / Tích cực" : "Phe Tán thành nói gì?",
        proItems,
        "Không có bình luận ủng hộ rõ rệt trong bài viết này.",
        true,
        "pro",
        { stance: "support", count: Number(sc.support || 0) }
      );
      appendArgumentGroup(
        cols,
        isNews ? "Luồng Phản biện / Bức xúc" : "Phe Phản đối nói gì?",
        antiItems,
        "Không có bình luận phản đối rõ rệt.",
        true,
        "anti",
        { stance: "oppose", count: Number(sc.oppose || 0) }
      );
      appendArgumentGroup(cols, "Điểm nóng / Bình luận đảo chiều", flashItems, "Chưa phát hiện điểm đảo chiều xung đột.", false, "flash");
    }
  }

  function renderHotspotBranches(payload) {
    const bar = $("hotspot-branch-bar");
    if (!bar) return;
    clearEl(bar);
    const branches = Array.isArray(payload?.large_thread_analysis?.hotspot_branches)
      ? payload.large_thread_analysis.hotspot_branches
      : [];
    const jobId = payload?.scan_job_id || state.activeScanJobId;
    if (!branches.length || !jobId) {
      bar.hidden = true;
      return;
    }
    bar.hidden = false;
    bar.appendChild(el("span", { className: "field-label" }, "Lọc nhánh sơ đồ:"));

    const activeBid = payload?.tree_view_meta?.active_branch_id || state.activeBranchId || "_all_top";
    const mkBtn = (bid, label) => {
      const btn = el(
        "button",
        {
          type: "button",
          className: "hotspot-btn",
          "aria-pressed": String(activeBid === bid),
        },
        label
      );
      btn.addEventListener("click", () => switchMegaBranch(jobId, bid));
      return btn;
    };
    const totalN = payload?.tree_view_meta?.total_nodes ? Math.max(0, Number(payload.tree_view_meta.total_nodes) - 1) : null;
    bar.appendChild(
      mkBtn("_all_top", totalN ? `Toàn cảnh (${fmtInt(totalN)} BL)` : "Toàn cảnh (tất cả)")
    );
    branches.forEach((b, idx) => {
      const snip = b.lead_text ? `: "${truncText(b.lead_text, 24)}"` : "";
      bar.appendChild(
        mkBtn(b.branch_id, `Nhánh #${idx + 1} (${b.size} BL · ${Math.round((b.oppose_ratio || 0) * 100)}% phản đối)${snip}`)
      );
    });
  }

  async function switchMegaBranch(jobId, branchId) {
    try {
      const resp = await apiGet(`/api/tier1/scan_jobs/${encodeURIComponent(jobId)}/branch/${encodeURIComponent(branchId)}`);
      if (!state.currentPayload) return;
      state.activeBranchId = branchId;
      state.currentPayload = {
        ...state.currentPayload,
        tree: resp.tree,
        tree_view_meta: resp.tree_view_meta,
      };
      if (state.viewMode !== "tree") {
        setViewMode("tree");
      } else {
        renderCurrentGraph();
      }
      renderHotspotBranches(state.currentPayload);
      state.megaPage = 1;
      await loadMegaCommentsPage(1);
    } catch (err) {
      showError(err.message);
    }
  }

  function renderStanceFilterChips(stanceTotals, currentVal) {
    const row = $("stance-filter-chips");
    if (!row) return;
    clearEl(row);
    const t = stanceTotals || {};
    const chips = [
      ["all", `Tất cả (${fmtInt(t.all ?? 0)})`],
      ["support", `Tán thành (${fmtInt(t.support ?? 0)})`],
      ["oppose", `Phản đối (${fmtInt(t.oppose ?? 0)})`],
      ["neutral", `Trung lập (${fmtInt(t.neutral ?? 0)})`],
      ["uncivil", `Công kích (${fmtInt(t.uncivil ?? 0)})`],
    ];
    if (Number(t.unscored || 0) > 0) {
      chips.push(["unscored", `Chưa chấm (${fmtInt(t.unscored)})`]);
    }
    for (const [val, label] of chips) {
      const btn = el(
        "button",
        { type: "button", className: "stance-filter-chip", "aria-pressed": String(currentVal === val) },
        label
      );
      btn.addEventListener("click", () => {
        if ($("mega-stance-filter")) $("mega-stance-filter").value = val;
        state.megaPage = 1;
        if (state.activeScanJobId || state.tier1Mode === "real") {
          loadMegaCommentsPage(1).catch((e) => showError(e.message));
        } else {
          renderClientCommentsPage(1);
        }
      });
      row.appendChild(btn);
    }
  }

  function renderActiveBranchBanner() {
    const note = $("active-branch-note");
    if (!note) return;
    clearEl(note);
    if (!state.activeScanJobId || !state.activeBranchId || state.activeBranchId === "_all_top") {
      note.hidden = true;
      return;
    }
    note.hidden = false;
    note.appendChild(
      el("span", {}, "Đang lọc danh sách bình luận theo nhánh nóng đã chọn.")
    );
    const resetBtn = el("button", { type: "button", className: "btn-outline btn-sm" }, "Hiện tất cả bình luận");
    resetBtn.addEventListener("click", () => switchMegaBranch(state.activeScanJobId, "_all_top"));
    note.appendChild(resetBtn);
  }

  function renderCommentsTableRows(items, startIdx) {
    const tbody = $("mega-comments-tbody");
    clearEl(tbody);
    if (!items.length) {
      const tr = el("tr");
      const td = el("td", { colSpan: "3" }, "Không tìm thấy bình luận nào phù hợp bộ lọc.");
      tr.appendChild(td);
      tbody.appendChild(tr);
      return;
    }
    items.forEach((item, idx) => {
      const tr = el("tr");
      tr.appendChild(el("td", { className: "tabular" }, String(startIdx + idx + 1)));
      const stanceKey = item.is_scored === false ? "unscored" : item.stance_class;
      const stInfo = STANCE_COLORS[stanceKey] || STANCE_COLORS.neutral;
      const stTd = el("td");
      stTd.appendChild(
        el(
          "span",
          { className: `badge-tag badge-${stanceKey === "unscored" ? "neutral" : stanceKey}` },
          stInfo.label
        )
      );
      const metaRow = el("div", { className: "comment-row-meta" });
      metaRow.appendChild(el("span", { className: "pill tabular" }, `Tầng ${item.depth ?? 1}`));
      if (item.is_early_k) {
        metaRow.appendChild(el("span", { className: "pill" }, "20 BL đầu"));
      }
      if (state.showProfanityBadge && item.is_uncivil) {
        metaRow.appendChild(el("span", { className: "badge-tag badge-uncivil" }, "Gay gắt"));
      }
      stTd.appendChild(metaRow);
      tr.appendChild(stTd);

      const textTd = el("td");
      if (item.parent_snippet && Number(item.depth || 1) >= 2) {
        textTd.appendChild(
          el("div", { className: "comment-parent-ctx" }, `Trả lời: "${item.parent_snippet}"`)
        );
      }
      textTd.appendChild(el("div", {}, item.text || ""));
      tr.appendChild(textTd);
      tbody.appendChild(tr);
    });
  }

  function renderClientCommentsPage(pageNum = 1) {
    const sec = $("mega-comments-section");
    const rows = Array.isArray(state.currentPayload?.comment_rows) ? state.currentPayload.comment_rows : [];
    if (!sec) return;
    if (!rows.length) {
      sec.hidden = true;
      return;
    }
    const stance = $("mega-stance-filter") ? $("mega-stance-filter").value : "all";
    const q = $("mega-search-input") ? $("mega-search-input").value.trim().toLowerCase() : "";
    const stanceTotals = {
      all: rows.length,
      support: rows.filter((r) => r.is_scored !== false && r.stance_class === "support").length,
      oppose: rows.filter((r) => r.is_scored !== false && r.stance_class === "oppose").length,
      neutral: rows.filter((r) => r.is_scored !== false && r.stance_class === "neutral").length,
      uncivil: rows.filter((r) => Boolean(r.is_uncivil)).length,
      unscored: rows.filter((r) => r.is_scored === false || r.stance_class === "unscored").length,
    };
    renderStanceFilterChips(stanceTotals, stance);
    renderActiveBranchBanner();

    const filtered = rows.filter((r) => {
      if (stance === "uncivil" && !r.is_uncivil) return false;
      if (stance === "unscored" && r.is_scored !== false && r.stance_class !== "unscored") return false;
      if (["support", "oppose", "neutral"].includes(stance) && (r.is_scored === false || r.stance_class !== stance)) return false;
      if (q && !String(r.text || "").toLowerCase().includes(q)) return false;
      return true;
    });

    const pageSize = 50;
    const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
    const curPage = Math.max(1, Math.min(totalPages, Number(pageNum || 1)));
    const startIdx = (curPage - 1) * pageSize;
    sec.hidden = false;
    state.megaPage = curPage;
    state.megaTotalPages = totalPages;
    $("mega-comments-total-pill").textContent = `${fmtInt(filtered.length)} / ${fmtInt(rows.length)} bình luận`;
    $("mega-page-info").textContent = `Trang ${curPage} / ${totalPages}`;
    $("mega-prev-page").disabled = curPage <= 1;
    $("mega-next-page").disabled = curPage >= totalPages;
    renderCommentsTableRows(filtered.slice(startIdx, startIdx + pageSize), startIdx);
  }

  async function loadMegaCommentsPage(pageNum = 1) {
    const sec = $("mega-comments-section");
    const tbody = $("mega-comments-tbody");
    const jobId = state.activeScanJobId;
    const realCid = state.tier1Mode === "real" ? (state.currentThreadId || state.currentPayload?.meta?.cascade_id) : null;
    if (!sec || !tbody || (!jobId && !realCid)) {
      if (Array.isArray(state.currentPayload?.comment_rows) && state.currentPayload.comment_rows.length) {
        renderClientCommentsPage(pageNum);
      } else if (sec) {
        sec.hidden = true;
      }
      return;
    }
    const reqSeq = ++state.megaCommentsReqSeq;
    const stance = $("mega-stance-filter") ? $("mega-stance-filter").value : "all";
    const q = $("mega-search-input") ? $("mega-search-input").value.trim() : "";
    const qs = new URLSearchParams({
      page: String(pageNum),
      page_size: "50",
      stance,
      q,
    });
    if (jobId && state.activeBranchId && state.activeBranchId !== "_all_top") {
      qs.set("branch_id", state.activeBranchId);
    }
    const endpoint = jobId
      ? `/api/tier1/scan_jobs/${encodeURIComponent(jobId)}/comments?${qs.toString()}`
      : `/api/tier1/thread/comments?id=${encodeURIComponent(realCid)}&${qs.toString()}`;
    const data = await apiGet(endpoint);
    if (reqSeq !== state.megaCommentsReqSeq) {
      return;
    }
    sec.hidden = false;
    state.megaPage = data.page || 1;
    state.megaTotalPages = data.total_pages || 1;
    $("mega-comments-total-pill").textContent = `${fmtInt(data.total_items)} bình luận khớp`;
    $("mega-page-info").textContent = `Trang ${state.megaPage} / ${state.megaTotalPages}`;
    $("mega-prev-page").disabled = state.megaPage <= 1;
    $("mega-next-page").disabled = state.megaPage >= state.megaTotalPages;
    if (data.stance_totals) {
      renderStanceFilterChips(data.stance_totals, stance);
    }
    renderActiveBranchBanner();
    const startIdx = (state.megaPage - 1) * 50;
    renderCommentsTableRows(data.items || [], startIdx);
  }

  function renderPayload(payload) {
    if (!payload.scan_job_id && state.megaPollTimer) {
      clearTimeout(state.megaPollTimer);
      state.megaPollTimer = null;
      state.pendingScanJobId = null;
      setMegaButtonsBusy(false);
    }
    if (payload.scan_job_id || payload.sandbox_mode) {
      state.tier1Mode = "sandbox";
    }
    state.currentPayload = payload;
    state.selectedNodeId = null;
    state.selectedUserId = null;
    if (payload.scan_job_id) {
      state.activeScanJobId = payload.scan_job_id;
      state.activeBranchId = payload.tree_view_meta?.active_branch_id || "_all_top";
    } else {
      state.activeScanJobId = null;
      state.activeBranchId = null;
    }
    renderHonestyBanner(payload);
    renderTier1KPIs(payload);
    renderTopicAnalysis(payload);
    renderLargePostUnderstanding(payload);
    renderContractionStrip(payload);
    renderTriggerEdges(payload);
    renderHotspotBranches(payload);
    renderCurrentGraph(true);
    if (state.activeScanJobId || state.tier1Mode === "real") {
      loadMegaCommentsPage(1).catch((err) => showError(err.message));
    } else {
      renderClientCommentsPage(1);
    }
  }

  function setViewMode(mode) {
    state.viewMode = mode;
    $("view-tree-btn").setAttribute("aria-pressed", String(mode === "tree"));
    $("view-user-btn").setAttribute("aria-pressed", String(mode === "user_graph"));
    renderCurrentGraph(true);
  }

  // =========================================================================
  // Data Loaders (Tier 1 & Sandbox)
  // =========================================================================
  async function loadRealThreads() {
    const idx = await apiGet("/api/tier1/threads");
    state.threadsIndex = idx;

    const realBtn = $("mode-real-btn");
    if (realBtn) {
      realBtn.textContent = idx.is_synthetic_fallback
        ? "Mẫu thử Offline"
        : "10 bài mẫu thật";
    }

    const select = $("thread-select");
    clearEl(select);
    if ($("thread-count-label")) $("thread-count-label").textContent = String(idx.count || 0);

    (idx.threads || []).forEach((item, i) => {
      const topicPart = item.primary_topic ? ` · ${item.primary_topic}` : "";
      const label =
        `Bài #${i + 1} · ${fmtInt(item.actual_total_replies)} BL${topicPart} — ` +
        `${truncText(item.title_snippet, 64)}`;
      select.appendChild(el("option", { value: item.cascade_id }, label));
    });

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

  function getSelectedAnalysisMode() {
    const checked = document.querySelector('input[name="analysis_mode"]:checked');
    return checked ? checked.value : "early";
  }

  function getLiveTopics() {
    const sb = $("sb-custom-topics");
    return sb ? sb.value.trim() : "";
  }

  async function runSandboxRequest(reqBody) {
    showError("");
    const submitBtn = $("btn-run-custom");
    if (submitBtn) submitBtn.disabled = true;
    try {
      const analysisMode = reqBody.analysis_mode || getSelectedAnalysisMode();
      const customTopics =
        reqBody.custom_topics !== undefined ? reqBody.custom_topics : getLiveTopics();
      const finalBody = {
        ...reqBody,
        analysis_mode: analysisMode,
        k_horizon: reqBody.k_horizon || 20,
        ...(analysisMode === "full"
          ? { max_full_comments: reqBody.max_full_comments || 500 }
          : {}),
        ...(customTopics ? { custom_topics: customTopics } : {}),
      };
      const payload = await apiPost("/api/tier1/sandbox", finalBody);
      renderPayload(payload);
      const stats = payload.junk_filter_stats || {};
      const topicLbl = payload.topic_analysis?.primary_topic || "Chưa phân loại";
      announce(
        `Đã phân tích xong: chủ đề ${topicLbl}, giữ ${stats.kept_comments || 0} bình luận, nối lại ${stats.relinked_edges || 0} nhánh.`
      );
    } catch (err) {
      showError(err.message);
    } finally {
      if (submitBtn) submitBtn.disabled = false;
    }
  }

  function switchTier1Mode(mode, runDefaultIfEmpty = false) {
    state.scanRunSeq += 1;
    state.pendingScanJobId = null;
    if (state.megaPollTimer) {
      clearTimeout(state.megaPollTimer);
      state.megaPollTimer = null;
      setMegaButtonsBusy(false);
    }
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
    } else if (runDefaultIfEmpty && !state.currentPayload) {
      runSandboxRequest({ mode: "preset_junk_1", junk_mode: getSelectedJunkMode() });
    }
  }

  // =========================================================================
  // Tab 2: Macro Network & 10-Model Signed Link Table
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

    const activeUsers = m.active_graph_users ?? m.num_users;
    const wccRatioActive =
      m.largest_signed_wcc_ratio_active ??
      (activeUsers ? Number(m.largest_signed_wcc_nodes || 0) / Number(activeUsers) : m.largest_signed_wcc_ratio);

    if ($("tier2-source-pill")) {
      $("tier2-source-pill").textContent = `nguồn nhãn cạnh: ${m.edge_label_source || "teacher"}`;
    }
    if ($("tier2-wcc-pill")) {
      $("tier2-wcc-pill").textContent =
        `WCC: ${fmtInt(m.largest_signed_wcc_nodes)} / ${fmtInt(activeUsers)} hoạt động (${fmtPct(wccRatioActive, 1)})`;
    }

    const nullQ = Number(pol.q_signed_null_mean ?? pol.null_signed_modularity_mean ?? pol.null_q_signed ?? (corpusKey === "vn" ? 0.1977 : 0.4716));
    const qPval = pol.q_signed_p_value ?? (corpusKey === "vn" ? 0.333 : 1.0);
    $("t2-q-signed").textContent = fmtNum(pol.signed_modularity, 3);
    $("t2-q-sub").textContent =
      corpusKey === "vn"
        ? `Đối chứng ngẫu nhiên: ${fmtNum(nullQ, 3)} (p=${Number(qPval).toFixed(2)}, chưa vượt ngẫu nhiên)`
        : `Đối chứng ngẫu nhiên: ${fmtNum(nullQ, 3)} (tách cụm theo sự kiện, không phải phòng vang)`;

    $("t2-rwc").textContent = fmtNum(pol.rwc, 4);
    const camps = m.camp_sizes || [0, 0];
    const rwcMeaning = Number(pol.rwc || 0) < 0 ? "Âm = tranh luận chéo mở" : "Dương = thiên hướng khép kín";
    $("t2-camps").textContent = `${rwcMeaning} · 2 nhóm WCC: ${fmtInt(camps[0])} vs ${fmtInt(camps[1])} người`;

    $("t2-z-score").textContent = fmtSigned(sb.balance_z_score, 2);
    const davisPct = sb.weak_balance_ratio ?? sb.weak_balance_davis_ratio ?? (corpusKey === "vn" ? 0.9405 : 0.8774);
    $("t2-triads").textContent =
      `Cân bằng yếu Davis: ${fmtPct(davisPct, 2)} (Tam giác 2 phe chặt: ${fmtPct(sb.balanced_ratio, 1)})`;

    const withinNeg = Number(pol.within_camp_neg_rate || 0);
    const crossNeg = Number(pol.cross_camp_neg_rate || 0);
    const deltaPp = (crossNeg - withinNeg) * 100;
    const negRatio = withinNeg > 1e-6 ? (crossNeg / withinNeg).toFixed(1) : "nhiều";
    $("t2-cross-delta").textContent = `${deltaPp >= 0 ? "+" : ""}${deltaPp.toFixed(1)} điểm %`;
    $("t2-neg-rates").textContent = `Liên phe ${fmtPct(crossNeg, 1)} · Trong phe ${fmtPct(withinNeg, 1)}`;

    const plainSum = $("tier2-plain-summary-text");
    if (plainSum) {
      clearEl(plainSum);
      plainSum.appendChild(el("strong", {}, "Tóm tắt: "));
      if (corpusKey === "vn") {
        plainSum.appendChild(
          document.createTextNode(
            `Cộng đồng Việt Nam không đóng kín thành 2 "phòng vang" biệt lập (RWC = ${fmtNum(pol.rwc, 4)}), mà tranh luận chéo rất mạnh (${fmtPct(crossNeg, 1)} tương tác đối đầu nằm giữa các nhóm).`
          )
        );
      } else {
        plainSum.appendChild(
          document.createTextNode(
            `Trên Twitter (PHEME), các nhóm sự kiện tách cụm rõ rệt hơn nhiều (đối đầu liên phe ${fmtPct(crossNeg, 1)}, gấp ${negRatio} lần trong phe ${fmtPct(withinNeg, 1)}).`
          )
        );
      }
    }

    $("fig5-title").textContent = `Bản đồ 2 phe · ${corp.label}`;
    $("fig5-img").src = corp.figure_url;
    $("dialog-fig-img").src = corp.figure_url;
    $("fig5-caption").textContent =
      `${corp.label}: ${fmtInt(activeUsers)} người dùng hoạt động, ${fmtInt(m.num_interactions)} tương tác.`;

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
  function getSelectedMegaStrategy() {
    const checked = document.querySelector('input[name="mega_strategy"]:checked');
    return checked ? checked.value : "stratified";
  }

  function getSelectedMegaMaxComments() {
    const sel = $("mega-max-comments");
    return sel ? Number(sel.value || 2000) : 2000;
  }

  function setMegaButtonsBusy(busy) {
    const btnStress = $("btn-mega-stress");
    const btnRun = $("btn-mega-run");
    const btnMainRun = $("btn-run-custom");
    const btnCancel = $("btn-mega-cancel");
    if (btnStress) btnStress.disabled = busy;
    if (btnRun) btnRun.disabled = busy;
    if (btnMainRun) btnMainRun.disabled = busy;
    if (btnCancel) btnCancel.hidden = !busy;
  }

  function clearPrefilledNewsMeta() {
    state.prefilledNewsMeta = false;
    if ($("sb-title")) $("sb-title").value = "";
    if ($("sb-root-text")) $("sb-root-text").value = "";
    if ($("sb-raw-lines")) {
      $("sb-raw-lines").placeholder = "Dán bài gốc và các bình luận (hỗ trợ @Tên để giữ đúng nhánh trả lời)...";
    }
    if ($("prefilled-meta-bar")) $("prefilled-meta-bar").hidden = true;
    if ($("prefilled-meta-label")) $("prefilled-meta-label").textContent = "";
    showSocialUrlGuide(null);
  }

  function showNewsMetaGuide(articleMeta, messageText) {
    const box = $("social-url-guide");
    const ttl = $("social-guide-title");
    const desc = $("social-guide-desc");
    if (!box) return;
    if (ttl) {
      const rawTitle = String(articleMeta?.title || "").trim();
      const shortTitle = rawTitle.length > 72 ? `${rawTitle.slice(0, 71).trim()}…` : rawTitle;
      const hostLabel = articleMeta?.hostname || "trang báo";
      ttl.textContent = shortTitle
        ? `Đã đọc tiêu đề từ ${hostLabel}: "${shortTitle}" — hãy dán bình luận bạn đọc để phân tích`
        : `Đã đọc tiêu đề từ ${hostLabel} — hãy dán bình luận bạn đọc để phân tích`;
    }
    if (desc) {
      desc.textContent =
        messageText ||
        "Mở bài báo, quét khối (Ctrl+C) phần bình luận bên dưới bài viết rồi dán vào ô văn bản vừa mở ở dưới (hệ thống tự lọc chữ thừa).";
    }
    box.hidden = false;
  }

  async function startMegaScanJob(reqBody) {
    showError("");
    const runSeq = ++state.scanRunSeq;
    if (state.megaPollTimer) {
      clearTimeout(state.megaPollTimer);
      state.megaPollTimer = null;
    }
    state.pendingScanJobId = null;
    const progBox = $("mega-progress-box");
    const progBar = $("mega-progress-bar");
    const progLbl = $("mega-progress-label");
    const progPct = $("mega-progress-pct");
    const progMeta = $("mega-progress-meta");
    if (progBox) progBox.hidden = false;
    if (progBar) progBar.value = 0;
    if (progLbl) progLbl.textContent = STAGE_LABELS_VI.queued;
    if (progPct) progPct.textContent = "0%";
    if (progMeta) progMeta.textContent = "";
    setMegaButtonsBusy(true);

    try {
      const created = await apiPost("/api/tier1/scan_jobs", reqBody);
      if (runSeq !== state.scanRunSeq) {
        return;
      }
      const jobId = created.job_id;
      state.pendingScanJobId = jobId;

      const poll = async () => {
        if (runSeq !== state.scanRunSeq) {
          return;
        }
        try {
          const st = await apiGet(`/api/tier1/scan_jobs/${encodeURIComponent(jobId)}`);
          if (runSeq !== state.scanRunSeq) {
            return;
          }
          const pct = Math.max(0, Math.min(100, Number(st.percent || 0)));
          if (progBar) progBar.value = pct;
          if (progPct) progPct.textContent = `${pct.toFixed(1)}%`;
          const stageKey = st.stage || st.status || "queued";
          if (progLbl) progLbl.textContent = STAGE_LABELS_VI[stageKey] || `Đang xử lý (${stageKey})...`;
          if (progMeta) {
            if (stageKey === "fetching_comments") {
              progMeta.textContent = `Đã tải ${fmtInt(st.processed_edges)} bình luận...`;
            } else {
              const etaTxt = st.eta_sec !== null && st.eta_sec !== undefined ? ` · còn ~${st.eta_sec}s` : "";
              progMeta.textContent = `${fmtInt(st.processed_edges)} / ${fmtInt(st.total_edges)} bình luận${etaTxt}`;
            }
          }
          if (st.status === "completed") {
            state.pendingScanJobId = null;
            state.megaPollTimer = null;
            setMegaButtonsBusy(false);
            if (progBox) progBox.hidden = true;
            if (!st.result) {
              showError("Tác vụ đã hoàn tất nhưng thiếu dữ liệu kết quả.");
              return;
            }
            renderPayload(st.result);
            const ms = st.result.mega_scan_summary || {};
            announce(`Hoàn tất phân tích (${fmtInt(ms.kept_comments)} bình luận hợp lệ).`);
            return;
          }
          if (st.status === "cancelled" || st.status === "failed") {
            state.pendingScanJobId = null;
            state.megaPollTimer = null;
            setMegaButtonsBusy(false);
            if (progBox) progBox.hidden = true;
            if (st.error) showError(st.error);
            return;
          }
          state.megaPollTimer = setTimeout(poll, 450);
        } catch (err) {
          if (runSeq !== state.scanRunSeq) {
            return;
          }
          state.pendingScanJobId = null;
          state.megaPollTimer = null;
          setMegaButtonsBusy(false);
          if (progBox) progBox.hidden = true;
          showError(err.message);
        }
      };
      await poll();
    } catch (err) {
      if (runSeq !== state.scanRunSeq) {
        return;
      }
      state.pendingScanJobId = null;
      setMegaButtonsBusy(false);
      if (progBox) progBox.hidden = true;
      if (err?.payload?.needs_manual_comments && err.payload.article_meta) {
        const am = err.payload.article_meta;
        const rawTitle = String(am.title || "").trim();
        const shortTitle = rawTitle.length > 72 ? `${rawTitle.slice(0, 71).trim()}…` : rawTitle;
        const hostLabel = am.hostname || "trang báo";
        if ($("sb-title") && am.title) $("sb-title").value = am.title;
        if ($("sb-root-text") && am.description) $("sb-root-text").value = am.description;
        state.prefilledNewsMeta = Boolean(am.title || am.description);
        if ($("prefilled-meta-bar") && state.prefilledNewsMeta) {
          if ($("prefilled-meta-label")) {
            $("prefilled-meta-label").textContent = shortTitle
              ? `Đang dùng tiêu đề lấy từ ${hostLabel}: "${shortTitle}"`
              : `Đã điền sẵn mô tả từ ${hostLabel}`;
          }
          $("prefilled-meta-bar").hidden = false;
        }
        if ($("sb-raw-lines")) {
          if (shortTitle) {
            $("sb-raw-lines").placeholder =
              `Đã giữ sẵn tiêu đề: "${shortTitle}" — chỉ cần Ctrl+V dán bình luận bạn đọc vào đây rồi bấm Phân tích...`;
          }
        }
        if ($("paste-drawer")) $("paste-drawer").open = true;
        if ($("sb-reddit-url")) $("sb-reddit-url").value = "";
        showNewsMetaGuide(am, err.message);
        if ($("sb-raw-lines")) $("sb-raw-lines").focus();
        return;
      }
      showError(err.message);
    }
  }

  function showSocialUrlGuide(platform) {
    const box = $("social-url-guide");
    const ttl = $("social-guide-title");
    const desc = $("social-guide-desc");
    if (!box) return;
    if (!platform) {
      box.hidden = true;
      return;
    }
    if (ttl) {
      ttl.textContent = `${platform} yêu cầu đăng nhập nên công cụ không tự lấy link được`;
    }
    if (desc) {
      desc.textContent =
        `Hãy bôi đen bài viết + bình luận trên ${platform}, nhấn Ctrl+C rồi dán vào mục "Hoặc dán trực tiếp nội dung" ở ngay bên dưới.`;
    }
    box.hidden = false;
  }

  function initControls() {
    $("mode-real-btn").addEventListener("click", () => switchTier1Mode("real"));
    $("mode-sandbox-btn").addEventListener("click", () => switchTier1Mode("sandbox", false));

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

    if ($("btn-preset-1")) {
      $("btn-preset-1").addEventListener("click", () => {
        runSandboxRequest({ mode: "preset_junk_1", junk_mode: getSelectedJunkMode() });
      });
    }
    if ($("btn-preset-2")) {
      $("btn-preset-2").addEventListener("click", () => {
        runSandboxRequest({ mode: "preset_junk_2", junk_mode: getSelectedJunkMode() });
      });
    }
    const presetLargeBtn = $("btn-preset-large");
    if (presetLargeBtn) {
      presetLargeBtn.addEventListener("click", () => {
        const fullRadio = $("analysis-full");
        if (fullRadio) fullRadio.checked = true;
        runSandboxRequest({
          mode: "preset_large_1",
          junk_mode: getSelectedJunkMode(),
          analysis_mode: "full",
          k_horizon: 20,
          max_full_comments: 500,
        });
      });
    }
    const btnClearPrefilled = $("btn-clear-prefilled-meta");
    if (btnClearPrefilled) {
      btnClearPrefilled.addEventListener("click", () => {
        clearPrefilledNewsMeta();
        announce("Đã xóa tiêu đề bài báo điền sẵn.");
      });
    }
    const btnOpenPaste = $("btn-open-paste-drawer");
    if (btnOpenPaste) {
      btnOpenPaste.addEventListener("click", () => {
        const drawer = $("paste-drawer");
        if (drawer) drawer.open = true;
        const urlIn = $("sb-reddit-url");
        if (urlIn) urlIn.value = "";
        showSocialUrlGuide(null);
        const rawEl = $("sb-raw-lines");
        if (rawEl) rawEl.focus();
      });
    }

    document.querySelectorAll('input[name="junk_mode"], input[name="analysis_mode"]').forEach((rad) => {
      rad.addEventListener("change", () => {
        if (state.tier1Mode !== "sandbox") return;
        const lastMode = state.currentPayload?.sandbox_mode || "preset_junk_1";
        if (lastMode === "preset_junk_1" || lastMode === "preset_junk_2" || lastMode === "preset_large_1") {
          runSandboxRequest({ mode: lastMode, junk_mode: getSelectedJunkMode() });
        }
      });
    });

    const rawLinesEl = $("sb-raw-lines");
    const urlInputEl = $("sb-reddit-url");
    const titleEl = $("sb-title");
    const liveTopicsEl = $("sb-custom-topics");
    const reclassifyForm = $("topic-reclassify-form");

    rawLinesEl.addEventListener("input", () => {
      rawLinesEl.removeAttribute("aria-invalid");
      showError("");
      const trimmed = rawLinesEl.value.trim();
      const socialInText = detectUnsupportedSocialUrl(trimmed);
      if (socialInText) {
        showSocialUrlGuide(socialInText);
      } else if (!detectUnsupportedSocialUrl(urlInputEl.value)) {
        showSocialUrlGuide(null);
      }
    });
    urlInputEl.addEventListener("input", () => {
      urlInputEl.removeAttribute("aria-invalid");
      showError("");
      const socialPlatform = detectUnsupportedSocialUrl(urlInputEl.value);
      showSocialUrlGuide(socialPlatform);
    });

    const rbParentSel = $("rb-parent-select");
    const rbTextInput = $("rb-text-input");
    const rbAddBtn = $("btn-rb-add");
    let rbSeq = 0;
    const appendReplyNode = () => {
      if (!rbTextInput || !rbParentSel) return;
      const msg = rbTextInput.value.trim();
      if (!msg) return;
      const parentVal = rbParentSel.value || "root";
      const curLines = rawLinesEl.value.split(/\r?\n/).filter((ln) => ln.trim().length > 0);
      if (curLines.length === 0) {
        const curTitle = titleEl && titleEl.value.trim() ? titleEl.value.trim() : "Thảo luận mới từ người dùng";
        rawLinesEl.value = `${curTitle}\n`;
      } else if (!rawLinesEl.value.endsWith("\n")) {
        rawLinesEl.value += "\n";
      }
      rbSeq += 1;
      const cid = `c${rbSeq}`;
      rawLinesEl.value += `${cid} (trả lời ${parentVal}): ${msg}\n`;
      rbParentSel.appendChild(el("option", { value: cid }, `Trả lời: BL #${rbSeq} (${truncText(msg, 28)})`));
      rbTextInput.value = "";
      rbTextInput.focus();
    };
    if (rbAddBtn) rbAddBtn.addEventListener("click", appendReplyNode);
    if (rbTextInput) {
      rbTextInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          appendReplyNode();
        }
      });
    }
    if (titleEl) {
      titleEl.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && (e.isComposing || e.keyCode === 229)) e.preventDefault();
      });
    }
    urlInputEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.isComposing || e.keyCode === 229)) e.preventDefault();
    });
    if (liveTopicsEl) {
      liveTopicsEl.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && (e.isComposing || e.keyCode === 229)) e.preventDefault();
      });
    }

    if (reclassifyForm) {
      reclassifyForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const current = state.currentPayload;
        if (!current) return;
        const nodes = Array.isArray(current.tree?.nodes)
          ? current.tree.nodes
          : Array.isArray(current.nodes)
            ? current.nodes
            : [];
        const root = nodes.find((n) => n.is_root || n.depth === 0) || nodes[0] || {};
        const snippets = nodes
          .filter((n) => n !== root && n.text)
          .slice(0, 6)
          .map((n) => n.text);
        const rawTopics = liveTopicsEl ? liveTopicsEl.value.trim() : "";
        const btn = $("btn-reclassify-topic");
        if (btn) btn.disabled = true;
        showError("");
        try {
          const resp = await apiPost("/api/tier1/classify_topic", {
            title: current.meta?.root_title || current.title || "",
            root_text: current.meta?.root_text || root.text || "",
            reply_snippets: snippets,
            custom_topics: rawTopics,
            lang: current.meta?.lang || "vi",
          });
          const topicPayload = resp.topic_analysis || resp;
          const updatedLarge = current.large_thread_analysis
            ? {
                ...current.large_thread_analysis,
                semantic_understanding: {
                  ...(current.large_thread_analysis.semantic_understanding || {}),
                },
              }
            : null;
          if (updatedLarge?.semantic_understanding) {
            const pTopic = topicPayload.primary_topic || "Chưa xác định";
            const pPct = `${(Number(topicPayload.primary_confidence ?? 0) * 100).toFixed(1)}%`;
            const curSum = updatedLarge.semantic_understanding.summary_vi || "";
            if (curSum) {
              updatedLarge.semantic_understanding.summary_vi = curSum.replace(
                /Chủ đề chính:\s*[^.(]+\s*\(độ tin cậy\s*[0-9.,]+%\)/,
                `Chủ đề chính: ${pTopic} (độ tin cậy ${pPct})`
              );
              updatedLarge.semantic_understanding.summary =
                updatedLarge.semantic_understanding.summary_vi;
            }
          }
          state.currentPayload = {
            ...current,
            topic_analysis: topicPayload,
            ...(updatedLarge ? { large_thread_analysis: updatedLarge } : {}),
          };
          renderTopicAnalysis(state.currentPayload);
          if (updatedLarge) renderLargePostUnderstanding(state.currentPayload);
          announce(`Đã phân loại lại chủ đề: ${topicPayload.primary_topic || "Hoàn tất"}.`);
        } catch (err) {
          showError(err.message);
        } finally {
          if (btn) btn.disabled = false;
        }
      });
    }

    $("sandbox-form").addEventListener("submit", (e) => {
      e.preventDefault();
      const junkMode = getSelectedJunkMode();
      const customTopics = getLiveTopics() || undefined;
      const maxC = getSelectedMegaMaxComments();

      if (state.megaUploadedJson) {
        if (state.prefilledNewsMeta) clearPrefilledNewsMeta();
        startMegaScanJob({
          mode: "json_upload",
          thread_json: state.megaUploadedJson,
          max_full_comments: maxC,
          scan_strategy: getSelectedMegaStrategy(),
          junk_mode: junkMode,
          custom_topics: customTopics,
        });
        return;
      }

      const rawLinesInput = rawLinesEl.value.trim();
      const urlFieldVal = urlInputEl.value.trim();
      const effectiveUrl = urlFieldVal || (looksLikeSingleUrl(rawLinesInput) ? rawLinesInput : "");

      if (effectiveUrl) {
        if (state.prefilledNewsMeta) clearPrefilledNewsMeta();
        const socialPlatform = detectUnsupportedSocialUrl(effectiveUrl);
        if (socialPlatform) {
          showSocialUrlGuide(socialPlatform);
          if ($("paste-drawer")) $("paste-drawer").open = true;
          showError(
            `${socialPlatform} chặn cào link tự động do yêu cầu đăng nhập. Vui lòng copy bài viết + bình luận trên ${socialPlatform} rồi dán vào mục 'Hoặc dán trực tiếp nội dung' ở ngay bên dưới.`
          );
          rawLinesEl.focus();
          return;
        }
        startMegaScanJob({
          mode: "reddit_url",
          reddit_url: effectiveUrl,
          max_full_comments: maxC,
          scan_strategy: getSelectedMegaStrategy(),
          junk_mode: junkMode,
          custom_topics: customTopics,
        });
        return;
      }

      const effectiveText = rawLinesInput || state.megaUploadedTxt || "";
      if (!effectiveText) {
        rawLinesEl.setAttribute("aria-invalid", "true");
        showError("Vui lòng dán link bài báo (VnExpress, Dân Trí, Tuổi Trẻ, VietnamNet...), dán nội dung bình luận hoặc chọn tệp (.txt, .json).");
        rawLinesEl.focus();
        return;
      }

      const manualTitle = titleEl ? titleEl.value.trim() : "";
      const manualRoot = $("sb-root-text") ? $("sb-root-text").value.trim() : "";
      const autoSplit = !manualTitle && !manualRoot;
      const nonEmptyCount = effectiveText.split(/\r?\n/).filter((ln) => ln.trim().length > 0).length;

      if (nonEmptyCount > 100) {
        startMegaScanJob({
          mode: "custom",
          title: manualTitle,
          root_text: manualRoot,
          raw_indented_text: effectiveText,
          auto_split_root: autoSplit,
          max_full_comments: maxC,
          scan_strategy: nonEmptyCount <= 500 ? "full" : getSelectedMegaStrategy(),
          junk_mode: junkMode,
          custom_topics: customTopics,
        });
        return;
      }

      runSandboxRequest({
        mode: "custom",
        title: manualTitle,
        root_text: manualRoot,
        raw_indented_text: effectiveText,
        auto_split_root: autoSplit,
        analysis_mode: nonEmptyCount > 25 ? "full" : getSelectedAnalysisMode(),
        junk_mode: junkMode,
        custom_topics: customTopics,
      });
    });

    const fileInputEl = $("mega-json-file");
    const btnClearFile = $("btn-mega-clear-file");
    if (fileInputEl) {
      fileInputEl.addEventListener("change", (ev) => {
        const f = ev.target.files && ev.target.files[0];
        if (!f) {
          state.megaUploadedJson = null;
          state.megaUploadedTxt = null;
          if (btnClearFile) btnClearFile.hidden = true;
          return;
        }
        const reader = new FileReader();
        reader.onload = () => {
          const content = String(reader.result || "");
          if (/\.txt$/i.test(f.name)) {
            state.megaUploadedJson = null;
            state.megaUploadedTxt = content;
            rawLinesEl.value = content;
          } else {
            state.megaUploadedTxt = null;
            state.megaUploadedJson = content;
          }
          if (btnClearFile) btnClearFile.hidden = false;
          announce(`Đã nạp tệp cục bộ: ${f.name}`);
        };
        reader.readAsText(f, "utf-8");
      });
    }
    if (btnClearFile) {
      btnClearFile.addEventListener("click", () => {
        state.megaUploadedJson = null;
        state.megaUploadedTxt = null;
        if (fileInputEl) fileInputEl.value = "";
        btnClearFile.hidden = true;
        announce("Đã bỏ chọn tệp.");
      });
    }

    const btnMegaStress = $("btn-mega-stress");
    if (btnMegaStress) {
      btnMegaStress.addEventListener("click", () => {
        const maxC = getSelectedMegaMaxComments();
        startMegaScanJob({
          mode: "preset_mega_stress",
          stress_n_comments: maxC,
          max_full_comments: maxC,
          scan_strategy: getSelectedMegaStrategy(),
          junk_mode: getSelectedJunkMode(),
          custom_topics: getLiveTopics() || undefined,
        });
      });
    }

    const btnMegaRun = $("btn-mega-run");
    if (btnMegaRun) {
      btnMegaRun.addEventListener("click", () => {
        const maxC = getSelectedMegaMaxComments();
        const strat = getSelectedMegaStrategy();
        const junkMode = getSelectedJunkMode();
        const customTopics = getLiveTopics() || undefined;
        if (state.megaUploadedJson) {
          if (state.prefilledNewsMeta) clearPrefilledNewsMeta();
          startMegaScanJob({
            mode: "json_upload",
            thread_json: state.megaUploadedJson,
            max_full_comments: maxC,
            scan_strategy: strat,
            junk_mode: junkMode,
            custom_topics: customTopics,
          });
          return;
        }
        const urlVal = urlInputEl.value.trim();
        if (urlVal) {
          if (state.prefilledNewsMeta) clearPrefilledNewsMeta();
          const socialPlatform = detectUnsupportedSocialUrl(urlVal);
          if (socialPlatform) {
            showSocialUrlGuide(socialPlatform);
            showError(
              `${socialPlatform} chặn cào link tự động do yêu cầu đăng nhập. Vui lòng copy bài viết + bình luận trên ${socialPlatform} rồi dán vào ô văn bản.`
            );
            return;
          }
          startMegaScanJob({
            mode: "reddit_url",
            reddit_url: urlVal,
            max_full_comments: maxC,
            scan_strategy: strat,
            junk_mode: junkMode,
            custom_topics: customTopics,
          });
          return;
        }
        const rawLines = rawLinesEl.value.trim();
        if (rawLines) {
          const mTitle = titleEl ? titleEl.value.trim() : "";
          const mRoot = $("sb-root-text") ? $("sb-root-text").value.trim() : "";
          startMegaScanJob({
            mode: "custom",
            title: mTitle,
            root_text: mRoot,
            raw_indented_text: rawLines,
            auto_split_root: !mTitle && !mRoot,
            max_full_comments: maxC,
            scan_strategy: strat,
            junk_mode: junkMode,
            custom_topics: customTopics,
          });
          return;
        }
        showError("Vui lòng chọn tệp JSON/JSONL, dán URL Reddit hoặc bấm Kiểm tải Stress.");
      });
    }

    const btnMegaCancel = $("btn-mega-cancel");
    if (btnMegaCancel) {
      btnMegaCancel.addEventListener("click", async () => {
        const targetJobId = state.pendingScanJobId;
        if (!targetJobId) return;
        try {
          await apiPost(`/api/tier1/scan_jobs/${encodeURIComponent(targetJobId)}/cancel`, {});
        } catch (err) {
          showError(err.message);
        }
      });
    }

    const stanceFilterEl = $("mega-stance-filter");
    const searchInputEl = $("mega-search-input");
    const prevPageBtn = $("mega-prev-page");
    const nextPageBtn = $("mega-next-page");
    const reloadComments = (p) => {
      if (state.activeScanJobId || state.tier1Mode === "real") {
        return loadMegaCommentsPage(p).catch((e) => showError(e.message));
      }
      renderClientCommentsPage(p);
      return Promise.resolve();
    };
    if (stanceFilterEl) {
      stanceFilterEl.addEventListener("change", () => reloadComments(1));
    }
    if (searchInputEl) {
      searchInputEl.addEventListener("input", () => {
        if (state.megaSearchDebounceTimer) {
          clearTimeout(state.megaSearchDebounceTimer);
        }
        state.megaSearchDebounceTimer = setTimeout(() => {
          reloadComments(1);
        }, 250);
      });
    }
    if (prevPageBtn) {
      prevPageBtn.addEventListener("click", () => reloadComments(Math.max(1, state.megaPage - 1)));
    }
    if (nextPageBtn) {
      nextPageBtn.addEventListener("click", () => reloadComments(Math.min(state.megaTotalPages, state.megaPage + 1)));
    }

    $("corpus-vn-btn").addEventListener("click", () => renderTier2Corpus("vn"));
    $("corpus-pheme-btn").addEventListener("click", () => renderTier2Corpus("pheme"));

    // Graph zoom, fit, drag-pan & fullscreen modal controls
    const inlineVp = document.querySelector("#panel-tier1 .graph-stage-card .svg-viewport");
    const modalVp = $("modal-svg-viewport");
    wirePanAndWheel(inlineVp, false);
    wirePanAndWheel(modalVp, true);

    if ($("btn-graph-zoom-in")) $("btn-graph-zoom-in").addEventListener("click", () => setGraphZoom(state.graphZoom + 0.2, false));
    if ($("btn-graph-zoom-out")) $("btn-graph-zoom-out").addEventListener("click", () => setGraphZoom(state.graphZoom - 0.2, false));
    if ($("btn-graph-zoom-reset")) $("btn-graph-zoom-reset").addEventListener("click", () => setGraphZoom(1.0, false, true));
    if ($("btn-graph-zoom-fit")) $("btn-graph-zoom-fit").addEventListener("click", () => fitGraphZoom(false));

    if ($("btn-modal-zoom-in")) $("btn-modal-zoom-in").addEventListener("click", () => setGraphZoom(state.modalZoom + 0.2, true));
    if ($("btn-modal-zoom-out")) $("btn-modal-zoom-out").addEventListener("click", () => setGraphZoom(state.modalZoom - 0.2, true));
    if ($("btn-modal-zoom-reset")) $("btn-modal-zoom-reset").addEventListener("click", () => setGraphZoom(1.0, true, true));
    if ($("btn-modal-zoom-fit")) $("btn-modal-zoom-fit").addEventListener("click", () => fitGraphZoom(true));

    const graphDialog = $("graph-fullscreen-dialog");
    if ($("btn-graph-fullscreen") && graphDialog) {
      $("btn-graph-fullscreen").addEventListener("click", () => {
        if (!state.currentPayload) return;
        state.modalZoom = 1.0;
        graphDialog.showModal();
        renderCurrentGraph(true);
        announce("Đã mở cửa sổ phóng to sơ đồ hội thoại.");
      });
    }
    if ($("btn-close-graph-dialog") && graphDialog) {
      $("btn-close-graph-dialog").addEventListener("click", () => graphDialog.close("dismiss"));
    }

    // Toolbar roving-tabindex fallback only when native focusgroup is unsupported
    if (!("focusGroup" in HTMLElement.prototype)) {
      document.querySelectorAll('[role="toolbar"][focusgroup]').forEach((bar) => {
        const items = Array.from(bar.querySelectorAll("button"));
        if (!items.length) return;
        items.forEach((b, idx) => { b.tabIndex = idx === 0 ? 0 : -1; });
        bar.addEventListener("focusin", ({ target }) => {
          if (items.includes(target)) {
            items.forEach((b) => { b.tabIndex = b === target ? 0 : -1; });
          }
        });
        bar.addEventListener("keydown", (ev) => {
          const curIdx = items.indexOf(document.activeElement);
          if (curIdx < 0) return;
          const rtl = getComputedStyle(bar).direction === "rtl";
          let nextIdx = null;
          if (ev.key === "ArrowRight") nextIdx = (curIdx + (rtl ? -1 : 1) + items.length) % items.length;
          else if (ev.key === "ArrowLeft") nextIdx = (curIdx + (rtl ? 1 : -1) + items.length) % items.length;
          else if (ev.key === "Home") nextIdx = 0;
          else if (ev.key === "End") nextIdx = items.length - 1;
          if (nextIdx !== null) {
            ev.preventDefault();
            items[nextIdx].focus();
          }
        });
      });
    }

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

    // Native <dialog> modal with closedby="any" progressive fallback for all dialogs
    const figDialog = $("figure-dialog");
    $("btn-zoom-fig5").addEventListener("click", () => figDialog.showModal());
    $("btn-close-dialog").addEventListener("click", () => figDialog.close("dismiss"));
    document.querySelectorAll("dialog").forEach((dlg) => {
      dlg.addEventListener("close", () => {
        dlg.returnValue = "";
      });
    });
    if (!("closedBy" in HTMLDialogElement.prototype)) {
      const outside = (ev, dlg) => {
        const r = dlg.getBoundingClientRect();
        return ev.clientX < r.left || ev.clientX > r.right || ev.clientY < r.top || ev.clientY > r.bottom;
      };
      document.querySelectorAll('dialog[closedby="any"]').forEach((dlg) => {
        let pressedOut = false;
        dlg.addEventListener("pointerdown", (ev) => {
          pressedOut = outside(ev, dlg);
        });
        dlg.addEventListener("pointerup", (ev) => {
          if (pressedOut && outside(ev, dlg)) dlg.close("backdrop");
          pressedOut = false;
        });
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
        txt.textContent = "AI sẵn sàng";
      } else {
        dot.dataset.ready = "warn";
        txt.textContent = "Chế độ mẫu thử";
      }
      await loadRealThreads();
      switchTier1Mode("real");
    } catch (err) {
      showError(`Không thể kết nối máy chủ cục bộ: ${err.message}`);
      $("health-text").textContent = "Lỗi kết nối";
    }
  }

  boot();
})();
