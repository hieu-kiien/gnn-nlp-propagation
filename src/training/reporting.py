"""Markdown and IEEE/Springer LaTeX table exporters for experiment results."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

# (metric_key, display_name, lower_is_better)
MAIN_METRICS: tuple[tuple[str, str, bool], ...] = (
    ("msle", "MSLE $\\downarrow$", True),
    ("mae", "MAE $\\downarrow$", True),
    ("r2_size", "$R^2_{\\text{size}}$ $\\uparrow$", False),
    ("rmse", "RMSE $\\downarrow$", True),
    ("r2_polar", "$R^2_{\\text{polar}}$ $\\uparrow$", False),
    ("pearson_r", "Pearson $r$ $\\uparrow$", False),
    ("spearman_polar", "Spearman $\\rho$ $\\uparrow$", False),
    ("macro_f1", "Macro-F1 $\\uparrow$", False),
    ("accuracy", "Acc $\\uparrow$", False),
)

VN_MAIN_METRICS: tuple[tuple[str, str, bool], ...] = (
    ("msle", "MSLE $\\downarrow$", True),
    ("mae", "MAE $\\downarrow$", True),
    ("r2_size", "$R^2_{\\text{size}}$ $\\uparrow$", False),
    ("rmse", "RMSE $\\downarrow$", True),
    ("r2_polar", "$R^2_{\\text{polar}}$ $\\uparrow$", False),
    ("pearson_r", "Pearson $r$ $\\uparrow$", False),
    ("spearman_polar", "Spearman $\\rho$ $\\uparrow$", False),
)

UPFD_METRICS: tuple[tuple[str, str, bool], ...] = (
    ("accuracy", "Accuracy $\\uparrow$", False),
    ("macro_f1", "Macro-F1 $\\uparrow$", False),
)


def _is_finite_num(val: Any) -> bool:
    """Return True iff `val` is a non-boolean real number whose float conversion is finite."""
    if val is None or isinstance(val, bool):
        return False
    try:
        return math.isfinite(float(val))
    except (TypeError, ValueError):
        return False


def significance_marker(
    p_value: float | None,
    *,
    direction: Literal["better", "worse", "tie", "na"] = "better",
    latex: bool = False,
) -> str:
    """Return directional two-sided significance markers: (+)* / (-)* (p < 0.05) or (+)** / (-)** (p < 0.01)."""
    if not _is_finite_num(p_value) or direction in {"tie", "na"}:
        return ""
    pv = float(p_value)
    sign_ch = "-" if direction == "worse" else "+"
    if pv < 0.01:
        return f"$^{{{sign_ch}**}}$" if latex else f"({sign_ch})**"
    if pv < 0.05:
        return f"$^{{{sign_ch}*}}$" if latex else f"({sign_ch})*"
    return ""


def format_mean_std(
    mean: float | None,
    std: float | None,
    *,
    decimals: int = 4,
    latex: bool = False,
    bold: bool = False,
    marker: str = "",
) -> str:
    """Format `mean ± std` cleanly for Markdown or LaTeX tables."""
    if not _is_finite_num(mean):
        return "--"
    mean_val = float(mean)
    pm = " $\\pm$ " if latex else " ± "
    std_str = "--" if not _is_finite_num(std) else f"{float(std):.{decimals}f}"
    core = f"{mean_val:.{decimals}f}{pm}{std_str}"
    if bold and latex:
        core = f"\\textbf{{{core}}}"
    elif bold:
        core = f"**{core}**"
    return f"{core}{marker}"


def _best_model_per_metric(
    summary_stats: Mapping[str, Mapping[str, Sequence[float | None]]],
    metrics: Sequence[tuple[str, str, bool]],
) -> dict[str, str]:
    best: dict[str, str] = {}
    for key, _, lower_is_better in metrics:
        candidates = [
            (model_name, float(stats[key][0]))
            for model_name, stats in summary_stats.items()
            if key in stats
            and stats[key] is not None
            and len(stats[key]) >= 1
            and _is_finite_num(stats[key][0])
        ]
        if not candidates:
            continue
        winner = (
            min(candidates, key=lambda item: item[1])[0]
            if lower_is_better
            else max(candidates, key=lambda item: item[1])[0]
        )
        best[key] = winner
    return best


def _resolve_marker_pvalue(benchmark_result: Mapping[str, Any], key: str) -> float | None:
    """Prefer Holm-Bonferroni adjusted sample-level bootstrap p-value, falling back to paired t-test."""
    for map_key in ("adjusted_bootstrap_p_values", "adjusted_p_values", "p_values"):
        pmap = benchmark_result.get(map_key) or {}
        if key in pmap:
            pv = pmap[key]
            if _is_finite_num(pv):
                return float(pv)
    return None


def _ref_vs_cmp_direction(
    summary_stats: Mapping[str, Mapping[str, Sequence[float | None]]],
    ref_model: str,
    cmp_model: str,
    key: str,
    lower_is_better: bool,
) -> Literal["better", "worse", "tie", "na"]:
    """Return directional comparison of `ref_model` against `cmp_model`."""
    ref_stat = summary_stats.get(ref_model, {}).get(key)
    cmp_stat = summary_stats.get(cmp_model, {}).get(key)
    if ref_stat is None or cmp_stat is None or len(ref_stat) < 1 or len(cmp_stat) < 1:
        return "na"
    if not (_is_finite_num(ref_stat[0]) and _is_finite_num(cmp_stat[0])):
        return "na"
    ref_val, cmp_val = float(ref_stat[0]), float(cmp_stat[0])
    if math.isclose(ref_val, cmp_val, rel_tol=1e-9, abs_tol=1e-12):
        return "tie"
    is_better = (ref_val < cmp_val) if lower_is_better else (ref_val > cmp_val)
    return "better" if is_better else "worse"


def _sanitize_json_value(obj: Any) -> Any:
    """Recursively replace non-finite floats (NaN/Inf) with None for strict RFC 8259 JSON."""
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return _sanitize_json_value(obj.item())
        except Exception:
            pass
    if isinstance(obj, Mapping):
        return {str(k): _sanitize_json_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json_value(v) for v in obj]
    return obj


def build_markdown_table(
    benchmark_result: Mapping[str, Any],
    metrics: Sequence[tuple[str, str, bool]] = MAIN_METRICS,
    *,
    decimals: int = 4,
) -> str:
    """Render aggregated 5-seed results and paired t-test p-values as Markdown."""
    summary_stats = benchmark_result["summary_stats"]
    p_values = benchmark_result.get("p_values") or {}
    ref_model = benchmark_result.get("reference_model", "SBiFAGAT")
    cmp_model = benchmark_result.get("comparator_model", "BiGCN")
    best_by_metric = _best_model_per_metric(summary_stats, metrics)

    headers = ["Model"] + [
        label.replace("$\\downarrow$", "(↓)")
        .replace("$\\uparrow$", "(↑)")
        .replace("$r$", "r")
        for _, label, _ in metrics
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join([":---"] + [":---:"] * len(metrics)) + " |",
    ]

    for model_name, stats in summary_stats.items():
        cells = [model_name]
        for key, _, lower_is_better in metrics:
            pair = stats.get(key)
            mean_v, std_v = (pair[0], pair[1]) if pair is not None and len(pair) >= 2 else (None, None)
            if model_name == ref_model:
                direction = _ref_vs_cmp_direction(
                    summary_stats, ref_model, cmp_model, key, lower_is_better
                )
                marker = significance_marker(
                    _resolve_marker_pvalue(benchmark_result, key),
                    direction=direction,
                    latex=False,
                )
            else:
                marker = ""
            cells.append(
                format_mean_std(
                    mean_v,
                    std_v,
                    decimals=decimals,
                    latex=False,
                    bold=(best_by_metric.get(key) == model_name),
                    marker=marker,
                )
            )
        lines.append("| " + " | ".join(cells) + " |")

    if p_values:
        p_cells = [f"p-value ({ref_model} vs. {cmp_model})"]
        for key, _, _ in metrics:
            pv = p_values.get(key)
            p_cells.append("--" if not _is_finite_num(pv) else f"{float(pv):.4e}")
        lines.append("| " + " | ".join(p_cells) + " |")

    np_label = str(benchmark_result.get("nonparametric_test_label") or "Bootstrap")
    boot_p_values = benchmark_result.get("bootstrap_p_values") or {}
    if boot_p_values:
        b_cells = [f"{np_label.lower()} p ({ref_model} vs. {cmp_model})"]
        for key, _, _ in metrics:
            bv = boot_p_values.get(key)
            b_cells.append("--" if not _is_finite_num(bv) else f"{float(bv):.4e}")
        lines.append("| " + " | ".join(b_cells) + " |")

    adj_p_values = benchmark_result.get("adjusted_p_values") or {}
    if adj_p_values:
        a_cells = [f"Holm-Bonf adj p ({ref_model} vs. {cmp_model})"]
        for key, _, _ in metrics:
            av = adj_p_values.get(key)
            a_cells.append("--" if not _is_finite_num(av) else f"{float(av):.4e}")
        lines.append("| " + " | ".join(a_cells) + " |")

    adj_boot_values = benchmark_result.get("adjusted_bootstrap_p_values") or {}
    if adj_boot_values:
        ab_cells = [f"Holm-Bonf {np_label.lower()} p ({ref_model} vs. {cmp_model})"]
        for key, _, _ in metrics:
            abv = adj_boot_values.get(key)
            ab_cells.append("--" if not _is_finite_num(abv) else f"{float(abv):.4e}")
        lines.append("| " + " | ".join(ab_cells) + " |")

    return "\n".join(lines)


def export_latex_table(
    benchmark_result: Mapping[str, Any],
    output_path: str | Path,
    *,
    caption: str,
    label: str,
    metrics: Sequence[tuple[str, str, bool]] = MAIN_METRICS,
    decimals: int = 4,
) -> Path:
    """Export IEEE/Springer `booktabs` LaTeX table with bold best & direction-aware significance markers."""
    summary_stats = benchmark_result["summary_stats"]
    p_values = benchmark_result.get("p_values") or {}
    ref_model = benchmark_result.get("reference_model", "SBiFAGAT")
    cmp_model = benchmark_result.get("comparator_model", "BiGCN")
    ref_esc = str(ref_model).replace("_", "\\_").replace("&", "\\&")
    cmp_esc = str(cmp_model).replace("_", "\\_").replace("&", "\\&")
    best_by_metric = _best_model_per_metric(summary_stats, metrics)

    col_spec = "l" + "c" * len(metrics)
    header_row = " & ".join(["Model"] + [m_label for _, m_label, _ in metrics]) + r" \\"

    body_rows: list[str] = []
    for model_name, stats in summary_stats.items():
        escaped_name = model_name.replace("_", "\\_").replace("&", "\\&")
        cells = [escaped_name]
        for key, _, lower_is_better in metrics:
            pair = stats.get(key)
            mean_v, std_v = (pair[0], pair[1]) if pair is not None and len(pair) >= 2 else (None, None)
            if model_name == ref_model:
                direction = _ref_vs_cmp_direction(
                    summary_stats, ref_model, cmp_model, key, lower_is_better
                )
                marker = significance_marker(
                    _resolve_marker_pvalue(benchmark_result, key),
                    direction=direction,
                    latex=True,
                )
            else:
                marker = ""
            cells.append(
                format_mean_std(
                    mean_v,
                    std_v,
                    decimals=decimals,
                    latex=True,
                    bold=(best_by_metric.get(key) == model_name),
                    marker=marker,
                )
            )
        body_rows.append(" & ".join(cells) + r" \\")

    p_row_str = ""
    if p_values:
        p_cells = [f"$p$-value ({ref_esc} vs. {cmp_esc})"]
        for key, _, _ in metrics:
            pv = p_values.get(key)
            p_cells.append("--" if not _is_finite_num(pv) else f"{float(pv):.2e}")
        p_row_str = "\\midrule\n" + " & ".join(p_cells) + r" \\" + "\n"

    np_label = str(benchmark_result.get("nonparametric_test_label") or "Bootstrap").replace("_", "\\_")
    boot_p_values = benchmark_result.get("bootstrap_p_values") or {}
    if boot_p_values:
        b_cells = [f"{np_label} $p$ ({ref_esc} vs. {cmp_esc})"]
        for key, _, _ in metrics:
            bv = boot_p_values.get(key)
            b_cells.append("--" if not _is_finite_num(bv) else f"{float(bv):.2e}")
        p_row_str = p_row_str + " & ".join(b_cells) + r" \\" + "\n"

    adj_p_values = benchmark_result.get("adjusted_p_values") or {}
    if adj_p_values:
        a_cells = [f"Holm-Bonferroni $p$ ({ref_esc} vs. {cmp_esc})"]
        for key, _, _ in metrics:
            av = adj_p_values.get(key)
            a_cells.append("--" if not _is_finite_num(av) else f"{float(av):.2e}")
        p_row_str = p_row_str + " & ".join(a_cells) + r" \\" + "\n"

    adj_boot_values = benchmark_result.get("adjusted_bootstrap_p_values") or {}
    if adj_boot_values:
        ab_cells = [f"Holm-Bonf {np_label} $p$ ({ref_esc} vs. {cmp_esc})"]
        for key, _, _ in metrics:
            abv = adj_boot_values.get(key)
            ab_cells.append("--" if not _is_finite_num(abv) else f"{float(abv):.2e}")
        p_row_str = p_row_str + " & ".join(ab_cells) + r" \\" + "\n"

    is_wide = len(metrics) >= 5
    env = "table*" if is_wide else "table"
    box_width = "\\textwidth" if is_wide else "\\columnwidth"

    latex = (
        f"\\begin{{{env}}}[t]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\resizebox{{{box_width}}}{{!}}{{\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\toprule\n"
        f"{header_row}\n"
        "\\midrule\n"
        + "\n".join(body_rows)
        + "\n"
        + p_row_str
        + "\\bottomrule\n"
        "\\end{tabular}\n"
        "}\n"
        f"\\end{{{env}}}\n"
    )

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(latex, encoding="utf-8")
    return out_file


def serialize_benchmark_json(
    benchmark_result: Mapping[str, Any],
    output_path: str | Path,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Persist seed-level metrics, summary stats, and p-values to strict RFC 8259 JSON."""
    payload: dict[str, Any] = {
        "seeds": list(benchmark_result.get("seeds", [])),
        "reference_model": benchmark_result.get("reference_model"),
        "comparator_model": benchmark_result.get("comparator_model"),
        "summary_stats": {
            m: {
                k: [
                    float(v[0]) if _is_finite_num(v[0]) else None,
                    float(v[1]) if _is_finite_num(v[1]) else None,
                ]
                for k, v in stats.items()
                if v is not None and len(v) >= 2
            }
            for m, stats in benchmark_result["summary_stats"].items()
        },
        "p_values": {
            k: (float(v) if _is_finite_num(v) else None)
            for k, v in (benchmark_result.get("p_values") or {}).items()
        },
        "adjusted_p_values": {
            k: (float(v) if _is_finite_num(v) else None)
            for k, v in (benchmark_result.get("adjusted_p_values") or {}).items()
        },
        "bootstrap_p_values": {
            k: (float(v) if _is_finite_num(v) else None)
            for k, v in (benchmark_result.get("bootstrap_p_values") or {}).items()
        },
        "adjusted_bootstrap_p_values": {
            k: (float(v) if _is_finite_num(v) else None)
            for k, v in (benchmark_result.get("adjusted_bootstrap_p_values") or {}).items()
        },
        "nonparametric_test_label": benchmark_result.get("nonparametric_test_label"),
        "wilcoxon_p_values": {
            k: (float(v) if _is_finite_num(v) else None)
            for k, v in (benchmark_result.get("wilcoxon_p_values") or {}).items()
        },
        "adjusted_wilcoxon_p_values": {
            k: (float(v) if _is_finite_num(v) else None)
            for k, v in (benchmark_result.get("adjusted_wilcoxon_p_values") or {}).items()
        },
        "cluster_paired_p_values": benchmark_result.get("cluster_paired_p_values"),
        "raw_results": benchmark_result.get("raw_results", {}),
    }
    if extra is not None:
        payload["extra"] = dict(extra)

    clean_payload = _sanitize_json_value(payload)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(clean_payload, indent=2, allow_nan=False), encoding="utf-8")
    return out_file
