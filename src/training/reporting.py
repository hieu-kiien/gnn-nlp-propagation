"""Markdown and IEEE/Springer LaTeX table exporters for experiment results."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

# (metric_key, display_name, lower_is_better)
MAIN_METRICS: tuple[tuple[str, str, bool], ...] = (
    ("msle", "MSLE $\\downarrow$", True),
    ("mae", "MAE $\\downarrow$", True),
    ("r2_size", "$R^2_{\\text{size}}$ $\\uparrow$", False),
    ("rmse", "RMSE $\\downarrow$", True),
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
    ("pearson_r", "Pearson $r$ $\\uparrow$", False),
    ("spearman_polar", "Spearman $\\rho$ $\\uparrow$", False),
)

UPFD_METRICS: tuple[tuple[str, str, bool], ...] = (
    ("accuracy", "Accuracy $\\uparrow$", False),
    ("macro_f1", "Macro-F1 $\\uparrow$", False),
)


def significance_marker(p_value: float | None, *, latex: bool = False) -> str:
    """Return significance stars for p < 0.05 (*) or p < 0.01 (**)."""
    if p_value is None or math.isnan(p_value):
        return ""
    if p_value < 0.01:
        return "$^{**}$" if latex else "**"
    if p_value < 0.05:
        return "$^{*}$" if latex else "*"
    return ""


def format_mean_std(
    mean: float,
    std: float,
    *,
    decimals: int = 4,
    latex: bool = False,
    bold: bool = False,
    marker: str = "",
) -> str:
    """Format `mean ± std` cleanly for Markdown or LaTeX tables."""
    if math.isnan(mean):
        return "--"
    pm = " $\\pm$ " if latex else " ± "
    core = f"{mean:.{decimals}f}{pm}{std:.{decimals}f}"
    if bold and latex:
        core = f"\\textbf{{{core}}}"
    elif bold:
        core = f"**{core}**"
    return f"{core}{marker}"


def _best_model_per_metric(
    summary_stats: Mapping[str, Mapping[str, tuple[float, float]]],
    metrics: Sequence[tuple[str, str, bool]],
) -> dict[str, str]:
    best: dict[str, str] = {}
    for key, _, lower_is_better in metrics:
        candidates = [
            (model_name, stats[key][0])
            for model_name, stats in summary_stats.items()
            if key in stats and not math.isnan(stats[key][0])
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


def build_markdown_table(
    benchmark_result: Mapping[str, Any],
    metrics: Sequence[tuple[str, str, bool]] = MAIN_METRICS,
    *,
    decimals: int = 4,
) -> str:
    """Render aggregated 5-seed results and paired t-test p-values as Markdown."""
    summary_stats = benchmark_result["summary_stats"]
    p_values = benchmark_result.get("p_values", {})
    ref_model = benchmark_result.get("reference_model", "SBiFAGAT")
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
        for key, _, _ in metrics:
            mean_v, std_v = stats.get(key, (float("nan"), float("nan")))
            marker = (
                significance_marker(p_values.get(key), latex=False)
                if model_name == ref_model
                else ""
            )
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
        cmp_model = benchmark_result.get("comparator_model", "BiGCN")
        p_cells = [f"p-value ({ref_model} vs. {cmp_model})"]
        for key, _, _ in metrics:
            pv = p_values.get(key, float("nan"))
            p_cells.append("--" if math.isnan(pv) else f"{pv:.4e}")
        lines.append("| " + " | ".join(p_cells) + " |")

    boot_p_values = benchmark_result.get("bootstrap_p_values", {})
    if boot_p_values:
        cmp_model = benchmark_result.get("comparator_model", "BiGCN")
        b_cells = [f"bootstrap p ({ref_model} vs. {cmp_model})"]
        for key, _, _ in metrics:
            bv = boot_p_values.get(key, float("nan"))
            b_cells.append("--" if math.isnan(bv) else f"{bv:.4e}")
        lines.append("| " + " | ".join(b_cells) + " |")

    adj_p_values = benchmark_result.get("adjusted_p_values", {})
    if adj_p_values:
        cmp_model = benchmark_result.get("comparator_model", "BiGCN")
        a_cells = [f"Holm-Bonf adj p ({ref_model} vs. {cmp_model})"]
        for key, _, _ in metrics:
            av = adj_p_values.get(key, float("nan"))
            a_cells.append("--" if math.isnan(av) else f"{av:.4e}")
        lines.append("| " + " | ".join(a_cells) + " |")

    adj_boot_values = benchmark_result.get("adjusted_bootstrap_p_values", {})
    if adj_boot_values:
        cmp_model = benchmark_result.get("comparator_model", "BiGCN")
        ab_cells = [f"Holm-Bonf bootstrap p ({ref_model} vs. {cmp_model})"]
        for key, _, _ in metrics:
            abv = adj_boot_values.get(key, float("nan"))
            ab_cells.append("--" if math.isnan(abv) else f"{abv:.4e}")
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
    """Export IEEE/Springer `booktabs` LaTeX table with bold best & t-test markers."""
    summary_stats = benchmark_result["summary_stats"]
    p_values = benchmark_result.get("p_values", {})
    ref_model = benchmark_result.get("reference_model", "SBiFAGAT")
    cmp_model = benchmark_result.get("comparator_model", "BiGCN")
    best_by_metric = _best_model_per_metric(summary_stats, metrics)

    col_spec = "l" + "c" * len(metrics)
    header_row = " & ".join(["Model"] + [m_label for _, m_label, _ in metrics]) + r" \\"

    body_rows: list[str] = []
    for model_name, stats in summary_stats.items():
        escaped_name = model_name.replace("_", "\\_")
        cells = [escaped_name]
        for key, _, _ in metrics:
            mean_v, std_v = stats.get(key, (float("nan"), float("nan")))
            marker = (
                significance_marker(p_values.get(key), latex=True)
                if model_name == ref_model
                else ""
            )
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
        p_cells = [f"$p$-value ({ref_model} vs. {cmp_model})"]
        for key, _, _ in metrics:
            pv = p_values.get(key, float("nan"))
            p_cells.append("--" if math.isnan(pv) else f"{pv:.2e}")
        p_row_str = "\\midrule\n" + " & ".join(p_cells) + r" \\" + "\n"

    boot_p_values = benchmark_result.get("bootstrap_p_values", {})
    if boot_p_values:
        b_cells = [f"Bootstrap $p$ ({ref_model} vs. {cmp_model})"]
        for key, _, _ in metrics:
            bv = boot_p_values.get(key, float("nan"))
            b_cells.append("--" if math.isnan(bv) else f"{bv:.2e}")
        p_row_str = p_row_str + " & ".join(b_cells) + r" \\" + "\n"

    adj_p_values = benchmark_result.get("adjusted_p_values", {})
    if adj_p_values:
        a_cells = [f"Holm-Bonferroni $p$ ({ref_model} vs. {cmp_model})"]
        for key, _, _ in metrics:
            av = adj_p_values.get(key, float("nan"))
            a_cells.append("--" if math.isnan(av) else f"{av:.2e}")
        p_row_str = p_row_str + " & ".join(a_cells) + r" \\" + "\n"

    adj_boot_values = benchmark_result.get("adjusted_bootstrap_p_values", {})
    if adj_boot_values:
        ab_cells = [f"Holm-Bonf Bootstrap $p$ ({ref_model} vs. {cmp_model})"]
        for key, _, _ in metrics:
            abv = adj_boot_values.get(key, float("nan"))
            ab_cells.append("--" if math.isnan(abv) else f"{abv:.2e}")
        p_row_str = p_row_str + " & ".join(ab_cells) + r" \\" + "\n"

    latex = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\toprule\n"
        f"{header_row}\n"
        "\\midrule\n"
        + "\n".join(body_rows)
        + "\n"
        + p_row_str
        + "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
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
    """Persist seed-level metrics, summary stats, and p-values to JSON cleanly."""
    payload: dict[str, Any] = {
        "seeds": list(benchmark_result.get("seeds", [])),
        "reference_model": benchmark_result.get("reference_model"),
        "comparator_model": benchmark_result.get("comparator_model"),
        "summary_stats": {
            m: {
                k: [None if math.isnan(v[0]) else v[0], None if math.isnan(v[1]) else v[1]]
                for k, v in stats.items()
            }
            for m, stats in benchmark_result["summary_stats"].items()
        },
        "p_values": {
            k: (None if math.isnan(v) else v)
            for k, v in benchmark_result.get("p_values", {}).items()
        },
        "adjusted_p_values": {
            k: (None if math.isnan(v) else v)
            for k, v in benchmark_result.get("adjusted_p_values", {}).items()
        },
        "bootstrap_p_values": {
            k: (None if math.isnan(v) else v)
            for k, v in benchmark_result.get("bootstrap_p_values", {}).items()
        },
        "adjusted_bootstrap_p_values": {
            k: (None if math.isnan(v) else v)
            for k, v in benchmark_result.get("adjusted_bootstrap_p_values", {}).items()
        },
        "raw_results": benchmark_result["raw_results"],
    }
    if extra is not None:
        payload["extra"] = dict(extra)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_file
