"""Merge parallel Kaggle paper outputs into unified IEEEtran LaTeX tables and validate syntax.

Performs strictly deterministic post-processing without fabricating any numbers:
  1. Merges `paper/results_dataset_eda_{pheme,vn}.json` into unified `paper/tables_dataset_eda.tex`
     (`\\label{tab:dataset_eda}`).
  2. Builds unified `paper/tables_efficiency.tex` (`\\label{tab:efficiency}`) from the available
     `paper/results_efficiency_{pheme,vn}.json`.
  3. Audits every `paper/tables_*.tex` file for unescaped underscores `_` outside math mode
     and LaTeX label/ref/cite/includegraphics commands to prevent `pdflatex` crashes.
  4. Writes an inventory manifest (`paper/assembly_manifest.json`) listing present vs. missing
     tables and vector figures across both PHEME and VN tracks.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

_MATH_ENVIRONMENTS: frozenset[str] = frozenset(
    {
        "multline",
        "multline*",
        "equation",
        "equation*",
        "align",
        "align*",
        "gather",
        "gather*",
        "eqnarray",
        "eqnarray*",
        "flalign",
        "flalign*",
        "alignat",
        "alignat*",
    }
)
_SKIP_ARG_MACROS: frozenset[str] = frozenset(
    {
        "label",
        "ref",
        "eqref",
        "pageref",
        "cite",
        "citet",
        "citep",
        "input",
        "include",
        "bibliography",
        "bibliographystyle",
        "includegraphics",
        "url",
    }
)
_FIRST_ARG_ONLY_MACROS: frozenset[str] = frozenset({"IfFileExists", "href"})
_TEXT_IN_MATH_MACROS: frozenset[str] = frozenset(
    {"text", "mbox", "textrm", "texttt", "textit", "textbf"}
)


def _skip_ws(text: str, pos: int) -> int:
    n = len(text)
    while pos < n and text[pos] in " \t":
        pos += 1
    return pos


def _skip_balanced(text: str, pos: int, open_ch: str, close_ch: str) -> int:
    n = len(text)
    if pos >= n or text[pos] != open_ch:
        return pos
    depth = 1
    pos += 1
    while pos < n and depth > 0:
        ch = text[pos]
        if ch == "\\" and pos + 1 < n:
            pos += 2
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
        pos += 1
    return pos


def find_unescaped_underscores_in_latex(tex_content: str) -> list[tuple[int, str]]:
    """Return `(line_number, line_text)` for lines containing unescaped `_` outside math/macros."""
    raw_lines = tex_content.splitlines()
    violation_lines: set[int] = set()

    n = len(tex_content)
    pos = 0
    line_no = 1
    brace_depth = 0
    mode_stack: list[tuple[str, Any]] = []  # ("math", delim) or ("text", entry_brace_depth)

    def _in_math() -> bool:
        return bool(mode_stack and mode_stack[-1][0] == "math")

    def _pop_math(delim: str) -> None:
        for idx in range(len(mode_stack) - 1, -1, -1):
            if mode_stack[idx] == ("math", delim):
                del mode_stack[idx:]
                return

    def _advance_pos(start: int, end: int) -> int:
        nonlocal line_no
        line_no += tex_content.count("\n", start, end)
        return end

    while pos < n:
        ch = tex_content[pos]
        if ch == "\n":
            line_no += 1
            pos += 1
            continue

        if ch == "%":
            nl = tex_content.find("\n", pos)
            if nl == -1:
                break
            line_no += 1
            pos = nl + 1
            continue

        if ch == "\\":
            if pos + 1 >= n:
                pos += 1
                continue
            nxt = tex_content[pos + 1]
            if not nxt.isalpha():
                if nxt == "(":
                    if not _in_math():
                        mode_stack.append(("math", "paren"))
                elif nxt == ")":
                    _pop_math("paren")
                elif nxt == "[":
                    if not _in_math():
                        mode_stack.append(("math", "bracket"))
                elif nxt == "]":
                    _pop_math("bracket")
                pos += 2
                continue

            cmd_end = pos + 1
            while cmd_end < n and tex_content[cmd_end].isalpha():
                cmd_end += 1
            cmd = tex_content[pos + 1 : cmd_end]
            pos = cmd_end

            if cmd == "verb" and pos < n and tex_content[pos] != "\n":
                delim = tex_content[pos]
                end_v = tex_content.find(delim, pos + 1)
                eol = tex_content.find("\n", pos + 1)
                if end_v != -1 and (eol == -1 or end_v < eol):
                    pos = end_v + 1
                continue

            if cmd in {"begin", "end"}:
                probe = _skip_ws(tex_content, pos)
                if probe < n and tex_content[probe] == "{":
                    end_env = _skip_balanced(tex_content, probe, "{", "}")
                    env_name = tex_content[probe + 1 : end_env - 1].strip()
                    pos = _advance_pos(pos, end_env)
                    if env_name in _MATH_ENVIRONMENTS:
                        tag = f"env:{env_name}"
                        if cmd == "begin":
                            mode_stack.append(("math", tag))
                        else:
                            _pop_math(tag)
                continue

            if cmd in _SKIP_ARG_MACROS:
                probe = _skip_ws(tex_content, pos)
                while probe < n and tex_content[probe] == "[":
                    probe = _skip_balanced(tex_content, probe, "[", "]")
                    probe = _skip_ws(tex_content, probe)
                if probe < n and tex_content[probe] == "{":
                    probe = _skip_balanced(tex_content, probe, "{", "}")
                pos = _advance_pos(pos, probe)
                continue

            if cmd in _FIRST_ARG_ONLY_MACROS:
                probe = _skip_ws(tex_content, pos)
                if probe < n and tex_content[probe] == "{":
                    probe = _skip_balanced(tex_content, probe, "{", "}")
                pos = _advance_pos(pos, probe)
                continue

            if cmd in _TEXT_IN_MATH_MACROS and _in_math():
                probe = _skip_ws(tex_content, pos)
                if probe < n and tex_content[probe] == "{":
                    pos = _advance_pos(pos, probe + 1)
                    brace_depth += 1
                    mode_stack.append(("text", brace_depth))
                continue

            continue

        if ch == "$":
            if pos + 1 < n and tex_content[pos + 1] == "$":
                if mode_stack and mode_stack[-1] == ("math", "$$"):
                    mode_stack.pop()
                else:
                    mode_stack.append(("math", "$$"))
                pos += 2
            else:
                if mode_stack and mode_stack[-1] == ("math", "$"):
                    mode_stack.pop()
                elif not _in_math():
                    mode_stack.append(("math", "$"))
                pos += 1
            continue

        if ch == "{":
            brace_depth += 1
            pos += 1
            continue

        if ch == "}":
            if mode_stack and mode_stack[-1] == ("text", brace_depth):
                mode_stack.pop()
            brace_depth = max(0, brace_depth - 1)
            pos += 1
            continue

        if ch == "_" and not _in_math():
            violation_lines.add(line_no)
        pos += 1

    return [
        (ln, raw_lines[ln - 1].strip())
        for ln in sorted(violation_lines)
        if 1 <= ln <= len(raw_lines)
    ]


def assemble_paper_artifacts(
    paper_dir: Path = Path("results"),
    *,
    strict_latex_check: bool = True,
) -> dict[str, Any]:
    """Merge multi-track JSON summaries into unified LaTeX tables and audit table syntax."""
    from scripts.benchmark_efficiency import format_efficiency_latex
    from scripts.compute_kappa import format_human_audit_latex
    from scripts.export_dataset_stats import format_dataset_stats_latex
    from src.data.user_network import format_macro_network_latex

    pdir = Path(paper_dir)
    pdir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    # 1. Merge PHEME + VN EDA JSON summaries into unified Table 1 (`tables_dataset_eda.tex`)
    eda_datasets: list[dict[str, Any]] = []
    for ds_tag in ("pheme", "vn"):
        eda_json = pdir / f"results_dataset_eda_{ds_tag}.json"
        if eda_json.is_file():
            blob = json.loads(eda_json.read_text(encoding="utf-8"))
            entries = blob.get("datasets") if isinstance(blob, dict) else blob
            if isinstance(entries, list):
                eda_datasets.extend(entries)
    if eda_datasets:
        eda_tex = pdir / "tables_dataset_eda.tex"
        eda_tex.write_text(
            format_dataset_stats_latex(eda_datasets, label="tab:dataset_eda"),
            encoding="utf-8",
        )
        generated.append(eda_tex.as_posix())

    # 1b. Merge PHEME + VN Macro Signed Social Network JSON summaries into `tables_macro_network.tex`
    macro_datasets: list[dict[str, Any]] = []
    for ds_tag in ("pheme", "vn"):
        macro_json = pdir / f"results_macro_network_{ds_tag}.json"
        if macro_json.is_file():
            blob = json.loads(macro_json.read_text(encoding="utf-8"))
            if isinstance(blob, dict):
                macro_datasets.append(blob)
    if macro_datasets:
        macro_tex = pdir / "tables_macro_network.tex"
        macro_tex.write_text(
            format_macro_network_latex(macro_datasets, label="tab:macro_network"),
            encoding="utf-8",
        )
        generated.append(macro_tex.as_posix())

    # 2. Ensure unified `tables_efficiency.tex` exists, preferring v3 ReZero outputs (>=137086 SBiFAGAT params)
    eff_tex = pdir / "tables_efficiency.tex"
    eff_candidates: list[tuple[int, list[dict[str, Any]]]] = []
    for ds_tag in ("vn", "pheme"):
        eff_json = pdir / f"results_efficiency_{ds_tag}.json"
        if eff_json.is_file():
            blob = json.loads(eff_json.read_text(encoding="utf-8"))
            rows = blob.get("models") if isinstance(blob, dict) else blob
            if isinstance(rows, list) and rows:
                sbif_params = next((int(r.get("params", 0)) for r in rows if r.get("model") == "SBiFAGAT"), 0)
                is_v3 = 1 if sbif_params >= 137086 else 0
                eff_candidates.append((is_v3, rows))
    if eff_candidates:
        eff_candidates.sort(key=lambda item: item[0], reverse=True)
        eff_tex.write_text(
            format_efficiency_latex(eff_candidates[0][1], label="tab:efficiency"),
            encoding="utf-8",
        )
        generated.append(eff_tex.as_posix())

    # 2b. Merge pair-level & thread-level human audit JSON reports without overwriting VI vs EN
    corpus_candidates = (
        ("results_human_audit_pairs.json", ""),
        ("results_human_audit_vi.json", "VN"),
        ("results_human_audit_en.json", "EN"),
        ("results_rumoureval_sdqc.json", ""),
        ("results_human_audit_vn_30_trees.json", "VN"),
        ("results_human_audit_vn_50_trees.json", "VN"),
    )
    corpus_reports: list[dict[str, Any]] = []
    seen_tree = False
    for candidate_name, corpus_prefix in corpus_candidates:
        if "trees" in candidate_name:
            if seen_tree:
                continue
        cand_path = pdir / candidate_name
        if not cand_path.is_file():
            continue
        blob = json.loads(cand_path.read_text(encoding="utf-8"))
        if not isinstance(blob, dict):
            continue
        if "trees" in candidate_name:
            seen_tree = True
        corpus_reports.append({"prefix": corpus_prefix, "report": blob})
    if corpus_reports:
        audit_tex = pdir / "tables_human_audit.tex"
        audit_tex.write_text(
            format_human_audit_latex({"corpora": corpus_reports}, label="tab:human_audit"),
            encoding="utf-8",
        )
        generated.append(audit_tex.as_posix())

    # 3. Validate LaTeX tables for unescaped underscores outside math mode
    latex_issues: dict[str, list[dict[str, Any]]] = {}
    for tex_file in sorted(pdir.glob("tables_*.tex")):
        content = tex_file.read_text(encoding="utf-8")
        bad = find_unescaped_underscores_in_latex(content)
        if bad:
            latex_issues[tex_file.name] = [
                {"line": ln, "text": txt} for ln, txt in bad
            ]

    expected_tables = [
        "tables_dataset_eda.tex",
        "tables_main_pheme.tex",
        "tables_loo_pheme.tex",
        "tables_ablation_pheme.tex",
        "tables_mechanisms_pheme.tex",
        "tables_main_vn.tex",
        "tables_temporal_vn.tex",
        "tables_ablation_vn.tex",
        "tables_mechanisms_vn.tex",
        "tables_upfd.tex",
        "tables_efficiency.tex",
    ]
    expected_figures = [
        "figures/pheme/fig1_umap_polarization.pdf",
        "figures/pheme/fig2_early_k_curve.pdf",
        "figures/pheme/fig3_cascade_heatmap.pdf",
        "figures/pheme/fig4_counterfactual_interventions.pdf",
        "figures/vn/fig1_umap_polarization.pdf",
        "figures/vn/fig2_early_k_curve.pdf",
        "figures/vn/fig3_cascade_heatmap.pdf",
        "figures/vn/fig4_counterfactual_interventions.pdf",
    ]
    optional_tables = [
        "tables_upfd_politifact.tex",
        "tables_human_audit.tex",
        "tables_macro_network.tex",
        "tables_signed_link_vn.tex",
        "tables_signed_link_pheme.tex",
    ]
    optional_figures = [
        "figures/vn/fig5_macro_signed_network.pdf",
        "figures/pheme/fig5_macro_signed_network.pdf",
    ]
    manifest: dict[str, Any] = {
        "paper_dir": pdir.as_posix(),
        "generated_tables": generated,
        "present_tables": [t for t in expected_tables if (pdir / t).is_file()],
        "missing_tables": [t for t in expected_tables if not (pdir / t).is_file()],
        "optional_tables_present": [t for t in optional_tables if (pdir / t).is_file()],
        "optional_tables_missing": [t for t in optional_tables if not (pdir / t).is_file()],
        "present_figures": [f for f in expected_figures if (pdir / f).is_file()],
        "missing_figures": [f for f in expected_figures if not (pdir / f).is_file()],
        "optional_figures_present": [f for f in optional_figures if (pdir / f).is_file()],
        "optional_figures_missing": [f for f in optional_figures if not (pdir / f).is_file()],
        "latex_underscore_issues": latex_issues,
    }
    (pdir / "assembly_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if strict_latex_check and latex_issues:
        raise ValueError(
            f"Detected unescaped '_' in LaTeX tables (pdflatex will fail): {json.dumps(latex_issues, indent=2)}"
        )
    return manifest


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="Assemble unified paper LaTeX tables from PHEME/VN outputs and validate LaTeX syntax."
    )
    parser.add_argument("--paper-dir", type=Path, default=Path("results"))
    parser.add_argument("--no-strict", action="store_true")
    args = parser.parse_args()

    manifest = assemble_paper_artifacts(
        args.paper_dir,
        strict_latex_check=not bool(args.no_strict),
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
