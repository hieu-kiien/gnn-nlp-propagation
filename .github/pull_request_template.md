## Summary
- **Jira Ticket:** `[SBIF-xx]` (link to `https://sbifagat-research.atlassian.net/browse/SBIF-xx`)
- **Pipeline Scope:** Unified 2D NLP (`dual_nli` Stance Verbalizer + `SpanPool` + Segment-Isolated CE) / SBi-FAGAT GNN (`hidden_dim=64`, `struct_x` 10D, `cascade_stats` 8D) / Docs / Governance

## Five Iron Laws Verification Checklist
- [ ] **No Fake Results:** No synthetic or mock tensors are used in benchmark result tables or `paper/*.tex`.
- [ ] **Zero Data Leakage:** Node/edge inputs strictly use root + earliest $K \le 20$ replies (`min_total_replies=25` for forecasting splits, canonical `topic_splits.json`, `edge_source="teacher"` for LOO).
- [ ] **Verified Citations Only:** Any citation added to `paper/references.bib` has verified authors, venue, year, and DOI/arXiv ID.
- [ ] **Compute & Reproducibility:** Offline `.pt` separation respected; 5-seed protocol `[42, 100, 2024, 777, 999]` preserved.
- [ ] **Test-Gated & Clean Tree:** `python -m pytest -q` passes 100% and zero `.pt`, `.parquet`, `.zip`, `.secret`, or `_machine_key.csv` files are staged.
