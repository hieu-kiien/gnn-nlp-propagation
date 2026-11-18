## Summary
- **Jira Ticket:** `[SBIF-xx]` (link to `https://sbifagat-research.atlassian.net/browse/SBIF-xx`)
- **Profile Scope:** `v1` (2D `legacy_nli` MVP) / `v2` (opt-in 4D `qwen_qlora` + 8D stats + beta-NLL) / Docs / Governance

## Five Iron Laws Verification Checklist
- [ ] **No Fake Results:** No synthetic or mock tensors are used in benchmark result tables or `paper/*.tex`.
- [ ] **Zero Data Leakage:** Node/edge inputs strictly use root + earliest $K \le 20$ replies (`min_total_replies=25` for forecasting splits, canonical `topic_splits.json`, `edge_source="teacher"` for LOO).
- [ ] **Verified Citations Only:** Any citation added to `paper/references.bib` has verified authors, venue, year, and DOI/arXiv ID.
- [ ] **Compute & Reproducibility:** Offline `.pt` separation respected; 5-seed protocol `[42, 100, 2024, 777, 999]` preserved.
- [ ] **Test-Gated & Clean Tree:** `python -m pytest -q` passes 100% and zero `.pt`, `.parquet`, `.zip`, `.secret`, or `_machine_key.csv` files are staged.
