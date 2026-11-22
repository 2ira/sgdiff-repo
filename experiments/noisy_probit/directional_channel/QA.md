# Validation and publication checks

- 800 independent networks, 1,600 paired network-budget records, 8,000 unique fits.
- All 20 numerical tests passed, including analytic-gradient and channel checks.
- Frozen source hashes, regenerated data hashes, and paired nested audit subsets verified.
- Zero failed fits, probability-floor events, or fitted numerical-bound contacts. Full numerical diagnostics are in `validation/integrity.json`.
- The posterior sampler uses the same density as the optimizer. Both main 48,000-draw runs met the frozen diagnostic rule; initial unsuccessful diagnostic attempts are retained. The wider-domain Full sensitivity also passed. These are fixed-dataset checks, not validation coverage estimates.
- Lossless fit and posterior archives were checked against all original local records and draws. Restore them with `archive extract` before re-analysis.
- English slides (12 pages) and report (4 pages) were rendered and visually inspected. Automated checks found no out-of-page words, overfull boxes, or undefined references. Artifact hashes are in `artifact_checks.json`.
- `MANIFEST.sha256` covers the published source, results, archives, and editable artifacts. It excludes itself and ignored intermediate files.

## Interpretation guardrails

At 200 audited dyads under asymmetric truth, Six reduces channel-entry RMSE by 24.7% relative to Five, while structural RMSE improves by about 1.0%. Both prespecified comparisons with Full pass the retention rule, including half the audit budget. Five also passes the half-budget rule in this replication: this study does not establish that the sixth parameter is necessary for audit savings. Prior geometry differs between Five and Six. Fresh seeds establish replication within the specified synthetic mechanisms, not external validity on real networks.
