# Capstone Research Report — Refresh / Content Opportunity Scoring

- **Author:** Cyb3rVigil
- **Lane:** Refresh / Content Opportunity Scoring
- **Repository:** https://github.com/Cyb3rVigil/flyrank-ml-internship
- **Paper:** https://cyb3rvigil.github.io/flyrank-ml-internship/
- **Data snapshot:** FlyRank Internship Warehouse v20260703
- **Status:** Research/decision-support only; automatic deployment blocked

## 1. Problem framing

**Research question:** Can five historical search-performance signals reliably rank client-content items that will experience a substantial decline in future daily impressions, particularly for clients not seen during training?

The operational decision is a fixed Top-20 queue for human investigation. The row grain is one pseudonymous client × content-item pair after aggregation across two disjoint calendar windows. A false positive consumes limited reviewer time; a missed positive may delay investigation. This is a decision-support ranking problem, not an automatic refresh or content-generation task.

## 2. Data safety and contract

- **Release:** FlyRank Internship Warehouse v20260703 (frozen snapshot; export date 3 July 2026).
- **Fact table:** `fact_content_daily_performance`, March 2026 partition only.
- **Feature window:** 1–21 March 2026.
- **Outcome window:** 22–31 March 2026.
- **Eligibility rules:** at least 14 historical days, at least 7 outcome days, at least 50 historical impressions, and `ga4_data_available IS TRUE`.
- **Eligible modeling frame:** 2,348 client-content rows across 14 pseudonymous clients; overall proxy-positive rate 23.55%; duplicate client-content pairs: 0.
- **Held-out partition:** 2,120 training rows / 10 clients; 228 test rows / 4 completely unseen clients; train/test client overlap 0; held-out proxy-positive rate 5.70% (13 positives).
- **Deliberately excluded:** June 2026 final partition and the `_sample` table, to avoid using the latest period as an accidental answer key; query-level table and dimensions, because they were not required for the selected five-feature experiment.
- **Public-safe restrictions:** no client names, URLs, domains, titles, private queries, raw exports, credentials, or identity mappings. Pseudonymous IDs are used only for grouping and deterministic tie-breaking.

The frozen release and table grain are described in the [FlyRank Internship Warehouse dataset card](https://huggingface.co/datasets/FlyRank/internship-warehouse). Data use is governed by the repository's [DATA_USE.md](../DATA_USE.md).

## 3. Baseline

The transparent rule weights four train-thresholded indicators: high historical impressions/day (4 points), average position from 4 to 20 (3 points), low CTR (2 points), and high position variability (1 point). Thresholds are computed only from training clients. The rule is a fair baseline because it is readable, uses the same five historical measurements as the models, and is evaluated on exactly the same grouped holdout rows.

## 4. Model and label

The label is positive if outcome-window average impressions/day are below 80% of historical-window average impressions/day. It is a decline proxy; it does not encode a cause or whether a refresh would be effective.

Five pre-decision features:
1. Historical impressions/day.
2. Historical clicks/day.
3. Historical CTR.
4. Historical impression-weighted average position.
5. Historical average-position standard deviation.

Compared approaches:
- **Transparent baseline:** training-derived rule thresholds and deterministic rule score.
- **Logistic Regression:** median imputation, standard scaling, class-balanced model, maximum 2,000 iterations, random seed 42.
- **Constrained Random Forest:** 400 trees, maximum depth 6, minimum leaf size 10, square-root feature selection, balanced subsampling, random seed 42.

No outcome-window variable, target copy, `trend_pct`, `trend_direction`, product decision score, client ID, or content ID is used as a predictive feature. Model preprocessing is fitted on the training partition only.

## 5. Evaluation design and results

The primary metric is Precision@20, matching the analyst's assumed review capacity. Precision@50, average precision, and ROC-AUC provide additional ranking context. All methods use the exact same client-grouped holdout.

| Method | Test base rate | Precision@20 | Precision@50 | Average precision | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Transparent rule baseline | 5.70% | 0.000 | 0.040 | 0.0478 | 0.3955 |
| Logistic Regression | 5.70% | 0.000 | 0.040 | 0.0461 | 0.3284 |
| Constrained Random Forest | 5.70% | 0.000 | 0.020 | 0.0591 | 0.5120 |

All models found zero proxy-positive examples in the top 20. The Random Forest's average precision (0.0591) is only marginally above the 0.057 test prevalence and its ROC-AUC (0.5120) is barely above the 0.5 chance reference. This does not make the model useful for the actual Top-20 decision.

A separate random-row diagnostic had Precision@20 = 0.700 but allowed 12 client groups to overlap between train and test. It is intentionally treated as a leakage-risk illustration, not a valid measure of unseen-client performance and not a like-for-like comparison to the client-grouped test.

## 6. Interpretation and limitations

Observed result: the five selected historical features and these models did not provide useful Top-20 ranking on this single unseen-client split.

Important limits:
- The holdout has only four unseen clients and 228 eligible observations.
- The held-out prevalence (5.70%) is substantially lower than the eligible-frame prevalence (23.55%); performance is population-sensitive.
- The ten-day future window may be affected by seasonality, demand changes, reporting variation, and short-term volatility.
- The target is an impression-decline proxy—not content quality, business impact, or causal response to an edit.
- One seeded group split is reproducible but does not provide confidence intervals or stable population-wide generalization.
- The model uses only five historical features and filters to pages with enough exposure and available GA4 data.

The analysis does not prove any Google ranking algorithm factor. It does not prove that refreshing a page causes traffic recovery. It does not support production automation.

## 7. Ranked recommendations

1. **Block automated content actions.** Do not automatically refresh, rewrite, delete, redirect, canonicalize, consolidate, publish, or make client-facing promises from this model.
2. **Use a transparent human-review queue.** Prioritize explainable signals and attach readable reason codes; treat scores as triage evidence only.
3. **Verify the cause before action.** Check indexability, canonical/redirect state, tracking continuity, query composition, intent match, cannibalization, seasonality, technical issues, page freshness, and actual user/business value.
4. **Collect more validation evidence.** Add more held-out clients, repeated group validation, and out-of-time tests on later windows. Include uncertainty intervals and base rates beside top-K metrics.
5. **Measure intervention outcomes separately.** Where permitted, compare approved changes with suitable untreated pages using predeclared follow-up windows; do not infer causality from observational before/after metrics.
6. **Extend features only with provenance and leakage checks.** Potential candidates include stable content categories, age/freshness, technical availability, and longer-period trends when known at the decision moment.

## 8. Reproducibility

The execution record and reusable code are in the notebooks below. Run in Google Colab or a compatible Python environment. For gated warehouse access, accept the Hugging Face dataset terms and store a read-only token in Colab Secrets as `HF_TOKEN`. Do not place a token in notebook source or commit it.

- [Capstone notebook](https://colab.research.google.com/github/Cyb3rVigil/flyrank-ml-internship/blob/main/work/notebooks/capstone.ipynb)
- [ML-08 model notebook](https://colab.research.google.com/github/Cyb3rVigil/flyrank-ml-internship/blob/main/work/notebooks/w05_model.ipynb)
- [ML-09 validation audit](https://colab.research.google.com/github/Cyb3rVigil/flyrank-ml-internship/blob/main/work/notebooks/w06_validation_audit.ipynb)
- [ML-10 action playbook](https://colab.research.google.com/github/Cyb3rVigil/flyrank-ml-internship/blob/main/work/notebooks/w07_action_playbook.ipynb)
- [Full GitHub repository](https://github.com/Cyb3rVigil/flyrank-ml-internship)

Random seed: 42. Reported numerical values are the saved outputs in the committed model and validation notebooks. Re-run the notebooks after any change to data version, SQL, eligibility filters, features, or library versions, then update the paper and report to match the new measured result. The current result should not be described as freshly rerun in this editing session.

## 9. Acknowledgments and data credit

Built on the FlyRank ML Internship dataset. Thanks to FlyRank for providing the educational, anonymized warehouse and the methodology guidance used in this research.

- [FlyRank AI](https://flyrank.ai)
- [FlyRank Internship Warehouse dataset card](https://huggingface.co/datasets/FlyRank/internship-warehouse)

---

**Claim standard:** observed · measured · directional · decision-support. Negative results are reported as measured, not hidden.
