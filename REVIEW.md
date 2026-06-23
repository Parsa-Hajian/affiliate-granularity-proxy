# Internal Self-Review

This manuscript was self-reviewed before release using three complementary
frameworks: a systematic **peer-review** pass (methodology, statistics,
reproducibility, ethics, figures, reporting), a **ScholarEval**-style dimension
scoring, and a **critical-thinking** pass on the central claims. This document
records the review and how comments were addressed; it is provided for
transparency and is not part of the manuscript.

## Summary assessment

A methods paper introducing *cross-unit temporal disaggregation*: estimating the
daily sales of low-frequency reporters in an affiliate ecosystem by transferring
a shared behavioral-proxy elasticity learned on calibration units. The core idea
is sound and clearly novel relative to classical (per-series) disaggregation; the
validation design (leave-units-out) directly tests the transfer claim; the
released dataset and code make the work reproducible. **Recommendation: minor
revisions** (addressed below).

## ScholarEval dimension scores (1-5)

| Dimension | Score | Notes |
|---|---|---|
| Problem formulation | 5 | Real, well-motivated; granularity heterogeneity is a genuine, underserved problem. |
| Literature grounding | 4 | Correctly positioned vs. Chow-Lin/Denton, nowcasting/MIDAS, panel/transfer learning; 24 verified references. |
| Methodology | 4 | Clean derivation; intercept cancellation is the key insight; benchmark constraint exact. |
| Data & reproducibility | 5 | Seeded synthetic generator, public CSVs, all code released; anonymization protocol explicit. |
| Analysis & statistics | 3.5 | Cross-validated with fold s.d.; honest about the power-vs-linear tie. No formal poolability test (left to future work). |
| Results interpretation | 4 | Claims matched to evidence; no over-reach; limitations frank. |
| Writing & structure | 4 | IMRAD + case study; figures self-contained. |

## Major comments (and resolution)

1. **The estimated-elasticity method is only marginally better than raw clicks
   on this dataset.** *Resolution:* this is reported honestly; the beta-sensitivity
   sweep (Fig. 3D) shows the advantage is a robustness guarantee that grows as
   the true elasticity departs from 1. Added explicit text that the on-dataset
   difference is within fold variability and not, alone, statistically meaningful.

2. **The shared-coefficient assumption is asserted, not formally tested.**
   *Resolution:* the per-brand elasticity spread (s.d. 0.052) is reported as
   informal support; the Discussion now states that a formal poolability test is
   the natural confirmatory step, and that the random-coefficient/partial-pooling
   relaxation is available when brands differ systematically.

3. **External validity: evidence is on synthetic data only.** *Resolution:*
   acknowledged in Limitations. Justified by the legal/commercial sensitivity of
   the real partner data; the generator embeds realistic structure and an honest
   assumption violation (s_beta > 0). Validation on other ecosystems' calibration
   units is flagged as future work.

## Minor comments (resolved)

- Fixed a typo ("syntheticdata" -> "synthetic-data").
- Bolding of the best method in Table 2 is retained but the text now flags the
  statistical tie with proxy-linear on this population.
- Aggregate-conservation error reported numerically (~ 9x10^-5 %), confirming the
  benchmark constraint.
- Data-availability statement points to the public repository; dataset is
  reproducible from the published seed.

## Critical-thinking pass on the headline claims

- *"The proxy enables accurate daily estimation."* Supported: held-out R2 = 0.86,
  17.6 % sMAPE reduction over uniform, on brands unseen during estimation.
- *"The relationship transfers across brands."* Supported by construction of the
  leave-units-out test (beta-hat estimated only on other brands) and by beta-hat = 0.882 vs
  true 0.88.
- *"Conservation is exact."* True by construction of the weight normalization;
  empirically ~ 0.
- *No causal claim* is made from the proxy-sales association; the method is
  explicitly an estimation/disaggregation tool, and daily outputs are labelled as
  modeled, not measured.
