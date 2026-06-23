# Estimating Daily Sales from a Shared Behavioral Proxy

**Cross-Unit Temporal Disaggregation in an Affiliate-Marketing Ecosystem**

Parsa Hajiannejad, Carlo di Bello — *Universitybox*

This repository accompanies the paper of the same name. It contains the
manuscript (LaTeX + compiled PDF), a fully synthetic and anonymized dataset, and
all code to reproduce the dataset, the estimation method, the experiments, and
the figures.

---

## What problem does this solve?

An affiliate platform consolidates sales reports from many partner brands, but
partners report at **different time granularities** — some daily, some weekly,
some only one monthly total. Analytics need a single **daily** panel, so the
missing daily detail of low-frequency reporters must be estimated.

We recover it from a **high-frequency behavioral proxy** (web tracking events
such as clicks/sessions, available daily for every brand). The key idea is that
the proxy→sales relationship is **shared across brands**: we learn it on
*calibration* brands that have true daily sales, then transfer it to *target*
brands that report only aggregates. A per-brand intercept cancels in the
disaggregation weights, so only a shared elasticity is needed — a brand with no
daily history can still be disaggregated, while its reported total is conserved
exactly.

## Headline results (on the released synthetic benchmark)

The benchmark's data-generating process shares the estimator's functional form,
so it verifies **correctness and sensitivity**, not real-world validity (see the
paper's Limitations and the misspecification study).

- The proxy recovers ~**29%** (monthly) / ~**31%** (weekly) of the *within-period*
  (shape-only) variance that a uniform split leaves entirely unexplained.
- Using the proxy at all drives the gain; estimating the elasticity is a
  **robustness safeguard** (statistically tied with raw-proxy on this data).
- Under deliberate misspecification the method **degrades gracefully** to the
  uniform baseline and never below it.
- Aggregate conservation is exact (a constraint, not an accuracy result).

## Repository layout

```
affiliate-granularity-proxy/
├── paper/
│   ├── main.tex            # manuscript (LaTeX)
│   ├── main.pdf            # compiled paper
│   ├── refs.bib            # bibliography (24 verified references)
│   └── figures/            # figures (PDF + PNG)
├── code/
│   ├── generate_dataset.py # reproducible synthetic-data generator (seeded)
│   ├── run_experiments.py  # estimation + repeated leave-units-out (monthly+weekly)
│   ├── robustness.py       # model-misspecification study (graceful degradation)
│   └── make_figures.py     # journal-quality figures
├── data/                   # released synthetic dataset (CSV) + dictionary
├── results/                # metrics + predictions produced by the experiments
├── REVIEW.md               # internal peer-review / ScholarEval record
├── RESPONSE_TO_REVIEW.md   # point-by-point response to external review
├── requirements.txt
└── LICENSE                 # MIT
```

## Reproduce everything

```bash
pip install -r requirements.txt

python code/generate_dataset.py     # writes data/*.csv (seed = 20260623)
python code/run_experiments.py      # repeated leave-units-out (monthly+weekly) -> results/*
python code/robustness.py           # misspecification study -> results/robustness.csv, fig5
python code/make_figures.py         # writes paper/figures/*.{pdf,png}

# compile the paper
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## A note on large files

To keep the repository light, the **large daily CSVs**, the full `results/`
predictions, the figure images, and the compiled `paper/main.pdf` are **not
committed**; they regenerate **bit-for-bit** from the seeded scripts (see
commands above). The small `data/brand_registry.csv`, `results/metrics.json`,
and `results/cv_summary_*.csv` / `results/robustness.csv` are included so the
dataset structure and headline numbers are visible without running anything.

## Data & privacy

The dataset is **fully synthetic**. It is calibrated to the realistic structure
of an operational pipeline (proxy↔sales relationship, weekly/annual seasonality,
cross-brand magnitude heterogeneity) but contains **no real partner names and no
real values** — every number is a fresh seeded random draw, brands are coded
`BR-001…`, and a per-brand secret factor rescales magnitudes. Re-identification
is impossible by construction. See `data/DATA_DICTIONARY.md` and Section 5 of the
paper.

## License

Code and data released under the MIT License (see `LICENSE`).

## Citation

```bibtex
@misc{HajiannejadDiBello2026,
  author = {Hajiannejad, Parsa and di Bello, Carlo},
  title  = {Estimating Daily Sales from a Shared Behavioral Proxy:
            Cross-Unit Temporal Disaggregation in an Affiliate-Marketing Ecosystem},
  year   = {2026},
  note   = {\url{https://github.com/Parsa-Hajian/affiliate-granularity-proxy}}
}
```
