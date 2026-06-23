#!/usr/bin/env python3
"""
generate_dataset.py
===================================================================
Reproducible generator for the ANONYMIZED, SYNTHETIC affiliate
granularity dataset released with the paper:

    "Estimating Daily Sales from a Shared Behavioral Proxy:
     Cross-Unit Temporal Disaggregation in an Affiliate-Marketing
     Ecosystem"

PRIVACY / LEGAL NOTE
--------------------
This script emits a *fully synthetic* dataset. It is CALIBRATED to the
realistic empirical structure of an operational affiliate pipeline
(relationship strength between a behavioral web proxy and sales, weekly
and annual seasonality, magnitude heterogeneity across partners), but it
contains NO real partner names and NO real values. Every number is a
fresh random draw, then rescaled by a per-brand secret factor, so no
released figure matches any real figure. Brands are labelled with neutral
codes (BR-001 ...) and generic sector tags. The dataset is therefore safe
for public release and fully reproducible from the fixed seed below.

DATA-GENERATING PROCESS (the "ground truth" the paper recovers)
---------------------------------------------------------------
For brand b on day t:
  proxy_events  x_{b,t}  ~ behavioral signal (clicks/sessions/GA4 events)
                           level_b * weekly(t) * annual(t) * trend(t) * noise
  daily sales   y_{b,t}  = exp(alpha_b) * x_{b,t}^{beta_b} * lognormal_noise
                           with SHARED elasticity beta plus a small
                           per-brand deviation  beta_b = beta + N(0, s_beta).
The paper's modelling assumption is that beta is shared across brands
(partial pooling); s_beta>0 makes the assumption only *approximately*
true, which is what the validation design is meant to stress-test.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 20260623
rng = np.random.default_rng(SEED)

OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------- config
START = "2023-01-01"
END   = "2025-12-31"
dates = pd.date_range(START, END, freq="D")
T = len(dates)

N_BRANDS = 60
SECTORS = ["Travel", "Electronics", "Food", "Fitness",
           "Fashion", "Events", "Mobility", "Beauty"]

# Shared structural parameters (the thing the method learns)
BETA_SHARED = 0.88           # shared proxy->sales elasticity (log-log slope)
S_BETA      = 0.05           # small per-brand slope deviation (assumption slack)
NOISE_SD    = 0.18           # multiplicative lognormal noise on daily sales

# ----------------------------------------------------------------- brands
def make_brands(n: int) -> pd.DataFrame:
    ids = [f"BR-{i:03d}" for i in range(1, n + 1)]
    sectors = rng.choice(SECTORS, size=n)
    # ~40% of brands are transactional (daily) -> calibration units;
    # the rest report monthly or weekly aggregates -> target units.
    role = rng.choice(["calibration", "target"], size=n, p=[0.40, 0.60])
    gran = []
    for r in role:
        if r == "calibration":
            gran.append("daily")
        else:
            gran.append(rng.choice(["monthly", "weekly"], p=[0.7, 0.3]))
    # heterogeneous magnitude across ~2 orders of magnitude ("big scale")
    base_level = np.exp(rng.normal(np.log(900), 0.9, size=n))      # mean daily events
    alpha = rng.normal(np.log(0.05), 0.45, size=n)                 # brand conversion intercept
    beta_b = BETA_SHARED + rng.normal(0, S_BETA, size=n)           # per-brand slope
    aov = np.exp(rng.normal(np.log(38), 0.5, size=n))              # avg order value (EUR)
    secret_scale = np.exp(rng.normal(0, 0.15, size=n))             # anonymizing rescale
    return pd.DataFrame(dict(
        brand_id=ids, sector=sectors, reporting_granularity=gran, role=role,
        _base_level=base_level, _alpha=alpha, _beta_b=beta_b,
        _aov=aov, _scale=secret_scale,
    ))

brands = make_brands(N_BRANDS)

# ----------------------------------------------------------------- seasonality
dow = dates.dayofweek.to_numpy()                     # 0=Mon .. 6=Sun
doy = dates.dayofyear.to_numpy()
# weekly pattern: mild weekday peak, weekend dip (consumer affiliate traffic)
weekly_base = np.array([1.05, 1.08, 1.06, 1.04, 1.10, 0.82, 0.78])[dow]
# annual pattern: Q4 / promo uplift + summer travel bump
annual = (1.0
          + 0.18 * np.sin(2 * np.pi * (doy - 80) / 365.25)    # spring/summer
          + 0.22 * np.exp(-((doy - 330) ** 2) / (2 * 18 ** 2)))  # late-Nov peak
trend = np.linspace(1.0, 1.15, T)                     # slow growth

# ----------------------------------------------------------------- emit rows
proxy_rows, cal_daily = [], []
tgt_agg = []

for _, b in brands.iterrows():
    # daily proxy (behavioral events)
    brand_week = weekly_base * np.exp(rng.normal(0, 0.03, 7)[dow])  # tiny brand-specific dow tweak
    mu_x = b._base_level * brand_week * annual * trend
    x = rng.poisson(np.maximum(mu_x, 1.0)).astype(float)
    x = np.maximum(x, 0)

    # daily sales (shared elasticity + per-brand intercept/slope + noise)
    eps = rng.lognormal(mean=0.0, sigma=NOISE_SD, size=T)
    mu_y = np.exp(b._alpha) * np.power(np.maximum(x, 1e-9), b._beta_b)
    y = np.round(mu_y * eps).astype(int)
    y = np.maximum(y, 0)
    value = np.round(y * b._aov * b._scale, 2)        # rescaled EUR value

    bid = b.brand_id
    # proxy is observed for EVERY brand (released)
    proxy_rows.append(pd.DataFrame(dict(brand_id=bid, date=dates, proxy_events=x.astype(int))))

    if b.role == "calibration":
        # true daily sales released (these are the labelled units)
        cal_daily.append(pd.DataFrame(dict(
            brand_id=bid, date=dates, sales_units=y, sales_value_eur=value)))
    else:
        # target unit: only the AGGREGATE per reporting period is released
        df = pd.DataFrame(dict(date=dates, sales_units=y, sales_value_eur=value))
        if b.reporting_granularity == "monthly":
            df["period"] = df["date"].dt.to_period("M")
        else:  # weekly (ISO weeks)
            df["period"] = df["date"].dt.to_period("W")
        g = df.groupby("period").agg(
            period_start=("date", "min"), period_end=("date", "max"),
            sales_units=("sales_units", "sum"), sales_value_eur=("sales_value_eur", "sum")).reset_index(drop=True)
        g.insert(0, "granularity", b.reporting_granularity)
        g.insert(0, "brand_id", bid)
        tgt_agg.append(g)

proxy = pd.concat(proxy_rows, ignore_index=True)
cal = pd.concat(cal_daily, ignore_index=True)
tgt = pd.concat(tgt_agg, ignore_index=True)

# public brand registry (no private columns)
registry = brands[["brand_id", "sector", "reporting_granularity", "role"]].copy()

# ----------------------------------------------------------------- write
proxy.to_csv(OUT / "behavioral_proxy_daily.csv", index=False)
cal.to_csv(OUT / "sales_calibration_daily.csv", index=False)
tgt.to_csv(OUT / "sales_target_aggregate.csv", index=False)
registry.to_csv(OUT / "brand_registry.csv", index=False)

print("=== synthetic affiliate dataset written ===")
print(f"seed                 : {SEED}")
print(f"date range           : {START} .. {END}  ({T} days)")
print(f"brands               : {N_BRANDS}  "
      f"(calibration={int((registry.role=='calibration').sum())}, "
      f"target={int((registry.role=='target').sum())})")
print(f"  granularity mix    : {registry.reporting_granularity.value_counts().to_dict()}")
print(f"behavioral_proxy_daily.csv : {len(proxy):>7} rows")
print(f"sales_calibration_daily.csv: {len(cal):>7} rows")
print(f"sales_target_aggregate.csv : {len(tgt):>7} rows")
print(f"brand_registry.csv         : {len(registry):>7} rows")
print(f"shared elasticity beta     : {BETA_SHARED}  (+/- {S_BETA} per brand)")
