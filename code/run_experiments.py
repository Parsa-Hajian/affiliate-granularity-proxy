#!/usr/bin/env python3
"""
run_experiments.py
===================================================================
Implements and validates the cross-unit temporal-disaggregation method.

METHOD
------
Shared log-log proxy->sales model with per-brand intercept:
      log E[y_{b,t}] = alpha_b + beta * log x_{b,t}.
To distribute a reported aggregate Y_{b,P} (period P) across its days t,
we use weights
      w_{b,t} = x_{b,t}^beta / sum_{s in P} x_{b,s}^beta ,
and set  yhat_{b,t} = w_{b,t} * Y_{b,P}.
The per-brand intercept alpha_b CANCELS in the weight ratio, so only the
SHARED elasticity beta is needed -> a target brand with no daily sales can
still be disaggregated. yhat sums exactly to Y_{b,P} (benchmark constraint).

beta is estimated on CALIBRATION units (true daily sales known) with a
within (fixed-effects) estimator that absorbs alpha_b.

VALIDATION (cross-unit transfer, train/test)
--------------------------------------------
5-fold leave-UNITS-out over calibration brands: estimate beta on the
training brands, then disaggregate each held-out brand's true monthly
totals and compare the reconstructed daily series to the known truth.
Baselines: UNIFORM (w=1/n) and PROXY-LINEAR (beta=1, raw clicks).
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
rng = np.random.default_rng(7)

# ----------------------------------------------------------------- load
proxy = pd.read_csv(DATA / "behavioral_proxy_daily.csv", parse_dates=["date"])
cal = pd.read_csv(DATA / "sales_calibration_daily.csv", parse_dates=["date"])
tgt = pd.read_csv(DATA / "sales_target_aggregate.csv",
                  parse_dates=["period_start", "period_end"])
reg = pd.read_csv(DATA / "brand_registry.csv")

cal = cal.merge(proxy, on=["brand_id", "date"], how="left")
cal["month"] = cal["date"].dt.to_period("M")


# ----------------------------------------------------------------- estimator
def estimate_beta(df: pd.DataFrame) -> float:
    """Within (fixed-effects) estimator of the shared log-log slope."""
    d = df[(df.sales_units > 0) & (df.proxy_events > 0)].copy()
    lx, ly = np.log(d.proxy_events.to_numpy()), np.log(d.sales_units.to_numpy())
    g = d.brand_id.to_numpy()
    # demean within brand
    lx_d = lx - pd.Series(lx).groupby(g).transform("mean").to_numpy()
    ly_d = ly - pd.Series(ly).groupby(g).transform("mean").to_numpy()
    return float(np.sum(lx_d * ly_d) / np.sum(lx_d * lx_d))


def disaggregate(period_days: pd.DataFrame, total: float, beta: float) -> np.ndarray:
    """Distribute `total` over the days of one period using proxy^beta weights."""
    x = period_days.proxy_events.to_numpy(dtype=float)
    if beta is None:                       # uniform
        w = np.ones_like(x)
    else:
        w = np.power(np.maximum(x, 1e-9), beta)
    s = w.sum()
    w = np.ones_like(x) / len(x) if s <= 0 else w / s
    return w * total


def smape(a: np.ndarray, f: np.ndarray) -> float:
    denom = np.abs(a) + np.abs(f)
    m = denom > 0
    return float(100.0 * np.mean(2.0 * np.abs(f[m] - a[m]) / denom[m]))


def r2(a: np.ndarray, f: np.ndarray) -> float:
    ss_res = np.sum((a - f) ** 2)
    ss_tot = np.sum((a - a.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def score_method(holdout: pd.DataFrame, beta) -> dict:
    """Reconstruct daily series for held-out brands by month and score vs truth."""
    a_all, f_all = [], []
    cons = []
    for (bid, mth), grp in holdout.groupby(["brand_id", "month"]):
        grp = grp.sort_values("date")
        total = grp.sales_units.sum()                     # pretend only the monthly total is known
        fhat = disaggregate(grp, total, beta)
        a_all.append(grp.sales_units.to_numpy(dtype=float))
        f_all.append(fhat)
        cons.append(abs(fhat.sum() - total))
    a = np.concatenate(a_all); f = np.concatenate(f_all)
    return dict(sMAPE=smape(a, f), MAE=float(np.mean(np.abs(a - f))),
                R2=r2(a, f), n_days=int(len(a)),
                aggregate_abs_error=float(np.mean(cons)))


# ----------------------------------------------------------------- full-sample beta + per-brand
beta_full = estimate_beta(cal)
per_brand = []
for bid, g in cal.groupby("brand_id"):
    per_brand.append(dict(brand_id=bid, beta_b=estimate_beta(g.assign(brand_id=bid)),
                          n_days=int((g.sales_units > 0).sum())))
per_brand = pd.DataFrame(per_brand)
per_brand.to_csv(RES / "per_brand_beta.csv", index=False)

# ----------------------------------------------------------------- 5-fold leave-units-out
cal_brands = sorted(cal.brand_id.unique())
rng.shuffle(cal_brands)
folds = np.array_split(cal_brands, 5)

rows, preds_for_fig = [], []
for k, test_brands in enumerate(folds):
    train = cal[~cal.brand_id.isin(test_brands)]
    test = cal[cal.brand_id.isin(test_brands)]
    beta_k = estimate_beta(train)                          # transfer: beta from OTHER brands only
    for name, b in [("proxy_power(beta_hat)", beta_k),
                    ("proxy_linear(beta=1)", 1.0),
                    ("uniform", None)]:
        m = score_method(test, b)
        m.update(fold=k, method=name, beta_used=(None if b is None else round(b, 4)),
                 n_test_brands=len(test_brands))
        rows.append(m)
    # keep one fold's reconstructions for the figures (our method)
    if k == 0:
        for (bid, mth), grp in test.groupby(["brand_id", "month"]):
            grp = grp.sort_values("date")
            fhat = disaggregate(grp, grp.sales_units.sum(), beta_k)
            tmp = grp[["brand_id", "date", "sales_units"]].copy()
            tmp["est_proxy_power"] = fhat
            tmp["est_uniform"] = disaggregate(grp, grp.sales_units.sum(), None)
            tmp["est_proxy_linear"] = disaggregate(grp, grp.sales_units.sum(), 1.0)
            preds_for_fig.append(tmp)

cv = pd.DataFrame(rows)
pd.concat(preds_for_fig, ignore_index=True).to_csv(RES / "holdout_predictions.csv", index=False)
cv.to_csv(RES / "cv_results_by_fold.csv", index=False)

summary = (cv.groupby("method")[["sMAPE", "MAE", "R2", "aggregate_abs_error"]]
           .agg(["mean", "std"]).round(4))
summary.to_csv(RES / "cv_summary.csv")

# ----------------------------------------------------------------- apply to true target units
tgt_daily = []
tgt_m = tgt.copy()
for _, r in tgt_m.iterrows():
    days = proxy[(proxy.brand_id == r.brand_id) &
                 (proxy.date >= r.period_start) & (proxy.date <= r.period_end)].sort_values("date")
    if days.empty:
        continue
    fhat = disaggregate(days, r.sales_units, beta_full)
    fval = disaggregate(days, r.sales_value_eur, beta_full)
    tgt_daily.append(pd.DataFrame(dict(
        brand_id=r.brand_id, date=days.date.to_numpy(),
        est_sales_units=np.round(fhat, 4), est_sales_value_eur=np.round(fval, 2))))
tgt_daily = pd.concat(tgt_daily, ignore_index=True)
tgt_daily.to_csv(RES / "target_daily_estimates.csv", index=False)

# conservation check on targets (should be ~0)
chk = (tgt_daily.groupby("brand_id").est_sales_units.sum()
       .reset_index().merge(tgt.groupby("brand_id").sales_units.sum().reset_index(),
                            on="brand_id"))
chk["abs_pct_err"] = 100 * (chk.est_sales_units - chk.sales_units).abs() / chk.sales_units

# ----------------------------------------------------------------- report
metrics = dict(
    seed_estimation=7,
    beta_true_shared=0.88,
    beta_estimated_full_sample=round(beta_full, 4),
    per_brand_beta_mean=round(per_brand.beta_b.mean(), 4),
    per_brand_beta_std=round(per_brand.beta_b.std(), 4),
    n_calibration_brands=len(cal_brands),
    n_target_brands=int(reg[reg.role == "target"].shape[0]),
    cv_summary={m: dict(sMAPE_mean=round(g.sMAPE.mean(), 3), sMAPE_std=round(g.sMAPE.std(), 3),
                        MAE_mean=round(g.MAE.mean(), 4), R2_mean=round(g.R2.mean(), 4))
                for m, g in cv.groupby("method")},
    target_aggregate_conservation_max_pct_err=round(chk.abs_pct_err.max(), 6),
    error_reduction_vs_uniform_pct=round(
        100 * (1 - cv[cv.method == "proxy_power(beta_hat)"].sMAPE.mean()
               / cv[cv.method == "uniform"].sMAPE.mean()), 2),
    error_reduction_vs_proxy_linear_pct=round(
        100 * (1 - cv[cv.method == "proxy_power(beta_hat)"].sMAPE.mean()
               / cv[cv.method == "proxy_linear(beta=1)"].sMAPE.mean()), 2),
)
with open(RES / "metrics.json", "w") as fh:
    json.dump(metrics, fh, indent=2)

print(json.dumps(metrics, indent=2))
print("\n=== CV summary (mean over 5 folds) ===")
print(summary)
