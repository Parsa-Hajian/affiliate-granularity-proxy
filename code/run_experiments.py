#!/usr/bin/env python3
"""
run_experiments.py
===================================================================
Implements and validates the cross-unit temporal-disaggregation method.

METHOD
------
Shared log-log proxy->sales model with per-brand intercept:
      log E[y_{b,t}] = alpha_b + beta * log x_{b,t}.
Distribute a reported aggregate Y_{b,P} over the days of period P with
      w_{b,t} = x_{b,t}^beta / sum_{s in P} x_{b,s}^beta ,  yhat = w * Y.
The intercept alpha_b cancels in the ratio -> only the SHARED beta is
needed, and yhat sums exactly to Y_{b,P} (benchmark constraint).
beta is estimated on CALIBRATION units with a within (fixed-effects) estimator.

VALIDATION (responding to reviewer comments)
--------------------------------------------
- Leave-UNITS-out CV, REPEATED over many random partitions (not one shuffle).
- Run for MONTHLY *and* WEEKLY reporting periods.
- Metrics include a WITHIN-PERIOD (period-demeaned) R^2 that isolates the
  *shape* recovery from the conserved-total contribution, plus scale-free
  errors (relative MAE per brand, and MASE vs the uniform naive). Pooled daily
  R^2 and sMAPE are also reported for continuity.
- Baselines: UNIFORM (beta=0) and PROXY-LINEAR (beta=1).

INTERPRETATION (stated plainly): the released benchmark's data-generating
process shares the estimator's functional form, so this study verifies
*correctness* and quantifies *sensitivity*; it does NOT establish that real
ecosystems satisfy the shared-elasticity assumption. See code/robustness.py
for a misspecification study and the paper's Limitations.
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

N_REPEATS = 30
N_FOLDS = 5

# ----------------------------------------------------------------- load
proxy = pd.read_csv(DATA / "behavioral_proxy_daily.csv", parse_dates=["date"])
cal = pd.read_csv(DATA / "sales_calibration_daily.csv", parse_dates=["date"])
tgt = pd.read_csv(DATA / "sales_target_aggregate.csv",
                  parse_dates=["period_start", "period_end"])
reg = pd.read_csv(DATA / "brand_registry.csv")

cal = cal.merge(proxy, on=["brand_id", "date"], how="left")
cal["M"] = cal["brand_id"] + "|" + cal["date"].dt.to_period("M").astype(str)
cal["W"] = cal["brand_id"] + "|" + cal["date"].dt.to_period("W").astype(str)


# ----------------------------------------------------------------- core (vectorized)
def estimate_beta(df: pd.DataFrame) -> float:
    d = df[(df.sales_units > 0) & (df.proxy_events > 0)]
    lx, ly = np.log(d.proxy_events.to_numpy()), np.log(d.sales_units.to_numpy())
    g = d.brand_id.to_numpy()
    lx_d = lx - pd.Series(lx).groupby(g).transform("mean").to_numpy()
    ly_d = ly - pd.Series(ly).groupby(g).transform("mean").to_numpy()
    return float(np.sum(lx_d * ly_d) / np.sum(lx_d * lx_d))


def fitted(df: pd.DataFrame, key: str, beta):
    """Vectorized disaggregated fit f and the period mean ybar for every row."""
    x = df.proxy_events.to_numpy(float)
    xb = np.ones_like(x) if beta is None else np.power(np.maximum(x, 1e-9), beta)
    s = pd.Series(xb, index=df.index)
    wsum = s.groupby(df[key]).transform("sum").to_numpy()
    tot = df.groupby(key).sales_units.transform("sum").to_numpy()
    n = df.groupby(key).sales_units.transform("size").to_numpy()
    w = np.where(wsum > 0, xb / wsum, 1.0 / n)
    return w * tot, tot / n


def score(df: pd.DataFrame, key: str, beta) -> dict:
    y = df.sales_units.to_numpy(float)
    f, ybar = fitted(df, key, beta)
    fu, _ = fitted(df, key, None)                      # uniform naive (MASE denom)
    wy, wf = y - ybar, f - ybar                        # within-period deviations
    denom = np.abs(y) + np.abs(f); mm = denom > 0
    smape = 100 * np.mean(2 * np.abs(f[mm] - y[mm]) / denom[mm])
    pooled = 1 - np.sum((y - f) ** 2) / np.sum((y - y.mean()) ** 2)
    within = 1 - np.sum((wy - wf) ** 2) / np.sum(wy ** 2)
    # per-brand scale-free relative MAE
    ae = pd.Series(np.abs(y - f), index=df.index)
    rel = (ae.groupby(df.brand_id).mean()
           / df.groupby("brand_id").sales_units.mean()).replace([np.inf], np.nan)
    mase = float(np.sum(np.abs(y - f)) / np.sum(np.abs(y - fu)))
    return dict(sMAPE=float(smape), MAE=float(np.mean(np.abs(y - f))),
                pooled_R2=float(pooled), within_R2=float(within),
                rel_MAE=float(rel.mean()), MASE_vs_uniform=mase)


def run_cv(key: str, repeats: int) -> pd.DataFrame:
    brands = np.array(sorted(cal.brand_id.unique()))
    rows = []
    for rep in range(repeats):
        rng = np.random.default_rng(1000 + rep)
        order = brands.copy(); rng.shuffle(order)
        for k, test_b in enumerate(np.array_split(order, N_FOLDS)):
            mask = cal.brand_id.isin(test_b)
            bhat = estimate_beta(cal[~mask])
            test = cal[mask]
            for name, b in [("proxy_power", bhat), ("proxy_linear", 1.0), ("uniform", None)]:
                m = score(test, key, b)
                m.update(repeat=rep, fold=k, method=name, freq=key)
                rows.append(m)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- full-sample beta + per-brand
beta_full = estimate_beta(cal)
per_brand = pd.DataFrame([
    dict(brand_id=bid, beta_b=estimate_beta(g), n_days=int((g.sales_units > 0).sum()),
         mean_daily_sales=float(g.sales_units.mean()))
    for bid, g in cal.groupby("brand_id")
])
per_brand.to_csv(RES / "per_brand_beta.csv", index=False)

# ----------------------------------------------------------------- repeated CV (monthly + weekly)
cv_m = run_cv("M", N_REPEATS)
cv_w = run_cv("W", N_REPEATS)
pd.concat([cv_m, cv_w], ignore_index=True).to_csv(RES / "cv_results_all.csv", index=False)


def summarize(df):
    out = {}
    for m, gg in df.groupby("method"):
        out[m] = {
            "sMAPE_mean": round(gg.sMAPE.mean(), 3), "sMAPE_sd": round(gg.sMAPE.std(), 3),
            "sMAPE_p2.5": round(gg.sMAPE.quantile(.025), 3),
            "sMAPE_p97.5": round(gg.sMAPE.quantile(.975), 3),
            "pooled_R2_mean": round(gg.pooled_R2.mean(), 3),
            "within_R2_mean": round(gg.within_R2.mean(), 3),
            "within_R2_sd": round(gg.within_R2.std(), 3),
            "rel_MAE_mean": round(gg.rel_MAE.mean(), 3),
            "MASE_vs_uniform_mean": round(gg.MASE_vs_uniform.mean(), 3),
        }
    return out


summary_m, summary_w = summarize(cv_m), summarize(cv_w)
pd.DataFrame(summary_m).T.to_csv(RES / "cv_summary_monthly.csv")
pd.DataFrame(summary_w).T.to_csv(RES / "cv_summary_weekly.csv")

# ----------------------------------------------------------------- holdout predictions (one monthly fold) for figures
rng = np.random.default_rng(1000)
order = np.array(sorted(cal.brand_id.unique())); rng.shuffle(order)
test_b = np.array_split(order, N_FOLDS)[0]
bk = estimate_beta(cal[~cal.brand_id.isin(test_b)])
test = cal[cal.brand_id.isin(test_b)].sort_values(["brand_id", "date"]).copy()
fp, _ = fitted(test, "M", bk); fu, _ = fitted(test, "M", None); fl, _ = fitted(test, "M", 1.0)
test["est_proxy_power"], test["est_uniform"], test["est_proxy_linear"] = fp, fu, fl
test[["brand_id", "date", "sales_units", "est_proxy_power", "est_uniform",
      "est_proxy_linear"]].to_csv(RES / "holdout_predictions.csv", index=False)

# ----------------------------------------------------------------- representativeness (cal vs target)
lvl = (proxy.merge(reg[["brand_id", "role"]], on="brand_id")
       .groupby(["brand_id", "role"]).proxy_events.mean().reset_index())
rep_check = lvl.groupby("role").proxy_events.agg(["mean", "median", "std"]).round(1).to_dict("index")

# ----------------------------------------------------------------- target application + conservation
tgt_daily = []
for _, r in tgt.iterrows():
    days = proxy[(proxy.brand_id == r.brand_id) &
                 (proxy.date >= r.period_start) & (proxy.date <= r.period_end)].sort_values("date")
    if days.empty:
        continue
    x = days.proxy_events.to_numpy(float)
    w = np.power(np.maximum(x, 1e-9), beta_full); w = w / w.sum() if w.sum() > 0 else np.ones_like(x) / len(x)
    tgt_daily.append(pd.DataFrame(dict(
        brand_id=r.brand_id, date=days.date.to_numpy(),
        est_sales_units=np.round(w * r.sales_units, 4),
        est_sales_value_eur=np.round(w * r.sales_value_eur, 2))))
tgt_daily = pd.concat(tgt_daily, ignore_index=True)
tgt_daily.to_csv(RES / "target_daily_estimates.csv", index=False)
chk = (tgt_daily.groupby("brand_id").est_sales_units.sum().reset_index()
       .merge(tgt.groupby("brand_id").sales_units.sum().reset_index(), on="brand_id"))
chk["abs_pct_err"] = 100 * (chk.est_sales_units - chk.sales_units).abs() / chk.sales_units

# ----------------------------------------------------------------- report
metrics = dict(
    n_repeats=N_REPEATS, n_folds=N_FOLDS,
    beta_true_shared=0.88, beta_estimated_full_sample=round(beta_full, 4),
    per_brand_beta_mean=round(per_brand.beta_b.mean(), 4),
    per_brand_beta_sd=round(per_brand.beta_b.std(), 4),
    n_calibration_brands=int((reg.role == "calibration").sum()),
    n_target_brands=int((reg.role == "target").sum()),
    monthly=summary_m, weekly=summary_w,
    target_conservation_max_pct_err=round(float(chk.abs_pct_err.max()), 6),
    representativeness_mean_daily_proxy_by_role=rep_check,
)
with open(RES / "metrics.json", "w") as fh:
    json.dump(metrics, fh, indent=2)

print(json.dumps(metrics, indent=2))
