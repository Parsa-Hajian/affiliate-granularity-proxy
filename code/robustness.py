#!/usr/bin/env python3
"""
robustness.py
===================================================================
MODEL-MISSPECIFICATION STUDY (responds to the reviewer's central point).

The main benchmark's data-generating process shares the estimator's functional
form, so it can only verify correctness and sensitivity. Here we deliberately
*break* that match and measure how the method degrades. Two independent
violations are swept:

  (1) PROXY-INDEPENDENT SALES (phi): a fraction phi of each day's sales comes
      from a component uncorrelated with the proxy (e.g. promotions, offline
      drivers, bot/measurement error in the proxy). phi = 0 is the matched
      case; phi = 1 means sales carry no proxy information at all.

  (2) NON-LOG-LINEAR LINK: sales saturate in the proxy (a concave plateau)
      rather than following a clean power law, so the log-log estimator is
      misspecified by construction.

For each setting we estimate beta on calibration brands and disaggregate
held-out brands' monthly totals, reporting the WITHIN-MONTH R^2 (shape-only).
The honest expectation: as the violation grows, the proxy method degrades
GRACEFULLY toward the uniform baseline (within-R^2 -> 0); it never does worse,
because the weights still conserve the total.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"; RES.mkdir(exist_ok=True)
FIG = ROOT / "paper" / "figures"; FIG.mkdir(parents=True, exist_ok=True)

OK = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#000000']
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                     "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
SEED = 4242
BETA = 0.88
N_BRANDS, N_CAL = 18, 12


def make_panel(rng, n, days):
    dow = days.dayofweek.to_numpy()
    wk = np.array([1.05, 1.08, 1.06, 1.04, 1.10, 0.82, 0.78])[dow]
    series = []
    for _ in range(n):
        lvl = np.exp(rng.normal(np.log(800), 0.8))
        x = rng.poisson(np.maximum(lvl * wk * (1 + 0.2 * np.sin(np.arange(len(days)) / 58)), 1.0)).astype(float)
        series.append(x)
    return series


def within_r2(df, key, beta):
    x = df.proxy_events.to_numpy(float)
    xb = np.ones_like(x) if beta is None else np.power(np.maximum(x, 1e-9), beta)
    s = pd.Series(xb, index=df.index)
    wsum = s.groupby(df[key]).transform("sum").to_numpy()
    tot = df.groupby(key).sales_units.transform("sum").to_numpy()
    n = df.groupby(key).sales_units.transform("size").to_numpy()
    w = np.where(wsum > 0, xb / wsum, 1.0 / n)
    f = w * tot
    y = df.sales_units.to_numpy(float); ybar = tot / n
    wy, wf = y - ybar, f - ybar
    return 1 - np.sum((wy - wf) ** 2) / np.sum(wy ** 2)


def estimate_beta(df):
    d = df[(df.sales_units > 0) & (df.proxy_events > 0)]
    lx, ly = np.log(d.proxy_events.to_numpy()), np.log(d.sales_units.to_numpy())
    g = d.brand_id.to_numpy()
    lxd = lx - pd.Series(lx).groupby(g).transform("mean").to_numpy()
    lyd = ly - pd.Series(ly).groupby(g).transform("mean").to_numpy()
    return float(np.sum(lxd * lyd) / np.sum(lxd * lxd))


def build(rng, xs, days, phi, nonlinear):
    """Return a long DataFrame of (brand, day, proxy, sales) for the given violation."""
    rows = []
    for i, x in enumerate(xs):
        if nonlinear:                       # saturating concave link (misspecified)
            base = np.power(x, BETA)
            link = base / (1 + base / np.quantile(base, 0.6))
            mu = np.exp(rng.normal(np.log(0.05), 0.4)) * link * 200
        else:
            mu = np.exp(rng.normal(np.log(0.05), 0.4)) * np.power(np.maximum(x, 1e-9), BETA)
        s_proxy = mu * rng.lognormal(0, 0.18, len(x))
        promo = rng.gamma(2.0, 1.0, len(x))
        spikes = (rng.random(len(x)) < 0.04) * rng.uniform(5, 15, len(x))
        s_indep = (promo + spikes)
        s_proxy *= 1.0 / max(s_proxy.mean(), 1e-9)
        s_indep *= 1.0 / max(s_indep.mean(), 1e-9)
        y = np.round(1000 * ((1 - phi) * s_proxy + phi * s_indep)).astype(float)
        bid = f"B{i:02d}"
        rows.append(pd.DataFrame(dict(brand_id=bid, date=days, proxy_events=x, sales_units=y)))
    df = pd.concat(rows, ignore_index=True)
    df["M"] = df.brand_id + "|" + df.date.dt.to_period("M").astype(str)
    return df


def sweep(nonlinear, phis):
    rng = np.random.default_rng(SEED + int(nonlinear))
    days = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    xs = make_panel(rng, N_BRANDS, days)
    out = []
    for phi in phis:
        df = build(rng, xs, days, phi, nonlinear)
        cal = df[df.brand_id.isin([f"B{i:02d}" for i in range(N_CAL)])]
        test = df[~df.brand_id.isin([f"B{i:02d}" for i in range(N_CAL)])]
        bhat = estimate_beta(cal)
        out.append(dict(phi=phi, nonlinear=nonlinear, beta_hat=round(bhat, 3),
                        within_R2_proxy=round(within_r2(test, "M", bhat), 4),
                        within_R2_uniform=round(within_r2(test, "M", None), 4)))
    return pd.DataFrame(out)


phis = np.round(np.linspace(0, 1, 11), 2)
res = pd.concat([sweep(False, phis), sweep(True, phis)], ignore_index=True)
res.to_csv(RES / "robustness.csv", index=False)
print(res.to_string(index=False))

# ----------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
for ax, nl, title in zip(axes, [False, True],
                         ["Proxy-independent sales fraction", "+ non-log-linear (saturating) link"]):
    d = res[res.nonlinear == nl]
    ax.plot(d.phi, d.within_R2_proxy, marker="o", ms=3, color=OK[2], label="Proxy (estimated $\\hat\\beta$)")
    ax.plot(d.phi, d.within_R2_uniform, marker="s", ms=3, color=OK[5], label="Uniform")
    ax.axhline(0, color="grey", lw=0.6, ls=":")
    ax.set_xlabel("$\\phi$ = share of sales independent of the proxy")
    ax.set_ylabel("Within-month $R^2$ (shape only)")
    ax.set_title(title)
    ax.set_ylim(-0.05, 0.4)
    ax.legend(frameon=False, fontsize=6.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
for ax, lab in zip(axes, "AB"):
    ax.text(-0.16, 1.07, lab, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"fig5_misspecification.{ext}")
print("\nwrote fig5_misspecification")
print(f"matched (phi=0): proxy within-R2 = {res[(~res.nonlinear)&(res.phi==0)].within_R2_proxy.iloc[0]:.3f}")
print(f"fully indep (phi=1): proxy within-R2 = {res[(~res.nonlinear)&(res.phi==1)].within_R2_proxy.iloc[0]:.3f}")
