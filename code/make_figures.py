#!/usr/bin/env python3
"""
make_figures.py - journal-quality figures (Okabe-Ito, vector PDF).
Produces paper/figures/*.pdf (+ png) from data/ and results/.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA, RES = ROOT / "data", ROOT / "results"
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

OK = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#000000']
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.prop_cycle": mpl.cycler(color=OK), "figure.dpi": 120,
    "savefig.dpi": 300, "savefig.bbox": "tight", "axes.grid": False,
})


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", name)


# ----------------------------------------------------------------- load
proxy = pd.read_csv(DATA / "behavioral_proxy_daily.csv", parse_dates=["date"])
cal = pd.read_csv(DATA / "sales_calibration_daily.csv", parse_dates=["date"])
reg = pd.read_csv(DATA / "brand_registry.csv")
per_brand = pd.read_csv(RES / "per_brand_beta.csv")
cv = pd.read_csv(RES / "cv_results_by_fold.csv")
preds = pd.read_csv(RES / "holdout_predictions.csv", parse_dates=["date"])

# =================================================================== FIG 1: the problem
tb = reg[(reg.role == "target") & (reg.reporting_granularity == "monthly")].brand_id.iloc[0]
px = proxy[proxy.brand_id == tb].sort_values("date")
px = px[px.date.dt.year == 2024]
fig, ax = plt.subplots(figsize=(5.6, 2.6))
ax.plot(px.date, px.proxy_events, color=OK[1], lw=0.9, label="Daily behavioral proxy (observed)")
ax.set_ylabel("Proxy events / day", color=OK[1])
ax.tick_params(axis="y", labelcolor=OK[1])
despine(ax)
ax2 = ax.twinx()
month_start = pd.to_datetime([f"2024-{m:02d}-15" for m in range(1, 13)])
for ms in month_start:
    ax2.axvspan(ms - pd.Timedelta(days=14), ms + pd.Timedelta(days=14),
                ymin=0.0, ymax=0.12, color=OK[5], alpha=0.25)
ax2.plot([], [], color=OK[5], lw=6, alpha=0.4, label="Reported sales (one monthly total)")
ax2.set_yticks([])
ax2.spines["top"].set_visible(False)
lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
labs = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
ax.legend(lines, labs, frameon=False, loc="upper left", fontsize=6.5)
ax.set_xlabel("Date (2024)")
ax.set_title(f"Granularity mismatch for a target unit ({tb}): high-frequency proxy vs one monthly sales total")
save(fig, "fig1_granularity_problem")

# =================================================================== FIG 2: beta recovery
beta_true = 0.88
fig, ax = plt.subplots(figsize=(3.5, 2.8))
ax.hist(per_brand.beta_b, bins=10, color=OK[1], alpha=0.75, edgecolor="white")
ax.axvline(beta_true, color=OK[5], ls="--", lw=1.4, label=f"True shared $\\beta$ = {beta_true}")
ax.axvline(per_brand.beta_b.mean(), color=OK[2], ls="-", lw=1.4,
           label=f"Mean of per-brand $\\hat{{\\beta}}_b$ = {per_brand.beta_b.mean():.3f}")
ax.set_xlabel("Per-brand estimated elasticity $\\hat{\\beta}_b$")
ax.set_ylabel("Number of calibration brands")
ax.set_title("Recovery of the shared proxy$\\to$sales elasticity")
ax.legend(frameon=False, fontsize=6.5)
despine(ax)
save(fig, "fig2_beta_recovery")

# =================================================================== FIG 3: results multi-panel
fig = plt.figure(figsize=(7.1, 5.4))
gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.32)

# (A) scatter true vs estimated daily (proxy power)
axA = fig.add_subplot(gs[0, 0])
a = preds.sales_units.to_numpy(float)
f = preds.est_proxy_power.to_numpy(float)
ssres = np.sum((a - f) ** 2); sstot = np.sum((a - a.mean()) ** 2)
R2 = 1 - ssres / sstot
axA.scatter(a, f, s=4, alpha=0.15, color=OK[4], edgecolors="none")
lim = [0, np.percentile(a, 99.5)]
axA.plot(lim, lim, color="black", lw=0.8, ls="--")
axA.set_xlim(lim); axA.set_ylim(lim)
axA.set_xlabel("True daily sales (units)")
axA.set_ylabel("Estimated daily sales (units)")
axA.set_title(f"Held-out reconstruction (proxy)\n$R^2$ = {R2:.3f}")
despine(axA)

# (B) method comparison bars (sMAPE & R2) with fold error bars
axB = fig.add_subplot(gs[0, 1])
order = ["uniform", "proxy_linear(beta=1)", "proxy_power(beta_hat)"]
labels = ["Uniform", "Proxy linear\n($\\beta$=1)", "Proxy power\n($\\hat{\\beta}$)"]
m = cv.groupby("method").sMAPE.agg(["mean", "std"]).reindex(order)
x = np.arange(len(order))
axB.bar(x, m["mean"], yerr=m["std"], capsize=3,
        color=[OK[5], OK[0], OK[2]], alpha=0.9, edgecolor="white")
axB.set_xticks(x); axB.set_xticklabels(labels, fontsize=6.5)
axB.set_ylabel("sMAPE (%)  - lower is better")
axB.set_title("Daily reconstruction error (5-fold leave-units-out)")
for xi, v in zip(x, m["mean"]):
    axB.text(xi, v + 0.3, f"{v:.1f}", ha="center", fontsize=6.5)
despine(axB)

# (C) reconstruction time-series example for one held-out brand-month
axC = fig.add_subplot(gs[1, 0])
ex_b = preds.brand_id.iloc[0]
sub = preds[preds.brand_id == ex_b].sort_values("date")
sub = sub[sub.date.dt.to_period("M") == sub.date.dt.to_period("M").iloc[0]]
axC.plot(sub.date, sub.sales_units, color="black", lw=1.3, label="True", zorder=3)
axC.plot(sub.date, sub.est_uniform, color=OK[5], lw=1.0, ls=":", label="Uniform")
axC.plot(sub.date, sub.est_proxy_power, color=OK[2], lw=1.1, ls="-", label="Proxy power")
axC.set_xlabel("Day"); axC.set_ylabel("Sales (units)")
axC.set_title(f"Within-month shape recovery ({ex_b})")
axC.legend(frameon=False, fontsize=6.5)
axC.tick_params(axis="x", rotation=30)
despine(axC)

# (D) beta sensitivity sweep - why estimate beta
axD = fig.add_subplot(gs[1, 1])
rng = np.random.default_rng(123)
betas = np.linspace(0.3, 1.7, 11)


def smape(av, fv):
    d = np.abs(av) + np.abs(fv); k = d > 0
    return 100 * np.mean(2 * np.abs(fv[k] - av[k]) / d[k])


def sweep_once(true_beta):
    days = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    dow = days.dayofweek.to_numpy()
    wk = np.array([1.05, 1.08, 1.06, 1.04, 1.10, 0.82, 0.78])[dow]
    res = {"uniform": [], "linear": [], "power": []}
    series = []
    for bk in range(12):
        lvl = np.exp(rng.normal(np.log(800), 0.8))
        x = rng.poisson(np.maximum(lvl * wk * (1 + 0.2 * np.sin(np.arange(len(days)) / 58)), 1.0)).astype(float)
        y = np.round(np.exp(rng.normal(np.log(0.05), 0.4)) * np.power(np.maximum(x, 1e-9), true_beta)
                     * rng.lognormal(0, 0.18, len(days))).astype(float)
        series.append((x, y))
    # estimate beta from first 8 brands (within), test on last 4
    lx = np.concatenate([np.log(np.maximum(s[0], 1)) for s in series[:8]])
    ly = np.concatenate([np.log(np.maximum(s[1], 1)) for s in series[:8]])
    g = np.concatenate([[i] * len(series[i][0]) for i in range(8)])
    lxd = lx - pd.Series(lx).groupby(g).transform("mean").to_numpy()
    lyd = ly - pd.Series(ly).groupby(g).transform("mean").to_numpy()
    bhat = np.sum(lxd * lyd) / np.sum(lxd * lxd)
    for (x, y) in series[8:]:
        df = pd.DataFrame({"date": days, "x": x, "y": y})
        df["mon"] = df.date.dt.to_period("M")
        for meth, b in [("uniform", None), ("linear", 1.0), ("power", bhat)]:
            fhat = np.zeros(len(df))
            for _, gg in df.groupby("mon"):
                idx = gg.index
                w = np.ones(len(gg)) if b is None else np.power(np.maximum(gg.x.values, 1e-9), b)
                w = w / w.sum() if w.sum() > 0 else np.ones(len(gg)) / len(gg)
                fhat[idx] = w * gg.y.sum()
            res[meth].append(smape(df.y.values, fhat))
    return {k: np.mean(v) for k, v in res.items()}


sw = pd.DataFrame([dict(beta=b, **sweep_once(b)) for b in betas])
axD.plot(sw.beta, sw.uniform, marker="s", ms=3, color=OK[5], label="Uniform")
axD.plot(sw.beta, sw.linear, marker="^", ms=3, color=OK[0], label="Proxy linear ($\\beta$=1)")
axD.plot(sw.beta, sw.power, marker="o", ms=3, color=OK[2], label="Proxy power ($\\hat{\\beta}$)")
axD.axvline(1.0, color="grey", lw=0.6, ls=":")
axD.set_xlabel("True proxy$\\to$sales elasticity $\\beta$")
axD.set_ylabel("sMAPE (%)")
axD.set_title("Why estimate $\\beta$: sensitivity sweep")
axD.legend(frameon=False, fontsize=6.5)
despine(axD)

for ax, lab in zip([axA, axB, axC, axD], "ABCD"):
    ax.text(-0.16, 1.06, lab, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
save(fig, "fig3_results")

print("\nKey numbers for the paper:")
print(f"  held-out R2 (proxy)          : {R2:.3f}")
print(f"  per-brand beta mean +/- sd   : {per_brand.beta_b.mean():.3f} +/- {per_brand.beta_b.std():.3f}")
