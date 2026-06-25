"""
synth.py — Synthetic control (standard and augmented) for the milk-quota study.

A second, independent estimate of the abolition's effect, on assumptions that
differ from DiD: for each treated country it builds a counterfactual from a
combination of control countries chosen to reproduce that country's own
pre-2015 trajectory — so it absorbs the pre-trend by construction rather than
assuming parallel trends.

Two variants
------------
1. Standard SCM (`fit_one`): convex weights (w >= 0, sum w = 1). The synthetic
   is an interpolation inside the donors' convex hull — never extrapolates.
   Robust and interpretable, but cannot reproduce treated units that lie at the
   edge of / outside that hull (here, the largest producers: DE, IE).

2. Augmented SCM (`fit_one_augmented`): relaxes non-negativity (weights may be
   negative => controlled extrapolation), regularised by a ridge penalty
   lambda * ||w||^2, keeping sum w = 1. Solves the constrained ridge problem in
   closed form via a Lagrange multiplier:

       w = A^{-1}(Y0' y1 - (mu/2) 1),   A = Y0'Y0 + lambda I,
       mu/2 = (1' A^{-1} Y0' y1 - 1) / (1' A^{-1} 1)

   lambda trades off pre-period fit (low lambda, more extrapolation/overfit)
   against stability (high lambda); it is chosen by pre-period cross-validation.
   The cost of extrapolation: weights are no longer a real convex mix, so the
   counterfactual is less directly interpretable. Pre-fit (pre_rmspe) is the
   objective gauge of whether the relaxation actually helped.

Outcome in logs, so gaps read as ~% effects, directly comparable to DiD. The
`outcome` argument switches the analysed series ("milk_deliveries", default, or
"milk_price"); the treatment definition is identical across outcomes, but the
donor/control universe is outcome-specific by construction (see `_prep`).

Requires:  numpy, pandas, scipy
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _prep(panel: pd.DataFrame, log: bool = True, outcome: str = "milk_deliveries"):
    """Wide (date x country) outcome matrix, plus treated/control geo lists.

    Units with *no* data at all for this outcome are dropped automatically, so
    the donor/control universe is outcome-specific by construction: e.g. for
    "milk_price" the UK (absent from the MMO series) silently leaves the control
    group (15 -> 14). A *treated* unit going fully missing is instead an error
    (fail loud), since the treated set is fixed by the policy and must not shrink
    silently.

    NOTE: this only removes *fully* absent units. Units with a *partial* gap
    (e.g. Croatia, price from 2013 only) keep a column with NaNs; `_pre_block`'s
    row-wise dropna then governs how that gap propagates — see the coverage note
    in the handoff. Decide the donor-coverage policy before trusting price fits.
    """
    out = panel.copy()
    if log:
        if (out[outcome] <= 0).any():        # NaN <= 0 is False, so gaps are safe
            raise ValueError(f"zero/negative {outcome} breaks the log transform.")
        out["y"] = np.log(out[outcome])
    else:
        out["y"] = out[outcome]
    wide = out.pivot_table(index="date", columns="geo", values="y")
    wide = wide.dropna(axis=1, how="all")    # drop units with no data for this outcome

    present = set(wide.columns)
    flagged_treated = set(out.loc[out["treated"] == 1, "geo"].unique())
    missing_treated = sorted(flagged_treated - present)
    if missing_treated:
        raise ValueError(f"treated unit(s) with no {outcome} data: {missing_treated}")

    treated = sorted(flagged_treated & present)
    control = sorted(set(out.loc[out["treated"] == 0, "geo"].unique()) & present)
    return wide, treated, control


def _pre_block(wide, unit, donors, pre_end):
    """Pre-period treated vector and donor matrix, rows complete across both."""
    block = wide.loc[wide.index <= pd.Timestamp(pre_end), [unit] + list(donors)].dropna()
    return block[unit].to_numpy(), block[donors].to_numpy()


def _assemble(wide, unit, donors, w):
    """Build synthetic/actual/gap series from a weight vector."""
    synthetic = pd.Series(wide[donors].to_numpy() @ w, index=wide.index, name="synthetic")
    gap = wide[unit] - synthetic
    return synthetic, gap


# ----------------------------------------------------------------------------- standard SCM

def fit_one(wide: pd.DataFrame, unit: str, donors: list[str], pre_end: str) -> dict:
    """Standard convex SCM: simplex-constrained least squares on the pre path."""
    y1, Y0 = _pre_block(wide, unit, donors, pre_end)
    n = len(donors)

    def loss(w):
        r = y1 - Y0 @ w
        return float(r @ r)

    res = minimize(
        loss, np.full(n, 1.0 / n), method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=({"type": "eq", "fun": lambda w: w.sum() - 1.0},),
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    w = np.clip(res.x, 0, None)
    w = w / w.sum()
    synthetic, gap = _assemble(wide, unit, donors, w)
    return {"unit": unit, "weights": pd.Series(w, index=donors).sort_values(ascending=False),
            "actual": wide[unit], "synthetic": synthetic, "gap": gap,
            "pre_rmspe": float(np.sqrt(np.mean((y1 - Y0 @ w) ** 2)))}


# ----------------------------------------------------------------------------- augmented SCM

def _ridge_affine(y1, Y0, lam):
    """Closed-form constrained ridge: min ||y1 - Y0 w||^2 + lam||w||^2  s.t. 1'w=1.

    Weights are unbounded (extrapolation allowed); lam regularises it.
    """
    n = Y0.shape[1]
    A = Y0.T @ Y0 + lam * np.eye(n)
    Ainv_Yty = np.linalg.solve(A, Y0.T @ y1)
    Ainv_1 = np.linalg.solve(A, np.ones(n))
    mu_half = (np.ones(n) @ Ainv_Yty - 1.0) / (np.ones(n) @ Ainv_1)
    return Ainv_Yty - mu_half * Ainv_1


def choose_lambda(y1, Y0, grid=None, n_val: int = 12) -> float:
    """Pick lambda by holding out the last `n_val` pre-period months."""
    if grid is None:
        grid = np.logspace(-4, 2, 25)
    split = len(y1) - n_val
    yt, Yt, yv, Yv = y1[:split], Y0[:split], y1[split:], Y0[split:]
    errs = [np.mean((yv - Yv @ _ridge_affine(yt, Yt, lam)) ** 2) for lam in grid]
    return float(grid[int(np.argmin(errs))])


def fit_one_augmented(wide, unit, donors, pre_end, lam=None) -> dict:
    """Augmented (ridge) SCM with controlled extrapolation."""
    y1, Y0 = _pre_block(wide, unit, donors, pre_end)
    if lam is None:
        lam = choose_lambda(y1, Y0)
    w = _ridge_affine(y1, Y0, lam)
    synthetic, gap = _assemble(wide, unit, donors, w)
    return {"unit": unit, "lambda": lam,
            "weights": pd.Series(w, index=donors).sort_values(ascending=False),
            "actual": wide[unit], "synthetic": synthetic, "gap": gap,
            "pre_rmspe": float(np.sqrt(np.mean((y1 - Y0 @ w) ** 2)))}


# ----------------------------------------------------------------------------- drivers

def run_all_treated(panel, pre_end="2014-12-01", log=True, augmented=False,
                    outcome="milk_deliveries"):
    """Fit a (standard or augmented) synthetic control for every treated country.

    Returns (results dict, gaps DataFrame [date x unit], average gap).
    """
    wide, treated, control = _prep(panel, log, outcome)
    fit = fit_one_augmented if augmented else fit_one
    results = {u: fit(wide, u, control, pre_end) for u in treated}
    gaps = pd.DataFrame({u: r["gap"] for u, r in results.items()})
    return results, gaps, gaps.mean(axis=1)


def pre_rmspe_table(results: dict) -> pd.Series:
    """Per-country pre-period fit quality (lower = better)."""
    return pd.Series({u: r["pre_rmspe"] for u, r in results.items()}).sort_values()


def post_effects(gaps: pd.DataFrame, cutoff="2015-04-01") -> pd.Series:
    """Per-country average gap in the post period."""
    return gaps.loc[gaps.index >= pd.Timestamp(cutoff)].mean().sort_values(ascending=False)


def plot_average_gap(avg_gap: pd.Series, cutoff="2015-04-01", ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    ax.plot(avg_gap.index, avg_gap.values, color="#D85A30", lw=2)
    ax.axhline(0, color="grey", lw=1)
    ax.axvline(pd.Timestamp(cutoff), ls="--", color="grey")
    ax.set(title="Synthetic control: average treated - synthetic gap",
           xlabel="date", ylabel="log gap (~ proportional effect)")
    return ax


# ----------------------------------------------------------------------------- placebo inference

def _gap_rmspe(gap: pd.Series, pre_end="2014-12-01", cutoff="2015-04-01"):
    """Pre- and post-period RMSPE of a gap series, and their ratio."""
    pre = gap[gap.index <= pd.Timestamp(pre_end)].dropna()
    post = gap[gap.index >= pd.Timestamp(cutoff)].dropna()
    pre_r = float(np.sqrt(np.mean(pre.values ** 2)))
    post_r = float(np.sqrt(np.mean(post.values ** 2)))
    ratio = post_r / pre_r if pre_r > 0 else np.nan
    return pre_r, post_r, ratio


def placebo_gaps(panel, pre_end="2014-12-01", cutoff="2015-04-01",
                 augmented=True, log=True, outcome="milk_deliveries"):
    """In-space placebos: treat each control as if treated, donors = the other
    controls. Returns a dict of results per control, each with its gap and
    pre/post RMSPE ratio — the null distribution for inference. The number of
    placebos (hence the minimum attainable p-value) follows the outcome-specific
    control set automatically.
    """
    wide, treated, control = _prep(panel, log, outcome)
    fit = fit_one_augmented if augmented else fit_one
    out = {}
    for c in control:
        donors = [d for d in control if d != c]
        r = fit(wide, c, donors, pre_end)
        pre_r, post_r, ratio = _gap_rmspe(r["gap"], pre_end, cutoff)
        out[c] = {**r, "pre_r": pre_r, "post_r": post_r, "ratio": ratio}
    return out


def _statistic(gap, statistic, pre_end, cutoff):
    """Placebo test statistic from a gap series, plus the pre/post RMSPE."""
    pre_r, post_r, ratio = _gap_rmspe(gap, pre_end, cutoff)
    if statistic == "ratio":
        val = ratio
    elif statistic == "post_rmspe":
        val = post_r
    elif statistic == "post_gap":          # signed mean post gap (effect, ~%)
        post = gap[gap.index >= pd.Timestamp(cutoff)].dropna()
        val = float(post.mean())
    else:
        raise ValueError("statistic must be 'ratio', 'post_rmspe' or 'post_gap'")
    return val, pre_r, post_r


def inference_table(treated_results, placebo_results, units=None,
                    statistic="ratio", pre_end="2014-12-01",
                    cutoff="2015-04-01") -> pd.DataFrame:
    """Per-treated-unit permutation p-value against the control placebos.

    statistic:
        'ratio'      post/pre RMSPE ratio (Abadie's default). Distorted when
                     some placebos fit the pre-period almost perfectly: a tiny
                     denominator inflates their ratio and swamps the treated.
        'post_rmspe' post-period RMSPE (magnitude of deviation).
        'post_gap'   signed mean post-period gap (effect size, ~%). The fair
                     choice when all units fit the pre-period comparably well,
                     since it needs no pre-period normalisation.

    One-sided p = (1 + #placebos with statistic >= treated) / (1 + #placebos).
    """
    placebo_vals = np.array([_statistic(p["gap"], statistic, pre_end, cutoff)[0]
                             for p in placebo_results.values()])
    n_p = len(placebo_vals)
    keys = list(treated_results.keys()) if units is None else list(units)
    rows = []
    for u in keys:
        val, pre_r, post_r = _statistic(treated_results[u]["gap"], statistic,
                                        pre_end, cutoff)
        p = (1 + int(np.sum(placebo_vals >= val))) / (1 + n_p)
        rows.append({"unit": u, "pre_rmspe": pre_r, "post_rmspe": post_r,
                     statistic: val, "p_value": p})
    return pd.DataFrame(rows).set_index("unit").sort_values(statistic, ascending=False)


def plot_placebo(treated_gap, placebo_results, cutoff="2015-04-01",
                 max_pre_rmspe=None, label="treated (avg)", ax=None):
    """Spaghetti plot: placebo gaps in grey, the treated effect in colour.

    `max_pre_rmspe` optionally hides placebos that fit their own pre-period
    badly (their wild post gaps would otherwise dominate the picture).
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    for p in placebo_results.values():
        if max_pre_rmspe is not None and p["pre_r"] > max_pre_rmspe:
            continue
        ax.plot(p["gap"].index, p["gap"].values, color="#C9C8C2", lw=0.8, alpha=0.7)
    ax.plot(treated_gap.index, treated_gap.values, color="#D85A30", lw=2.2, label=label)
    ax.axhline(0, color="grey", lw=1)
    ax.axvline(pd.Timestamp(cutoff), ls="--", color="grey")
    ax.set(title="Placebo inference: treated effect vs control placebos",
           xlabel="date", ylabel="log gap")
    ax.legend()
    return ax