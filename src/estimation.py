"""
estimation.py: Difference-in-differences estimation for the milk-quota study.

Event study (dynamic DiD): one treated-vs-control coefficient per year relative
to a reference year (default 2014, the last full pre-treatment year), so each
coefficient reads as the treated–control gap in that year *relative to 2014*.

Formula (pyfixest / fixest syntax):
    log_<outcome> ~ i(year, treated, ref=2014) | geo + date

Why each piece:
- log outcome  : country FE absorb the level gaps; log-coefficients read as ~%.
- geo FE       : fixed structural differences between countries.
- date (month) FE : seasonality and shocks common across countries in a given
                    month (the 2014 Russian ban, the 2015-16 price crash) to the
                    extent they hit both groups alike.
- cluster by geo : the outcome is serially correlated within a country.

The reference year only fixes the origin of the coordinate system; it asserts
nothing about the data. Anticipation (the soft landing) shows up as pre-2015
coefficients drifting away from zero, not as something the choice of base hides.

The `outcome` argument switches the analysed series ("milk_deliveries", default,
or "milk_price") without touching the design: treatment, FE and clustering are
identical, because "binding" is a property of the policy, not the outcome.

Requires:  pip install pyfixest
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import pyfixest as pf
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install pyfixest: pip install pyfixest") from exc


def _prepare(panel: pd.DataFrame, outcome: str = "milk_deliveries") -> pd.DataFrame:
    """Add the model variables (log outcome, calendar year); fail on bad logs.

    The log column is named ``log_<outcome>`` so the fitted data is
    self-documenting (and so price and deliveries never collide). Genuine
    zero/negative values raise; missing values (e.g. countries absent from the
    price series) become NaN under the log and are dropped by the estimator,
    which correctly leaves the DiD unbalanced rather than corrupting it.
    """
    out = panel.copy()
    if (out[outcome] <= 0).any():            # NaN <= 0 is False, so gaps are safe
        raise ValueError(f"zero/negative {outcome} breaks the log transform.")
    out[f"log_{outcome}"] = np.log(out[outcome])
    out["year"] = out["date"].dt.year
    return out


def run_event_study(panel: pd.DataFrame, ref_year: int = 2014,
                    cluster: str = "geo", outcome: str = "milk_deliveries"):
    """Fit the dynamic DiD (event study) and return the fitted pyfixest model."""
    data = _prepare(panel, outcome)
    return pf.feols(
        f"log_{outcome} ~ i(year, treated, ref={ref_year}) | geo + date",
        data=data,
        vcov={"CRV1": cluster},
    )


def event_study_coefs(model) -> pd.DataFrame:
    """Per-year coefficient table (with CI columns), sorted by year.

    The reference year is the omitted baseline, so it does not appear here.
    Outcome-agnostic: reads the fitted model, not the panel.
    """
    t = model.tidy().reset_index()
    coef_col = t.columns[0]
    yr = t[coef_col].astype(str).str.extract(r"(\d{4})")[0]
    t = t.assign(year=yr).loc[yr.notna()].copy()
    t["year"] = t["year"].astype(int)
    return t.sort_values("year").reset_index(drop=True)


def plot_event_study(model, ref_year: int | None = None, ax=None,
                     outcome: str = "milk_deliveries", break_year: int = 2015,
                     crash: tuple[float, float] | None = None):
    """Coefficient-and-CI plot: flat near zero pre-treatment => parallel trends.

    The dashed divider marks the treatment break (`break_year`, default 2015) and
    is independent of `ref_year`: the reference only sets where the zero baseline
    sits, not where the policy starts. `ref_year` is inferred from the omitted
    year when not given; `outcome` only sets the axis label. `crash`, if given as
    a (start, end) span in fractional years, shades an exogenous price shock
    (e.g. the 2015-16 dairy crash ~ (2015.5, 2016.7)).
    """
    import matplotlib.pyplot as plt

    t = event_study_coefs(model)

    if ref_year is None:
        yrs = set(t["year"])
        gaps = [y for y in range(min(yrs), max(yrs) + 1) if y not in yrs]
        ref_year = gaps[0] if len(gaps) == 1 else None

    ci_cols = [c for c in t.columns if str(c).endswith("%")]
    if len(ci_cols) >= 2:
        lo, hi = t[ci_cols[0]], t[ci_cols[1]]
    else:
        lo = t["Estimate"] - 1.96 * t["Std. Error"]
        hi = t["Estimate"] + 1.96 * t["Std. Error"]
    yerr = [t["Estimate"] - lo, hi - t["Estimate"]]

    label = outcome.replace("milk_", "")
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    if crash is not None:
        ax.axvspan(crash[0], crash[1], color="grey", alpha=0.12, lw=0,
                   label="2015–16 dairy crash")
    ax.errorbar(t["year"], t["Estimate"], yerr=yerr,
                fmt="o", capsize=3, color="#D85A30")
    if ref_year is not None:
        ax.plot([ref_year], [0], marker="s", color="grey")   # reference marker (=0)
    ax.axvline(break_year - 0.5, ls="--", color="grey")       # treatment break (fixed at 2015)
    ax.axhline(0, color="grey", lw=1)
    if crash is not None:
        ax.legend()
    ax.set(title=f"Event study: treated vs control by year (ref = {ref_year})",
           xlabel="year", ylabel=f"log-{label} gap vs reference")
    return ax