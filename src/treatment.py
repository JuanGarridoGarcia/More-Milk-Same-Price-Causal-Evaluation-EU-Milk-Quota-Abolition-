"""
treatment.py — Treatment, control universe and period assignment for the
milk-quota difference-in-differences.

    treated : 1 if the Member State had a *binding* milk quota at abolition,
              0 otherwise. Property of the COUNTRY, constant over time.
    post    : 1 for observations from the abolition date (April 2015) onward.

The analysis universe is EU-28 (the UK was still a member in 2015): 12 binding
(treated) + 16 non-binding (control). Countries outside this set — EU candidate
and third countries that Eurostat also reports (Albania, Norway, Switzerland,
Serbia, Turkey, ...) — are NOT valid counterfactuals and are dropped.

On the definition of `treated`
------------------------------
A quota is *binding* when it actively constrains production: the country
delivers up to (or beyond, paying the superlevy) its national quota, so its
quota shadow value is > 0. The clean measure is the pre-abolition quota fill
rate; the documented list of the 12 binding Member States below was derived
from exactly that criterion. The fill-rate reconstruction is kept as a
robustness check for when/if the historical quota series is located.
"""

from __future__ import annotations

import pandas as pd

# Quota abolition: 1 April 2015 (end of the 2014/15 marketing year).
ABOLITION_DATE = pd.Timestamp("2015-04-01")

# 12 Member States with a binding quota at abolition (Eurostat geo codes).
BINDING_COUNTRIES = frozenset({
    "DE", "IE", "NL", "DK", "AT", "BE", "IT", "PL", "ES", "CY", "LU", "EE",
})

# 16 non-binding EU-28 Member States — the control / donor pool.
# Note the Eurostat quirks: Greece is 'EL', the UK is 'UK'.
# Watch list for later trimming: HR (joined the EU only in 2013, short
# pre-period) and MT (negligible industrial collection).
CONTROL_COUNTRIES = frozenset({
    "BG", "HR", "CZ", "FI", "FR", "EL", "HU", "LV",
    "LT", "MT", "PT", "RO", "SK", "SI", "SE", "UK",
})

# Full analysis universe (EU-28).
ANALYSIS_COUNTRIES = BINDING_COUNTRIES | CONTROL_COUNTRIES


def assign_treatment(
    df: pd.DataFrame,
    country_col: str = "geo",
    date_col: str = "date",
    abolition_date: pd.Timestamp = ABOLITION_DATE,
    binding: frozenset[str] = BINDING_COUNTRIES,
    universe: frozenset[str] = ANALYSIS_COUNTRIES,
    restrict: bool = True,
) -> pd.DataFrame:
    """Restrict to the EU-28 universe and add `treated`, `post`, `treated_post`.

    With `restrict=True` (default), rows for countries outside `universe`
    (non-EU candidates / third countries) are dropped. Returns a copy with
    three int columns; `treated` is country-level, `post` is time-level, and
    their product carries the DiD estimate under two-way fixed effects.
    """
    if country_col not in df.columns:
        raise KeyError(f"'{country_col}' not in dataframe columns.")
    if date_col not in df.columns:
        raise KeyError(f"'{date_col}' not in dataframe columns.")

    out = df.copy()
    if restrict:
        out = out[out[country_col].isin(universe)].copy()

    # Fail loud if any binding country is missing (catches geo-code surprises:
    # Greece is 'EL', the UK is 'UK', not the ISO 'GR'/'GB').
    present = set(out[country_col].unique())
    missing = set(binding) - present
    if missing:
        raise ValueError(
            f"Binding countries absent from the panel: {sorted(missing)}. "
            "Check the geo codes (Greece is 'EL', UK is 'UK' in Eurostat)."
        )

    out["treated"] = out[country_col].isin(binding).astype("int8")
    out["post"] = (out[date_col] >= abolition_date).astype("int8")
    out["treated_post"] = (out["treated"] * out["post"]).astype("int8")
    return out


if __name__ == "__main__":
    demo = pd.DataFrame({
        "geo": ["DE", "DE", "FR", "FR", "AL", "AL"],   # AL should be dropped
        "date": pd.to_datetime(["2014-01-01", "2015-06-01"] * 3),
        "milk_deliveries": [2650, 2810, 2050, 2080, 0, 0],
    })
    print(assign_treatment(demo))