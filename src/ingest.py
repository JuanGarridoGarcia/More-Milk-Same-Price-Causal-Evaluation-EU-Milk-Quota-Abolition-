"""
ingest.py — Pull and tidy Eurostat dairy data for the milk-quota DiD panel.

Fetches monthly cows' milk collected by dairies and reshapes it into a long
country x month panel with a `milk_deliveries` column, which
`treatment.assign_treatment` then annotates with `treated` / `post`.

Data source
-----------
Eurostat 'apro_mk_colm' — Cows' milk collection and products obtained,
monthly data. The dataset carries several products in a product dimension
(commonly 'dairyprod'); we keep the raw collected cows' milk element. The
exact dimension name, element code and unit MUST be confirmed against the
live codelist before filtering — Eurostat renames codes occasionally — which
is what `inspect_codes` is for. That confirmation happens in 01_data.ipynb.

Requires the Eurostat client:  pip install eurostat
"""

from __future__ import annotations

import re

import pandas as pd

try:
    import eurostat
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install the Eurostat client: pip install eurostat") from exc


# Matches Eurostat period labels: '2010-01', '2010M01', '2010', '2010-Q1'.
_PERIOD_RE = re.compile(r"^\d{4}([-M]?(\d{2}|Q\d))?$")

# Product range: raw cow’s milk collected by dairy processors.
MILK_DELIVERIES = {"dairyprod": "D1110D", "milkitem": "PRD", "unit": "THS_T"}

def fetch_long(code: str = "apro_mk_colm") -> pd.DataFrame:
    """Download a Eurostat dataset and return it tidy (long) format.

    One row per (dimension combination, period) with columns for each
    dimension plus 'geo', 'time' (raw label), 'date' (datetime) and 'value'.
    """
    raw = eurostat.get_data_df(code)

    period_cols = [c for c in raw.columns if _PERIOD_RE.match(str(c))]
    id_cols = [c for c in raw.columns if c not in period_cols]

    # The geo dimension column is often named like 'geo\\TIME_PERIOD'.
    geo_col = next((c for c in id_cols if str(c).lower().startswith("geo")), None)

    long = raw.melt(
        id_vars=id_cols, value_vars=period_cols, var_name="time", value_name="value"
    )
    if geo_col and geo_col != "geo":
        long = long.rename(columns={geo_col: "geo"})

    long["date"] = _parse_period(long["time"])
    return long


def _parse_period(s: pd.Series) -> pd.Series:
    """Parse monthly Eurostat period labels ('2010M01' or '2010-01') to dates."""
    cleaned = s.astype(str).str.replace("M", "-", regex=False)
    return pd.to_datetime(cleaned, format="%Y-%m", errors="coerce")


def inspect_codes(long: pd.DataFrame, dim: str) -> list:
    """List the distinct codes present in a dimension — use this in the
    notebook to confirm the right product element and unit before filtering.
    """
    if dim not in long.columns:
        raise KeyError(f"'{dim}' not a column. Available: {list(long.columns)}")
    return sorted(long[dim].dropna().astype(str).unique())


def build_panel(long, filters, value="value"):
    """Filtra `long` a una única serie y pasa a panel geo x mes.

    filters: dict {dimensión: código}, p.ej.
        {"dairyprod": "D1110D", "unit": "THS_T"}
    Hay que fijar TODA dimensión que no sea geo/time; si no, se solapan
    varias series.
    """
    df = long.copy()
    for dim, code in filters.items():
        df = df[df[dim].astype(str) == str(code)]

    df = df[~df["geo"].astype(str).str.startswith(("EU", "EA"))]
    df = df.dropna(subset=[value, "date"])

    # Salvaguarda: tras filtrar, como mucho una fila por país-mes.
    dup = df.groupby(["geo", "date"]).size()
    if (dup > 1).any():
        leftover = [c for c in long.columns
                    if c not in ({"time", "date", value, "geo"} | set(filters))]
        raise ValueError(
            "Más de una fila por país-mes tras filtrar: queda una dimensión "
            f"sin fijar. Fija alguna de: {leftover}"
        )

    return (df.groupby(["geo", "date"], as_index=False)[value]
              .first()
              .rename(columns={value: "milk_deliveries"})
              .sort_values(["geo", "date"])
              .reset_index(drop=True))

if __name__ == "__main__":  # pragma: no cover
    # Needs network. In the notebook, run step by step and inspect first:
    long = fetch_long("apro_mk_colm")
    print("Columns:", list(long.columns))
    # print(inspect_codes(long, "dairyprod"))   # find the collected-milk code
    # print(inspect_codes(long, "unit"))        # find the right unit
    # panel = build_panel(long, product="<CONFIRM>", unit="<CONFIRM>")
    # print(panel.head())
