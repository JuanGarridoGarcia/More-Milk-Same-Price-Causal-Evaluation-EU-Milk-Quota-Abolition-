"""
ingest_price.py — Load EU Milk Market Observatory farm-gate raw milk prices
and shape them into a geo x month panel for the price analysis.

Source: agridata.ec.europa.eu (Milk Market Observatory) "Raw milk prices",
monthly, EUR/100 kg, Member State notifications via ISAMM.

The MMO export codes Member States as EU protocol-order numbers — Belgium=10,
Bulgaria=20, ..., with Croatia inserted as 105 (it joined the EU in 2013, so it
was slotted between France=100 and Italy=110 without renumbering), ...,
Sweden=260. We map those to Eurostat geo codes so the price panel aligns with
the deliveries panel.

`build_price_panel` returns a self-contained DiD panel (geo, date, milk_price +
treatment/period flags) built entirely from the price data — the price analysis
is independent of the deliveries sample. `merge_price` is kept only for the
income finale, where price and deliveries are joined country by country.

IMPORTANT: the price export is EU-27 — the United Kingdom is absent. UK is a
control in the deliveries analysis but cannot be one for the price outcome.
"""

from __future__ import annotations

import pandas as pd

# MMO numeric Member-State codes (EU protocol order) -> Eurostat geo codes.
CODE2GEO = {
    10: "BE", 20: "BG", 30: "CZ", 40: "DK", 50: "DE", 60: "EE", 70: "IE",
    80: "EL", 90: "ES", 100: "FR", 105: "HR", 110: "IT", 120: "CY", 130: "LV",
    140: "LT", 150: "LU", 160: "HU", 170: "MT", 180: "NL", 190: "AT", 200: "PL",
    210: "PT", 220: "RO", 230: "SI", 240: "SK", 250: "FI", 260: "SE",
}

# The only thing carried over from the quantity project: the 12 Member States
# with a binding quota (positive shadow price). Treatment is a property of the
# policy, identical across outcomes, so the list is reused verbatim. Defined here
# to keep the price ingest self-contained.
BINDING_COUNTRIES = ["AT", "BE", "CY", "DE", "DK", "EE",
                     "ES", "IE", "IT", "LU", "NL", "PL"]
ABOLITION_DATE = pd.Timestamp("2015-04-01")   # quota gone from 1 Apr 2015


def load_price(path, code_col="Member State", year_col="Year",
               month_col="Month", price_col="Price(€/100kg)") -> pd.DataFrame:
    """Load the MMO Excel into a long ['geo', 'date', 'milk_price'] panel."""
    raw = pd.read_excel(path, header=0)
    raw.columns = [str(c).strip() for c in raw.columns]

    geo = raw[code_col].map(CODE2GEO)
    unmapped = sorted(raw.loc[geo.isna(), code_col].dropna().unique().tolist())
    if unmapped:
        raise ValueError(f"Unmapped Member State codes: {unmapped}")

    out = pd.DataFrame({
        "geo": geo,
        "date": pd.to_datetime(
            raw[year_col].astype(int).astype(str) + "-" + raw[month_col].astype(str),
            format="%Y-%b"),
        "milk_price": pd.to_numeric(raw[price_col], errors="coerce"),
    })
    return (out.dropna(subset=["milk_price"])
               .sort_values(["geo", "date"]).reset_index(drop=True))


def build_price_panel(path) -> pd.DataFrame:
    """Read the MMO price Excel and return the full DiD panel.

    Output columns: geo, date, milk_price, treated, post, treated_post (one row
    per country-month). Reuses `load_price` for the Excel parsing and the
    CODE2GEO mapping, then stamps the treatment/period flags — so the price panel
    is built from the price data alone, not from the deliveries sample.

    treated : 1 if the country had a binding quota (BINDING_COUNTRIES), else 0
              — constant over time, a property of the country.
    post    : 1 from the April-2015 abolition onward, else 0.

    Fails loud if any binding country is absent from the price data: the EL/UK
    mapping trap would otherwise corrupt the treated/control split silently.
    """
    panel = load_price(path)                       # -> geo, date, milk_price
    panel["treated"] = panel["geo"].isin(BINDING_COUNTRIES).astype(int)
    panel["post"] = (panel["date"] >= ABOLITION_DATE).astype(int)
    panel["treated_post"] = panel["treated"] * panel["post"]

    missing = set(BINDING_COUNTRIES) - set(panel["geo"].unique())
    if missing:
        raise ValueError(f"binding countries absent from price data: {sorted(missing)}")

    return panel[["geo", "date", "milk_price", "treated", "post", "treated_post"]]


def merge_price(panel: pd.DataFrame, price: pd.DataFrame, how="left") -> pd.DataFrame:
    """Merge milk_price onto the deliveries panel on [geo, date].

    With how='left', UK (and any country absent from the price data) keeps its
    rows with milk_price = NaN — drop those before the price analysis. Kept for
    the income finale (price x quantity), not for building the price panel.
    """
    return panel.merge(price, on=["geo", "date"], how=how)