"""Tests for the Milk Market Observatory price ingest."""

import pandas as pd
import pytest

import ingest_price
from treatment import BINDING_COUNTRIES


def _write_mmo_excel(path, rows):
    pd.DataFrame(rows, columns=["Member State", "Year", "Month", "Price(€/100kg)"]).to_excel(
        path, index=False
    )


def _all_binding_rows():
    geo2code = {g: c for c, g in ingest_price.CODE2GEO.items()}
    rows = []
    for geo in sorted(BINDING_COUNTRIES) + ["FR"]:
        rows.append((geo2code[geo], 2015, "Mar", 30.0))
        rows.append((geo2code[geo], 2015, "Apr", 29.5))
    return rows


def test_code_mapping_covers_eu27_without_uk():
    geos = set(ingest_price.CODE2GEO.values())
    assert len(geos) == 27
    assert "UK" not in geos
    assert ingest_price.CODE2GEO[105] == "HR"  # Croatia slotted in after France
    assert ingest_price.CODE2GEO[80] == "EL"


def test_build_price_panel(tmp_path):
    path = tmp_path / "prices.xlsx"
    _write_mmo_excel(path, _all_binding_rows())

    panel = ingest_price.build_price_panel(path)

    assert list(panel.columns) == ["geo", "date", "milk_price", "treated", "post", "treated_post"]
    assert panel.loc[panel["geo"] == "FR", "treated"].eq(0).all()
    assert panel.loc[panel["geo"] == "DE", "treated"].eq(1).all()
    apr = panel[panel["date"] == pd.Timestamp("2015-04-01")]
    assert apr["post"].eq(1).all()


def test_unmapped_code_fails_loud(tmp_path):
    path = tmp_path / "prices.xlsx"
    _write_mmo_excel(path, _all_binding_rows() + [(999, 2015, "Mar", 30.0)])
    with pytest.raises(ValueError, match="Unmapped Member State codes"):
        ingest_price.load_price(path)


def test_missing_price_rows_are_dropped(tmp_path):
    path = tmp_path / "prices.xlsx"
    rows = _all_binding_rows() + [(100, 2015, "May", None)]  # FR, no price
    _write_mmo_excel(path, rows)
    out = ingest_price.load_price(path)
    assert out["milk_price"].notna().all()
    assert not ((out["geo"] == "FR") & (out["date"] == pd.Timestamp("2015-05-01"))).any()
