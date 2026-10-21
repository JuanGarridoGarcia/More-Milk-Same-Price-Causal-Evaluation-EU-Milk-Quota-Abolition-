"""Tests for treatment assignment and the EU-28 analysis universe."""

import pandas as pd
import pytest

import treatment
from treatment import (
    ABOLITION_DATE,
    ANALYSIS_COUNTRIES,
    BINDING_COUNTRIES,
    CONTROL_COUNTRIES,
    assign_treatment,
)


def _panel(geos, dates):
    return pd.DataFrame(
        [(g, pd.Timestamp(d)) for g in geos for d in dates], columns=["geo", "date"]
    )


def test_universe_is_eu28_split_12_16():
    assert len(BINDING_COUNTRIES) == 12
    assert len(CONTROL_COUNTRIES) == 16
    assert BINDING_COUNTRIES.isdisjoint(CONTROL_COUNTRIES)
    assert len(ANALYSIS_COUNTRIES) == 28


def test_eurostat_geo_codes_not_iso():
    # Eurostat uses EL for Greece and UK for the United Kingdom.
    assert "EL" in CONTROL_COUNTRIES and "GR" not in ANALYSIS_COUNTRIES
    assert "UK" in CONTROL_COUNTRIES and "GB" not in ANALYSIS_COUNTRIES


def test_flags_and_restriction():
    df = _panel(sorted(BINDING_COUNTRIES) + ["FR", "AL"], ["2014-12-01", "2015-04-01"])
    out = assign_treatment(df)

    assert "AL" not in set(out["geo"])  # third country dropped
    de = out[out["geo"] == "DE"].set_index("date")
    fr = out[out["geo"] == "FR"].set_index("date")
    assert de["treated"].tolist() == [1, 1]
    assert fr["treated"].tolist() == [0, 0]
    # The abolition month itself counts as post.
    assert de.loc[ABOLITION_DATE, "post"] == 1
    assert de.loc[pd.Timestamp("2014-12-01"), "post"] == 0
    assert (out["treated_post"] == out["treated"] * out["post"]).all()


def test_missing_binding_country_fails_loud():
    df = _panel(["DE", "FR"], ["2015-01-01"])
    with pytest.raises(ValueError, match="Binding countries absent"):
        assign_treatment(df)


def test_input_not_mutated():
    df = _panel(sorted(BINDING_COUNTRIES), ["2015-01-01"])
    before = df.copy()
    assign_treatment(df)
    pd.testing.assert_frame_equal(df, before)


def test_single_source_of_truth_for_price_ingest():
    import ingest_price

    assert ingest_price.BINDING_COUNTRIES is treatment.BINDING_COUNTRIES
    assert ingest_price.ABOLITION_DATE == treatment.ABOLITION_DATE
