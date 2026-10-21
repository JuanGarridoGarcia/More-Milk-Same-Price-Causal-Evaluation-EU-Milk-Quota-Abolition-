"""Tests for the event-study DiD (skipped when pyfixest is not installed)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyfixest")

import estimation  # noqa: E402


def _panel(effect=0.08, seed=0):
    """Balanced panel: 6 treated and 6 control units, monthly 2010-2019.

    Treated units get a log effect of `effect` from 2015 on; no pre-trend.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", "2019-12-01", freq="MS")
    month_shock = rng.normal(0, 0.05, len(dates))
    rows = []
    for u in range(12):
        treated = int(u < 6)
        level = rng.normal(6, 0.5)
        for t, d in enumerate(dates):
            y = level + month_shock[t] + rng.normal(0, 0.01)
            if treated and d.year >= 2015:
                y += effect
            rows.append({"geo": f"U{u}", "date": d, "treated": treated,
                         "milk_deliveries": np.exp(y)})
    return pd.DataFrame(rows)


def test_event_study_recovers_effect_and_flat_pre_trend():
    model = estimation.run_event_study(_panel(), ref_year=2014)
    coefs = estimation.event_study_coefs(model).set_index("year")["Estimate"]

    assert 2014 not in coefs.index  # reference year omitted
    assert coefs.loc[2010:2013].abs().max() < 0.02
    assert coefs.loc[2015:2019].mean() == pytest.approx(0.08, abs=0.02)


def test_non_positive_outcome_rejected():
    df = _panel()
    df.loc[0, "milk_deliveries"] = 0.0
    with pytest.raises(ValueError, match="zero/negative"):
        estimation.run_event_study(df)
