"""Tests for the synthetic control estimators and placebo inference."""

import numpy as np
import pandas as pd
import pytest

import synth

DATES = pd.date_range("2010-01-01", "2019-12-01", freq="MS")
PRE_END = "2014-12-01"
CUTOFF = "2015-04-01"


def _panel(effect=0.0, seed=0):
    """Panel with 5 controls and 1 treated unit built as a known convex mix.

    The treated unit is 0.6*C1 + 0.4*C2 in logs, plus `effect` from the cutoff.
    """
    rng = np.random.default_rng(seed)
    n = len(DATES)
    logs = {f"C{i}": 5 + 0.002 * i * np.arange(n) + 0.05 * rng.standard_normal(n).cumsum()
            for i in range(1, 6)}
    treated = 0.6 * logs["C1"] + 0.4 * logs["C2"]
    treated = treated + effect * (DATES >= pd.Timestamp(CUTOFF))
    logs["T1"] = treated

    rows = []
    for geo, y in logs.items():
        for d, v in zip(DATES, np.exp(y), strict=True):
            rows.append({"geo": geo, "date": d, "milk_deliveries": v,
                         "treated": int(geo == "T1")})
    return pd.DataFrame(rows)


def test_standard_scm_recovers_convex_weights():
    results, gaps, _ = synth.run_all_treated(_panel(), pre_end=PRE_END)
    w = results["T1"]["weights"]
    assert w["C1"] == pytest.approx(0.6, abs=1e-3)
    assert w["C2"] == pytest.approx(0.4, abs=1e-3)
    assert (w >= 0).all() and w.sum() == pytest.approx(1.0)
    assert results["T1"]["pre_rmspe"] < 1e-4
    assert gaps["T1"].abs().max() < 1e-3


def test_standard_scm_recovers_known_effect():
    _, gaps, _ = synth.run_all_treated(_panel(effect=0.10), pre_end=PRE_END)
    post = synth.post_effects(gaps, cutoff=CUTOFF)
    assert post["T1"] == pytest.approx(0.10, abs=1e-3)


def test_ridge_weights_sum_to_one_and_can_be_negative():
    rng = np.random.default_rng(1)
    Y0 = rng.standard_normal((60, 4))
    y1 = Y0 @ np.array([1.5, -0.5, 0.0, 0.0])  # outside the convex hull
    w = synth._ridge_affine(y1, Y0, lam=1e-6)
    assert w.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(w, [1.5, -0.5, 0.0, 0.0], atol=1e-4)


def test_ridge_penalty_shrinks_weights():
    rng = np.random.default_rng(2)
    Y0 = rng.standard_normal((60, 4))
    y1 = Y0 @ np.array([1.5, -0.5, 0.0, 0.0])
    w_small = synth._ridge_affine(y1, Y0, lam=1e-6)
    w_big = synth._ridge_affine(y1, Y0, lam=1e6)
    assert np.linalg.norm(w_big) < np.linalg.norm(w_small)
    # A huge penalty pushes towards equal weights (the minimum-norm point with sum 1).
    np.testing.assert_allclose(w_big, np.full(4, 0.25), atol=1e-3)


def test_missing_treated_unit_fails_loud():
    df = _panel()
    df.loc[df["geo"] == "T1", "milk_deliveries"] = np.nan
    with pytest.raises(ValueError, match="treated unit"):
        synth.run_all_treated(df)


def test_placebo_p_value_formula():
    # An effect far larger than any placebo gap the random-walk controls produce.
    df = _panel(effect=2.0)
    results, _, _ = synth.run_all_treated(df, pre_end=PRE_END)
    placebos = synth.placebo_gaps(df, pre_end=PRE_END, cutoff=CUTOFF, augmented=False)
    table = synth.inference_table(results, placebos, statistic="post_gap",
                                  pre_end=PRE_END, cutoff=CUTOFF)
    # 5 placebos; a large effect beats all of them: p = 1 / (1 + 5).
    assert len(placebos) == 5
    assert table.loc["T1", "p_value"] == pytest.approx(1 / 6)


def test_unknown_statistic_rejected():
    df = _panel()
    results, _, _ = synth.run_all_treated(df, pre_end=PRE_END)
    with pytest.raises(ValueError):
        synth.inference_table(results, {}, statistic="median")


def _placebo_results(n=15, sd=0.02, seed=0):
    """Fake placebo_gaps output: zero-mean gap series for n controls."""
    rng = np.random.default_rng(seed)
    return {f"C{i}": {"gap": pd.Series(rng.normal(0, sd, len(DATES)), index=DATES)}
            for i in range(n)}


def test_group_permutation_detects_a_shifted_treated_group():
    placebos = _placebo_results()
    treated = np.full(11, 0.05)
    res = synth.group_permutation_test(treated, placebos, cutoff=CUTOFF, n_perm=2_000)
    assert res["treated_mean"] == pytest.approx(0.05)
    assert res["observed"] == pytest.approx(res["treated_mean"] - res["control_mean"])
    assert res["p_value"] < 0.01


def test_group_permutation_is_not_significant_under_the_null():
    placebos = _placebo_results(seed=1)
    cutoff = pd.Timestamp(CUTOFF)
    control_pg = [p["gap"][p["gap"].index >= cutoff].mean() for p in placebos.values()]
    # Treated effects drawn from the same distribution as the placebo effects.
    treated = np.random.default_rng(2).normal(np.mean(control_pg), np.std(control_pg), 11)
    res = synth.group_permutation_test(treated, placebos, cutoff=CUTOFF, n_perm=2_000)
    assert 0.05 < res["p_value"] <= 1


def test_group_permutation_is_reproducible_with_a_seed():
    placebos = _placebo_results()
    treated = np.linspace(0.0, 0.02, 11)
    a = synth.group_permutation_test(treated, placebos, cutoff=CUTOFF, n_perm=500, seed=3)
    b = synth.group_permutation_test(treated, placebos, cutoff=CUTOFF, n_perm=500, seed=3)
    assert a == b
