"""Тесты модуля evaluation."""
import numpy as np
import pandas as pd
import pytest

from newsalpha.evaluation.granger import adf_test, granger_causality, evaluate_granger
from newsalpha.evaluation.direction import direction_metrics, direction_table
from newsalpha.evaluation.fdr import benjamini_hochberg, per_ticker_spearman


@pytest.fixture
def sample_series():
    rng = np.random.default_rng(42)
    n = 200
    idx = pd.Series(rng.normal(0.5, 0.1, n),
                    index=pd.date_range("2015-01-02", periods=n, freq="W-FRI"),
                    name="index")
    ret = pd.Series(rng.normal(0, 0.02, n),
                    index=idx.index, name="returns")
    return idx, ret


def test_adf_stationary(sample_series):
    idx, _ = sample_series
    r = adf_test(idx, "test")
    assert "pvalue" in r
    assert "stationary" in r


def test_adf_too_short():
    s = pd.Series([1, 2, 3])
    r = adf_test(s, "short")
    assert r["pvalue"] is np.nan


def test_granger_basic(sample_series):
    idx, ret = sample_series
    r = granger_causality(idx, ret, maxlag=3)
    assert r["maxlag"] == 3
    assert len(r["results"]) == 3
    for lag, v in r["results"].items():
        assert 0 <= v["pvalue"] <= 1 or np.isnan(v["pvalue"])


def test_evaluate_granger(sample_series):
    idx, ret = sample_series
    r = evaluate_granger(idx, ret, maxlag=3)
    assert "adf_index" in r
    assert "granger" in r
    assert r["conclusion"] in ("GRANGER-CAUSAL", "NO EVIDENCE")


def test_direction_metrics(sample_series):
    idx, ret = sample_series
    m = direction_metrics(idx, ret, horizon=1)
    assert 0 <= m["accuracy"] <= 1
    assert 0 <= m["f1"] <= 1 or np.isnan(m["f1"])
    assert -1 <= m["spearman_rho"] <= 1


def test_direction_table(sample_series):
    idx, ret = sample_series
    t = direction_table(idx, ret, horizons=[1, 2])
    assert len(t) == 2
    assert list(t["horizon"]) == [1, 2]


def test_direction_short():
    idx = pd.Series([0.5] * 5)
    ret = pd.Series([0.01] * 5)
    m = direction_metrics(idx, ret)
    assert np.isnan(m["accuracy"])


def test_bh_basic():
    pvals = np.array([0.001, 0.01, 0.03, 0.5, 0.9])
    r = benjamini_hochberg(pvals, alpha=0.05)
    assert r["n_total"] == 5
    assert r["n_significant"] >= 0
    assert r["n_significant"] <= 5


def test_bh_all_nan():
    pvals = np.array([np.nan, np.nan])
    r = benjamini_hochberg(pvals)
    assert r["n_significant"] == 0
    assert r["n_total"] == 0


def test_per_ticker_spearman():
    rng = np.random.default_rng(42)
    n, nt = 100, 3
    idx = pd.DataFrame(rng.normal(0.5, 0.1, (n, nt)),
                       columns=["A", "B", "C"])
    ret = pd.DataFrame(rng.normal(0, 0.02, (n, nt)),
                       columns=["A", "B", "C"])
    df = per_ticker_spearman(idx, ret)
    assert len(df) == 3
    assert all(0 <= p <= 1 for p in df["pvalue"].dropna())
