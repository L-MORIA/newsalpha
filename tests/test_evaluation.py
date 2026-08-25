"""Тесты модуля evaluation."""
import numpy as np
import pandas as pd
import pytest

from newsalpha.evaluation.granger import adf_test, granger_causality, evaluate_granger
from newsalpha.evaluation.direction import direction_metrics, direction_table
from newsalpha.evaluation.fdr import benjamini_hochberg, per_ticker_spearman
from newsalpha.evaluation.placebo import shuffle_returns, _build_index_panel, placebo_test
from newsalpha.evaluation.sensitivity import _nan_row, plateau_diagnosis


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


# ---------------------------------------------------------------------------
# placebo.py tests
# ---------------------------------------------------------------------------

def test_shuffle_returns_deterministic():
    rng = np.random.default_rng(99)
    s = pd.Series(rng.normal(0, 1, 50))
    s1 = shuffle_returns(s, seed=42)
    s2 = shuffle_returns(s, seed=42)
    pd.testing.assert_series_equal(s1, s2)


def test_shuffle_returns_preserves_values():
    rng = np.random.default_rng(99)
    s = pd.Series(rng.normal(0, 1, 50))
    shuffled = shuffle_returns(s, seed=42)
    assert sorted(shuffled.tolist()) == sorted(s.tolist())


def test_shuffle_returns_fixed_start():
    rng = np.random.default_rng(99)
    s = pd.Series(rng.normal(0, 1, 50))
    shuffled = shuffle_returns(s, seed=42, fixed_start=20)
    pd.testing.assert_series_equal(s.iloc[:20], shuffled.iloc[:20])
    assert sorted(shuffled.iloc[20:].tolist()) == sorted(s.iloc[20:].tolist())


def test_build_index_panel_basic():
    rng = np.random.default_rng(42)
    n, nt = 300, 5
    weeks = pd.date_range("2015-01-02", periods=n, freq="W-FRI")
    ret = pd.DataFrame(rng.normal(0.005, 0.03, (n, nt)),
                       index=weeks, columns=[f"T{i}" for i in range(nt)])
    theta = rng.dirichlet(np.ones(3), size=n).T
    c_words = np.abs(rng.normal(10, 3, (5, n)))
    tw = [[("a", 0.4)], [("b", 0.5)], [("c", 0.3)]]
    vocab = {"a": 0, "b": 1, "c": 2}
    idx = _build_index_panel(ret, theta, c_words, tw, vocab,
                             list(weeks), 0.05, 0.3, 2017, 2, "sigmoid")
    assert idx.shape[0] > 0
    assert idx.shape[1] <= nt


def test_build_index_panel_too_few_weeks():
    rng = np.random.default_rng(42)
    n, nt = 10, 3
    weeks = pd.date_range("2020-01-03", periods=n, freq="W-FRI")
    ret = pd.DataFrame(rng.normal(0, 0.01, (n, nt)),
                       index=weeks, columns=[f"T{i}" for i in range(nt)])
    theta = rng.dirichlet(np.ones(2), size=n).T
    c_words = np.abs(rng.normal(10, 3, (3, n)))
    tw = [[("a", 0.5)], [("b", 0.5)]]
    vocab = {"a": 0, "b": 1}
    idx = _build_index_panel(ret, theta, c_words, tw, vocab,
                             list(weeks), 0.05, 0.3, 2020, 2, "sigmoid")
    assert idx.shape == (0, 0)


def test_placebo_test_smoke():
    rng = np.random.default_rng(42)
    n, nt = 300, 8
    weeks = pd.date_range("2015-01-02", periods=n, freq="W-FRI")
    ret = pd.DataFrame(rng.normal(0.005, 0.03, (n, nt)),
                       index=weeks, columns=[f"T{i}" for i in range(nt)])
    theta = rng.dirichlet(np.ones(3), size=n).T
    c_words = np.abs(rng.normal(10, 3, (5, n)))
    tw = [[("a", 0.4)], [("b", 0.5)], [("c", 0.3)]]
    vocab = {"a": 0, "b": 1, "c": 2}
    result = placebo_test(ret, theta, c_words, tw, vocab,
                          list(weeks), n_sims=3, first_test_year=2017)
    assert "real_sharpe" in result
    assert "z_score" in result
    assert "conclusion" in result
    assert result["conclusion"] in ("SIGNIFICANT", "NOT SIGNIFICANT", "NO DATA")


def test_placebo_test_empty_panel():
    rng = np.random.default_rng(42)
    n = 10
    weeks = pd.date_range("2020-01-03", periods=n, freq="W-FRI")
    ret = pd.DataFrame(rng.normal(0, 0.01, (n, 1)),
                       index=weeks, columns=["T0"])
    theta = rng.dirichlet(np.ones(2), size=n).T
    c_words = np.abs(rng.normal(10, 3, (3, n)))
    tw = [[("a", 0.5)], [("b", 0.5)]]
    vocab = {"a": 0, "b": 1}
    result = placebo_test(ret, theta, c_words, tw, vocab,
                          list(weeks), n_sims=2, first_test_year=2020)
    assert "NO DATA" in result["conclusion"]
    assert np.isnan(result["z_score"])


# ---------------------------------------------------------------------------
# sensitivity.py tests
# ---------------------------------------------------------------------------

def test_nan_row():
    row = _nan_row(0.05, 0.3)
    assert row["gamma"] == 0.05
    assert row["prob_mass"] == 0.3
    assert np.isnan(row["gross_sharpe"])
    assert row["n_tickers"] == 0


def test_plateau_diagnosis_plat():
    grid = pd.DataFrame({
        "gross_sharpe": [1.2, 1.1, 1.0, 0.9, 0.8],
    })
    d = plateau_diagnosis(grid)
    assert d["verdict"] == "PLATEAU"


def test_plateau_diagnosis_peak():
    grid = pd.DataFrame({
        "gross_sharpe": [1.5, 0.3, 0.2, 0.1, 0.05],
    })
    d = plateau_diagnosis(grid)
    assert d["verdict"] == "PEAK (fragile)"


def test_plateau_diagnosis_empty():
    grid = pd.DataFrame({"gross_sharpe": []})
    d = plateau_diagnosis(grid)
    assert d["verdict"] == "NO DATA"
