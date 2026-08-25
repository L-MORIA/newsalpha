import numpy as np
import pandas as pd
import pytest

from newsalpha.backtest.portfolio import _select, backtest, summarize


@pytest.fixture
def wide():
    idx = pd.date_range("2020-01-03", periods=6, freq="W-FRI")
    cols = [f"T{i}" for i in range(4)]
    sig = pd.DataFrame(np.arange(24, dtype=float).reshape(6, 4) % 4, index=idx,
                       columns=cols)
    ret = pd.DataFrame(0.01, index=idx, columns=cols)
    return sig, ret


def test_select_top_and_equal_weights():
    s = pd.Series({"a": 3.0, "b": 1.0, "c": 2.0, "d": np.nan})
    w = _select(s, None, top_pct=0.5, min_names=3)
    assert list(w.index) == ["a", "c"]
    assert w.sum() == pytest.approx(1.0)
    assert w["a"] == w["c"] == pytest.approx(0.5)


def test_select_none_when_universe_small():
    s = pd.Series({"a": 1.0})
    assert _select(s, None, top_pct=0.2, min_names=3) is None


def test_backtest_perfect_signal_gross(wide):
    """Сигнал = будущая доходность → портфель держит лучшую бумагу."""
    sig, ret = wide
    ret.iloc[:, :] = 0.0
    for i in range(len(sig) - 1):
        best = int(sig.iloc[i].values.argmax())
        ret.iloc[i + 1, best] = 0.02
    res = backtest(sig, ret, top_pct=0.25, min_names=3, rates=(0.0,))
    assert len(res) == len(sig) - 1
    assert (res["gross"] > 0.019).all()


def test_backtest_turnover_zero_when_same_basket():
    idx = pd.date_range("2020-01-03", periods=4, freq="W-FRI")
    sig = pd.DataFrame({"a": [9, 9, 9, 9], "b": [1, 1, 1, 1],
                        "c": [0, 0, 0, 0]}, index=idx, dtype=float)
    ret = pd.DataFrame(0.01, index=idx, columns=sig.columns)
    res = backtest(sig, ret, top_pct=1 / 3, min_names=3, rates=(0.0015,))
    # корзина не меняется → оборот только на первой неделе
    assert res["turnover"].iloc[1:].eq(0).all()
    assert res["net_0.0015"].iloc[1] == pytest.approx(res["gross"].iloc[1])


def test_backtest_cost_applied_on_change():
    idx = pd.date_range("2020-01-03", periods=4, freq="W-FRI")
    sig = pd.DataFrame({"a": [5, 5, 1, 1], "b": [1, 1, 5, 5], "c": [0, 0, 0, 0]},
                       index=idx, dtype=float)
    ret = pd.DataFrame(0.0, index=idx, columns=["a", "b", "c"])
    ret.iloc[3, ret.columns.get_loc("b")] = 0.10
    res = backtest(sig, ret, top_pct=1 / 3, min_names=3,
                   rates=(0.0015,), slippage=0.0005)
    # неделя 3: полная смена корзины a→b, Σ|Δw| = |0−1| + |0−1| = 2.0
    assert res["turnover"].iloc[2] == pytest.approx(2.0)
    expected = 0.10 - (0.0015 + 0.0005) * 2.0
    assert res["net_0.0015"].iloc[2] == pytest.approx(expected)


def test_delta_mode_skips_first_week(wide):
    sig, ret = wide
    res = backtest(sig, ret, mode="delta", min_names=3)
    # первая строка сигналов NaN → неделя t1 реализуется со второй строки
    assert res.index[0] == sig.index[2]


def test_object_dtype_with_none_treated_as_nan(wide):
    """Parquet-колонки с None (object dtype) должны вести себя как NaN."""
    sig, ret = wide
    obj = sig.astype(object).where(sig.notna(), None)
    res_obj = backtest(obj, ret, mode="delta", min_names=3)
    res_nan = backtest(sig, ret, mode="delta", min_names=3)
    pd.testing.assert_frame_equal(res_obj, res_nan)


def test_summarize_metrics():
    r = pd.Series([0.01, 0.02, -0.005, 0.03])
    m = summarize(r)
    assert m["sharpe"] == pytest.approx(
        (r.mean() * 52) / (r.std(ddof=1) * np.sqrt(52)))
    assert -1.0 < m["max_drawdown"] <= 0.0


def test_summarize_constant_series_guard():
    m = summarize(pd.Series([0.01] * 5))
    assert m == {"n": 5}
