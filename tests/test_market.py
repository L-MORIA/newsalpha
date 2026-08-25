"""Тесты рыночного модуля: формула недельной доходности статьи."""
import numpy as np
import pandas as pd
import pytest

from newsalpha.io.market import load_all_tickers, load_ticker_csv, weekly_returns


def _df(days: dict[str, tuple[float, float]]):
    rows = [{"TRADEDATE": pd.Timestamp(d), "OPEN": o, "CLOSE": c} for d, (o, c) in days.items()]
    return pd.DataFrame(rows)


def test_full_week_fri_close_over_mon_open():
    df = _df({
        "2020-01-06": (100.0, 101.0),  # пн
        "2020-01-07": (101.5, 102.0),
        "2020-01-08": (102.0, 99.0),
        "2020-01-09": (99.5, 100.0),
        "2020-01-10": (100.0, 110.0),  # пт
    })
    r = weekly_returns(df)
    assert len(r) == 1
    assert r.iloc[0] == pytest.approx((110.0 - 100.0) / 100.0)


def test_shortened_week_uses_actual_first_last():
    # вторник и четверг: праздники в пн/пт не должны ломать расчёт
    df = _df({"2020-01-07": (50.0, 52.0), "2020-01-09": (51.0, 49.0)})
    r = weekly_returns(df)
    assert r.dropna().iloc[0] == pytest.approx((49.0 - 50.0) / 50.0)


def test_empty_week_is_nan():
    # две торгуемые недели с пустой неделей между ними
    df = _df({"2020-01-08": (100.0, 105.0), "2020-01-22": (50.0, 55.0)})
    r = weekly_returns(df)
    assert len(r) == 3
    assert np.isnan(r.iloc[1])  # неделя без сделок → NaN
    assert r.iloc[0] == pytest.approx(0.05)
    assert r.iloc[2] == pytest.approx(0.1)


def test_load_all_tickers_wide(tmp_path):
    for t in ("AAA", "BBB"):
        p = tmp_path / f"shares_TQBR_{t}.csv"
        p.write_text(
            "TRADEDATE,OPEN,CLOSE\n2020-01-06,10,11\n2020-01-10,11,12\n",
            encoding="utf-8",
        )
    wide = load_all_tickers(tmp_path, ["AAA", "BBB"])
    assert wide.shape[1] == 2
    assert wide["AAA"].dropna().iloc[0] == pytest.approx(0.2)


def test_real_sber_smoke():
    from pathlib import Path

    path = Path("data/raw/prices/shares_TQBR_SBER.csv")
    if not path.exists():
        pytest.skip("нет локальных данных SBER")
    r = weekly_returns(load_ticker_csv(path)).dropna()
    assert len(r) > 300
    assert r.abs().max() < 0.5  # нет абсурдных скачков
