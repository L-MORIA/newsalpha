"""Рыночные данные: недельные доходности MOEX (Этап 1c/4-пререквизит).

Формула статьи (config: weekly_return=fri_close_over_mon_open):
r_week = (close_последнего_торгового_дня − open_первого_торгового_дня) / open.
Календарь W-FRI; в укороченных неделях берутся фактические первый/последний дни.
"""
from pathlib import Path

import pandas as pd


def load_ticker_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["TRADEDATE", "OPEN", "CLOSE"], parse_dates=["TRADEDATE"])
    return df.sort_values("TRADEDATE").dropna(subset=["OPEN", "CLOSE"]).reset_index(drop=True)


def weekly_returns(df: pd.DataFrame, date_col: str = "TRADEDATE") -> pd.Series:
    """Дневные OHLC → серия недельных доходностей (индекс — конец недели W-FRI)."""
    g = df.groupby(pd.Grouper(key=date_col, freq="W-FRI"))
    first_open = g["OPEN"].first()
    last_close = g["CLOSE"].last()
    return (last_close - first_open) / first_open


def load_all_tickers(
    prices_dir: str | Path, tickers: list[str], board: str = "TQBR"
) -> pd.DataFrame:
    """Все тикеры → широкая матрица [недели × тикеры] доходностей."""
    out = {}
    for t in tickers:
        path = Path(prices_dir) / f"shares_{board}_{t}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        out[t] = weekly_returns(load_ticker_csv(path))
    return pd.DataFrame(out)
