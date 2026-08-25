"""Placebo-тесты: перетасовка доходностей (Этап 6.2).

Идея: если метод работает, то на случайных (перетасованных) доходностях
 Sharpe должен быть значительно ниже реального.
N сидов по умолчанию: 100 (из PLAN.md); для финального отчёта — 10 сидов.

Версия 2 (2026-08-25): кросс-секционный бэктест (аналогично sensitivity.py).
Старая версия вызывала backtest() с неверными kwargs (position=, cost=).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from newsalpha.sttm.pipeline import sttm_expanding
from newsalpha.backtest.portfolio import backtest, summarize


def shuffle_returns(
    returns: pd.Series,
    seed: int = 42,
    fixed_start: int | None = None,
) -> pd.Series:
    """Перетасовка доходностей с фиксированным ядром.

    fixed_start: если задан, не двигает первые fixed_start элементов
    (аналог in-sample portion).
    """
    rng = np.random.default_rng(seed)
    vals = returns.to_numpy().copy()
    if fixed_start is not None:
        head = vals[:fixed_start].copy()
        tail = vals[fixed_start:].copy()
        rng.shuffle(tail)
        return pd.Series(np.concatenate([head, tail]), index=returns.index)
    else:
        rng.shuffle(vals)
        return pd.Series(vals, index=returns.index)


def _build_index_panel(
    returns_panel: pd.DataFrame,
    theta: np.ndarray,
    c_words: np.ndarray,
    tw_lists: list,
    vocab: dict,
    weeks: list,
    gamma: float,
    prob_mass: float,
    first_test_year: int,
    initial_train_years: int,
    norm: str,
) -> pd.DataFrame:
    """Построить панель STTM-индексов [недели x тикеры] для кросс-секционного backtest."""
    index_panel = {}
    for ticker in returns_panel.columns:
        r = returns_panel[ticker].dropna()
        if len(r) < 100:
            continue
        try:
            idx = sttm_expanding(
                r, theta, c_words, tw_lists, vocab, weeks,
                gamma=gamma, prob_mass=prob_mass,
                first_test_year=first_test_year,
                initial_train_years=initial_train_years,
                norm=norm,
            )
            if len(idx) >= 20:
                index_panel[ticker] = idx
        except Exception:
            continue
    if not index_panel:
        return pd.DataFrame()
    return pd.DataFrame(index_panel)


def placebo_test(
    returns_panel: pd.DataFrame,
    theta: np.ndarray,
    c_words: np.ndarray,
    tw_lists: list,
    vocab: dict,
    weeks: list,
    n_sims: int = 100,
    gamma: float = 0.05,
    prob_mass: float = 0.3,
    first_test_year: int = 2015,
    initial_train_years: int = 2,
    top_pct: float = 0.20,
    cost: float = 0.0005,
    index_norm: str = "sigmoid",
    seed: int = 42,
) -> dict:
    """Placebo: n_sims перетасованных доходностей → распределение Sharpe.

    Кросс-секционный бэктест: для каждого тикера sttm_expanding → панель
    индексов → backtest (top-20%, равные веса) → gross/net Sharpe.
    Сравнивает реальный Sharpe с распределением placebo-Sharpe.
    z-score: (real - mean(placebo)) / std(placebo).
    """
    sig_df = _build_index_panel(
        returns_panel, theta, c_words, tw_lists, vocab, weeks,
        gamma=gamma, prob_mass=prob_mass,
        first_test_year=first_test_year,
        initial_train_years=initial_train_years,
        norm=index_norm,
    )
    if sig_df.empty:
        return {
            "real_sharpe": np.nan, "placebo_mean": np.nan, "placebo_std": np.nan,
            "placebo_median": np.nan, "placebo_95ci": (np.nan, np.nan),
            "z_score": np.nan, "p_value": np.nan, "n_sims": 0,
            "conclusion": "NO DATA — index panel empty",
        }

    ret_df = returns_panel[sig_df.columns]
    common_idx = sig_df.index.intersection(ret_df.index)
    sig_df = sig_df.loc[common_idx]
    ret_df = ret_df.loc[common_idx]

    real_bt = backtest(sig_df, ret_df, top_pct=top_pct, rates=(cost,), mode="level")
    real_sharpe = summarize(real_bt["gross"]).get("sharpe", np.nan)

    placebo_sharpes = []
    rng_seeds = np.random.default_rng(seed).integers(0, 2**31, size=n_sims)

    for i, s in enumerate(rng_seeds):
        shuffled = returns_panel.apply(lambda col: shuffle_returns(col, seed=int(s) + hash(col.name) % (2**31)))
        try:
            sig_p = _build_index_panel(
                shuffled, theta, c_words, tw_lists, vocab, weeks,
                gamma=gamma, prob_mass=prob_mass,
                first_test_year=first_test_year,
                initial_train_years=initial_train_years,
                norm=index_norm,
            )
            if sig_p.empty:
                continue
            ret_p = shuffled[sig_p.columns]
            common = sig_p.index.intersection(ret_p.index)
            sig_p = sig_p.loc[common]
            ret_p = ret_p.loc[common]
            bt = backtest(sig_p, ret_p, top_pct=top_pct, rates=(cost,), mode="level")
            sh = summarize(bt["gross"]).get("sharpe", np.nan)
            if not np.isnan(sh):
                placebo_sharpes.append(sh)
        except Exception:
            continue

    placebo_arr = np.array(placebo_sharpes)
    if len(placebo_arr) > 1 and placebo_arr.std() > 0:
        z = (real_sharpe - placebo_arr.mean()) / placebo_arr.std()
    else:
        z = np.nan

    p_value = (placebo_arr >= real_sharpe).mean() if len(placebo_arr) > 0 else np.nan

    return {
        "real_sharpe": real_sharpe,
        "placebo_mean": placebo_arr.mean() if len(placebo_arr) > 0 else np.nan,
        "placebo_std": placebo_arr.std() if len(placebo_arr) > 0 else np.nan,
        "placebo_median": np.median(placebo_arr) if len(placebo_arr) > 0 else np.nan,
        "placebo_95ci": (
            float(np.percentile(placebo_arr, 2.5)),
            float(np.percentile(placebo_arr, 97.5)),
        ) if len(placebo_arr) > 0 else (np.nan, np.nan),
        "z_score": z,
        "p_value": p_value,
        "n_sims": len(placebo_arr),
        "conclusion": (
            "SIGNIFICANT" if (not np.isnan(z) and abs(z) > 2.0)
            else "NOT SIGNIFICANT"
        ),
    }
