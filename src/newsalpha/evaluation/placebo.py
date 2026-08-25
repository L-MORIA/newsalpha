"""Placebo-тесты: перетасовка доходностей (Этап 6.2).

Идея: если метод работает, то на случайных (перетасованных) доходностях
 Sharpe должен быть значительно ниже реального.
N сидов по умолчанию: 100 (из PLAN.md); для финального отчёта — 10 сидов.
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


def placebo_test(
    returns: pd.Series,
    theta: np.ndarray,
    c_words: np.ndarray,
    tw_lists: list,
    vocab: dict,
    weeks: list,
    n_sims: int = 100,
    gamma: float = 0.05,
    prob_mass: float = 0.3,
    first_test_year: int = 2015,
    cost: float = 0.0005,
    seed: int = 42,
) -> dict:
    """Placebo: n_sims перетасованных доходностей → распределение Sharpe.

    Сравнивает реальный Sharpe с распределением placebo-Sharpe.
    z-score: (real - mean(placebo)) / std(placebo).
    """
    real_idx = sttm_expanding(
        returns, theta, c_words, tw_lists, vocab, weeks,
        gamma=gamma, prob_mass=prob_mass, first_test_year=first_test_year,
    )

    real_bt = backtest(real_idx, returns, position="long_only", cost=cost)
    real_sharpe = summarize(real_bt)["sharpe"]

    placebo_sharpes = []
    rng_seeds = np.random.default_rng(seed).integers(0, 2**31, size=n_sims)

    for i, s in enumerate(rng_seeds):
        shuffled = shuffle_returns(returns, seed=int(s))
        try:
            idx_p = sttm_expanding(
                shuffled, theta, c_words, tw_lists, vocab, weeks,
                gamma=gamma, prob_mass=prob_mass, first_test_year=first_test_year,
            )
            bt = backtest(idx_p, shuffled, position="long_only", cost=cost)
            sh = summarize(bt)["sharpe"]
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
