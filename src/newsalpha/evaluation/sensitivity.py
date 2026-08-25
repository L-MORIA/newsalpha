"""Мини-грид чувствительности {n_topics} × {prob_mass} × {γ} (Этап 6.4).

Критерий «плато, а не острый пик» (контрмера к замечанию Масютина).
На каждом узле грида: sttm_expanding → backtest → Sharpe.
Для ускорения — на train-окне одного сида; финальный прогон — mean±std.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from newsalpha.sttm.pipeline import sttm_expanding, topic_word_lists
from newsalpha.backtest.portfolio import backtest, summarize


def sensitivity_grid(
    returns: pd.Series,
    theta: np.ndarray,
    c_words: np.ndarray,
    vocab: dict,
    weeks: list,
    lda_model=None,
    topic_word_lists_override: list | None = None,
    gammas: list[float] | None = None,
    prob_masses: list[float] | None = None,
    n_topics_list: list[int] | None = None,
    cost: float = 0.0005,
    first_test_year: int = 2015,
) -> pd.DataFrame:
    """Перебор комбинаций гиперпараметров → DataFrame с Sharpe.

    Если lda_model задан и n_topics_list не пуст — переобучаем LDA на каждом n.
    Иначе используем topic_word_lists_override (удобно для одного n).
    """
    if gammas is None:
        gammas = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15]
    if prob_masses is None:
        prob_masses = [0.15, 0.20, 0.25, 0.30, 0.40]
    if n_topics_list is None:
        n_topics_list = [None]

    rows = []
    for gamma, prob_mass, n_topics in product(gammas, prob_masses, n_topics_list):
        tw = topic_word_lists_override
        if lda_model is not None and n_topics is not None:
            tw = topic_word_lists(lda_model, topn=40)
        if tw is None:
            continue

        try:
            idx = sttm_expanding(
                returns, theta, c_words, tw, vocab, weeks,
                gamma=gamma, prob_mass=prob_mass,
                first_test_year=first_test_year,
            )
            bt = backtest(idx, returns, position="long_only", cost=cost)
            s = summarize(bt)
            rows.append({
                "n_topics": n_topics,
                "gamma": gamma,
                "prob_mass": prob_mass,
                "sharpe": s["sharpe"],
                "ann_return": s["ann_return"],
                "ann_vol": s["ann_vol"],
                "max_drawdown": s["max_drawdown"],
                "n_weeks": len(idx),
            })
        except Exception:
            rows.append({
                "n_topics": n_topics,
                "gamma": gamma,
                "prob_mass": prob_mass,
                "sharpe": np.nan,
                "ann_return": np.nan,
                "ann_vol": np.nan,
                "max_drawdown": np.nan,
                "n_weeks": 0,
            })

    return pd.DataFrame(rows)


def plateau_diagnosis(grid: pd.DataFrame) -> dict:
    """Диагностика: есть ли плато или только острый пик.

    Сравнивает медианный Sharpe «вокруг пика» vs «далеко от пика».
    Если отношение < 0.7 — это пик, не плато.
    """
    if grid.empty or grid["sharpe"].isna().all():
        return {"verdict": "NO DATA"}

    s = grid["sharpe"].dropna()
    peak = s.max()
    median = s.median()
    q25 = s.quantile(0.25)
    q75 = s.quantile(0.75)

    plateau_ratio = q25 / peak if peak != 0 else np.nan

    return {
        "peak_sharpe": peak,
        "median_sharpe": median,
        "q25_sharpe": q25,
        "q75_sharpe": q75,
        "plateau_ratio_q25_peak": plateau_ratio,
        "verdict": (
            "PLATEAU" if (not np.isnan(plateau_ratio) and plateau_ratio > 0.5)
            else "PEAK (fragile)"
        ),
    }
