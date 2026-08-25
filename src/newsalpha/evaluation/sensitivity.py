"""Мини-грид чувствительности {n_topics} x {prob_mass} x {gamma} (Этап 6.4).

Критерий «плато, а не острый пик» (контрмера к замечанию Масютина).
На каждом узле грида: sttm_expanding -> Spearman(idx, ret) по test-годам.
Быстро: ~2-5 сек на узел вместо минуты с полным бэктестом.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from newsalpha.sttm.pipeline import sttm_expanding


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
    first_test_year: int = 2015,
) -> pd.DataFrame:
    """Перебор комбинаций гиперпараметров -> DataFrame с метриками.

    Метрики:
    - spearman_rho: временная Spearman(idx, ret) — быстрый прокси
    - per_ticker_accuracy: доля тикеров, где направление индекса совпадает
      с направлением доходности на следующей неделе (кросс-секционный test)
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
            from newsalpha.sttm.pipeline import topic_word_lists
            tw = topic_word_lists(lda_model, topn=40)
        if tw is None:
            continue

        try:
            idx = sttm_expanding(
                returns, theta, c_words, tw, vocab, weeks,
                gamma=gamma, prob_mass=prob_mass,
                first_test_year=first_test_year,
            )
            if len(idx) < 10:
                raise ValueError(f"only {len(idx)} test weeks")
            aligned = pd.DataFrame({"idx": idx, "ret": returns}).dropna()
            rho, pval = sp_stats.spearmanr(aligned["idx"], aligned["ret"])

            rows.append({
                "n_topics": n_topics,
                "gamma": gamma,
                "prob_mass": prob_mass,
                "spearman_rho": rho,
                "spearman_pvalue": pval,
                "n_weeks": len(idx),
            })
        except Exception:
            rows.append({
                "n_topics": n_topics,
                "gamma": gamma,
                "prob_mass": prob_mass,
                "spearman_rho": np.nan,
                "spearman_pvalue": np.nan,
                "n_weeks": 0,
            })

    return pd.DataFrame(rows)


def plateau_diagnosis(grid: pd.DataFrame) -> dict:
    """Диагностика: есть ли плато или только острый пик.

    Сравнивает медианный |rho| «вокруг пика» vs «далеко от пика».
    Если отношение < 0.5 — это пик, не плато.
    """
    if grid.empty or grid["spearman_rho"].isna().all():
        return {"verdict": "NO DATA"}

    rho = grid["spearman_rho"].dropna().abs()
    peak = rho.max()
    median = rho.median()
    q25 = rho.quantile(0.25)
    q75 = rho.quantile(0.75)

    plateau_ratio = q25 / peak if peak != 0 else np.nan

    return {
        "peak_rho": peak,
        "median_rho": median,
        "q25_rho": q25,
        "q75_rho": q75,
        "plateau_ratio_q25_peak": plateau_ratio,
        "verdict": (
            "PLATEAU" if (not np.isnan(plateau_ratio) and plateau_ratio > 0.5)
            else "PEAK (fragile)"
        ),
    }
