"""Метрики направления: Acc/F1/AUC/Spearman (Этап 6.3).

Предсказание: индекс(t) → направление доходности(t+1).
Spearman: корреляция Спирмена между индексом и будущей доходностью.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def direction_metrics(
    index: pd.Series,
    returns: pd.Series,
    horizon: int = 1,
) -> dict:
    """Метрики предсказания направления на горизонте horizon недель.

    horizon: на сколько недель вперёд предсказываем (1, 2, 4).
    Returns: accuracy, F1, AUC, Spearman rho, precision, recall.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    df = pd.DataFrame({"idx": index, "ret": returns}).dropna()
    if len(df) < 20:
        return _empty_metrics(horizon)

    future_ret = df["ret"].shift(-horizon)
    df = df.assign(future_ret=future_ret).dropna()
    if len(df) < 20:
        return _empty_metrics(horizon)

    y_true = (df["future_ret"] > 0).astype(int).to_numpy()
    y_score = df["idx"].to_numpy()
    y_pred = (y_score > 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = 0.5
    rho, rho_p = sp_stats.spearmanr(df["idx"], df["future_ret"])

    return {
        "horizon": horizon,
        "n": len(df),
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "auc": auc,
        "spearman_rho": rho,
        "spearman_pvalue": rho_p,
    }


def direction_table(
    index: pd.Series,
    returns: pd.Series,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Таблица метрик по нескольким горизонтам (1, 2, 4 недели)."""
    if horizons is None:
        horizons = [1, 2, 4]
    rows = [direction_metrics(index, returns, h) for h in horizons]
    return pd.DataFrame(rows)


def _empty_metrics(horizon: int) -> dict:
    return {
        "horizon": horizon, "n": 0,
        "accuracy": np.nan, "f1": np.nan, "precision": np.nan,
        "recall": np.nan, "auc": np.nan,
        "spearman_rho": np.nan, "spearman_pvalue": np.nan,
    }
