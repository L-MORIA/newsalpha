"""Грейнджер-причинность индекса → доходность (Этап 6.1).

H0: индекс не Грейнджер-причинствует доходность.
Максимальный лаг: maxlag=5 (по статье). ADF для проверки стационарности.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def adf_test(series: pd.Series, name: str = "") -> dict:
    """Augmented Dickey-Fuller test. H0: unit root (non-stationary)."""
    from statsmodels.tsa.stattools import adfuller

    clean = series.dropna()
    if len(clean) < 20:
        return {"name": name, "adf_stat": np.nan, "pvalue": np.nan,
                "stationary": False, "nobs": len(clean)}
    result = adfuller(clean, autolag="AIC")
    return {
        "name": name,
        "adf_stat": result[0],
        "pvalue": result[1],
        "nobs": result[3],
        "stationary": result[1] < 0.05,
    }


def granger_causality(
    index: pd.Series,
    returns: pd.Series,
    maxlag: int = 5,
    test_type: str = "ssr_ftest",
) -> dict:
    """Granger causality: does index Granger-cause returns?

    Returns dict with F-stat, p-value per lag, and summary.
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    df = pd.DataFrame({"ret": returns, "idx": index}).dropna()
    if len(df) < maxlag + 20:
        return {"maxlag": maxlag, "significant_lags": 0,
                "min_pvalue": np.nan, "results": {}}

    results = {}
    for lag in range(1, maxlag + 1):
        try:
            test = grangercausalitytests(df[["ret", "idx"]], maxlag=lag, verbose=False)
            pval = test[lag][0][test_type][1]
            fstat = test[lag][0][test_type][0]
            results[lag] = {"f_stat": fstat, "pvalue": pval}
        except Exception:
            results[lag] = {"f_stat": np.nan, "pvalue": np.nan}

    sig_lags = sum(1 for v in results.values() if v["pvalue"] < 0.05)
    min_p = min((v["pvalue"] for v in results.values()), default=np.nan)

    return {
        "maxlag": maxlag,
        "significant_lags": sig_lags,
        "min_pvalue": min_p,
        "any_significant": sig_lags > 0,
        "results": results,
    }


def evaluate_granger(
    index: pd.Series,
    returns: pd.Series,
    maxlag: int = 5,
) -> dict:
    """Полная оценка: ADF + Грейнджер для stock_index → returns."""
    adf_idx = adf_test(index, "index")
    adf_ret = adf_test(returns, "returns")
    gr = granger_causality(index, returns, maxlag=maxlag)

    return {
        "adf_index": adf_idx,
        "adf_returns": adf_ret,
        "granger": gr,
        "conclusion": (
            "GRANGER-CAUSAL" if gr.get("any_significant")
            else "NO EVIDENCE"
        ),
    }
