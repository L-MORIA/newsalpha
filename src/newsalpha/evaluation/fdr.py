"""Benjamini-Hochberg FDR: доля значимых акций (Этап 6.5).

На каждую акцию считаем Spearman(idx, ret) p-value за тестовый период.
BH процедура: сортируем p-values → порог i/m * α → значимые акции.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def per_ticker_spearman(
    index_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Spearman(idx, ret) для каждого тикера.

    index_panel, returns_panel: DataFrame[недели × тикеры].
    Возвращает DataFrame с колонками [ticker, rho, pvalue, n].
    """
    tickers = index_panel.columns.intersection(returns_panel.columns)
    rows = []
    for ticker in sorted(tickers):
        idx = index_panel[ticker].dropna()
        ret = returns_panel[ticker].dropna()
        common = idx.index.intersection(ret.index)
        if len(common) < 10:
            rows.append({"ticker": ticker, "rho": np.nan,
                         "pvalue": np.nan, "n": len(common)})
            continue
        rho, pval = stats.spearmanr(idx.loc[common], ret.loc[common])
        rows.append({"ticker": ticker, "rho": rho, "pvalue": pval, "n": len(common)})
    return pd.DataFrame(rows)


def benjamini_hochberg(
    pvalues: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """BH процедура.

    Возвращает: н Significant, список Significant_tickers, adjusted p-values.
    """
    p = np.asarray(pvalues)
    valid = ~np.isnan(p)
    p_valid = p[valid]
    m = len(p_valid)

    if m == 0:
        return {"n_significant": 0, "n_total": 0, "frac_significant": np.nan,
                "adjusted_pvalues": np.array([]), "significant_mask": np.array([])}

    order = np.argsort(p_valid)
    adjusted = np.full(m, np.nan)
    adjusted[order] = p_valid[order] * m / (np.arange(1, m + 1))
    adjusted = np.minimum(adjusted, 1.0)

    sig = adjusted < alpha
    return {
        "n_significant": int(sig.sum()),
        "n_total": m,
        "frac_significant": sig.sum() / m if m > 0 else 0.0,
        "alpha": alpha,
        "adjusted_pvalues": adjusted,
        "significant_mask": sig,
    }


def fdr_analysis(
    index_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    alpha: float = 0.05,
) -> dict:
    """Полный анализ FDR: per-ticker Spearman + BH."""
    spearman_df = per_ticker_spearman(index_panel, returns_panel)
    bh = benjamini_hochberg(spearman_df["pvalue"].to_numpy(), alpha=alpha)
    return {
        "per_ticker": spearman_df,
        "fdr": bh,
        "conclusion": (
            f"{bh['n_significant']}/{bh['n_total']} tickers significant "
            f"at α={alpha} (FDR corrected)"
        ),
    }
