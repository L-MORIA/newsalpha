"""Мини-грид чувствительности {gamma} x {prob_mass} (Этап 6.4).

Критерий «плато, а не острый пик» (контрмера к замечанию Масютина).
На каждом узле грида: sttm_expanding на КАЖДОМ тикере → панель [недели x тикеры]
→ кросс-секционный backtest (топ-20%, равные веса) → gross/net Sharpe.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from newsalpha.backtest.portfolio import backtest, summarize
from newsalpha.sttm.pipeline import sttm_expanding


def sensitivity_grid(
    returns_panel: pd.DataFrame,
    theta: np.ndarray,
    c_words: np.ndarray,
    vocab: dict,
    weeks: list,
    lda_model=None,
    topic_word_lists_override: list | None = None,
    gammas: list[float] | None = None,
    prob_masses: list[float] | None = None,
    first_test_year: int = 2015,
    top_pct: float = 0.20,
    cost: float = 0.0005,
    initial_train_years: int = 2,
    index_norm: str = "sigmoid",
) -> pd.DataFrame:
    """Перебор gamma x prob_mass с кросс-секционным backtest.

    returns_panel: DataFrame[недели x тикеры] недельных доходностей.
    На каждом узле: для каждого тикера sttm_expanding → панель индексов →
    backtest (top-20%, равные веса, оборот) → gross/net Sharpe.
    """
    if gammas is None:
        gammas = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15]
    if prob_masses is None:
        prob_masses = [0.15, 0.20, 0.25, 0.30, 0.40]

    tw = topic_word_lists_override
    if lda_model is not None:
        from newsalpha.sttm.pipeline import topic_word_lists
        tw = topic_word_lists(lda_model, topn=40)
    if tw is None:
        raise ValueError("topic_word_lists_override or lda_model required")

    tickers = list(returns_panel.columns)
    total = len(gammas) * len(prob_masses)
    print(f"  Grid: {len(gammas)} x {len(prob_masses)} = {total} uzkov, "
          f"{len(tickers)} tickerov", flush=True)

    rows = []
    for gi, (gamma, prob_mass) in enumerate(product(gammas, prob_masses)):
        index_panel = {}
        rho_per_ticker = []
        for ticker in tickers:
            r = returns_panel[ticker].dropna()
            if len(r) < 100:
                continue
            try:
                idx = sttm_expanding(
                    r, theta, c_words, tw, vocab, weeks,
                    gamma=gamma, prob_mass=prob_mass,
                    initial_train_years=initial_train_years,
                    norm=index_norm,
                    first_test_year=first_test_year,
                )
                if len(idx) < 20:
                    continue
                index_panel[ticker] = idx
                aligned = pd.concat(
                    [idx.rename("idx"), r.rename("ret")], axis=1
                ).dropna()
                if len(aligned) > 10:
                    rho, _ = sp_stats.spearmanr(aligned["idx"], aligned["ret"])
                    rho_per_ticker.append(rho)
            except Exception:
                continue

        if len(index_panel) < 10:
            rows.append(_nan_row(gamma, prob_mass))
            print(f"  [{gi+1}/{total}] g={gamma:.2f} pm={prob_mass:.2f} "
                  f"-> SKIP (only {len(index_panel)} tickers)", flush=True)
            continue

        sig_df = pd.DataFrame(index_panel)
        ret_df = returns_panel[sig_df.columns]
        common_idx = sig_df.index.intersection(ret_df.index)
        sig_df = sig_df.loc[common_idx]
        ret_df = ret_df.loc[common_idx]

        bt = backtest(sig_df, ret_df, top_pct=top_pct, rates=(cost,), mode="level")
        gross_series = bt["gross"]
        gs = summarize(gross_series)
        net_col = [c for c in bt.columns if c.startswith("net_")]
        ns = summarize(bt[net_col[0]]) if net_col else {"sharpe": np.nan}

        rho_arr = np.array(rho_per_ticker) if rho_per_ticker else np.array([np.nan])

        row = {
            "gamma": gamma,
            "prob_mass": prob_mass,
            "gross_sharpe": gs.get("sharpe", np.nan),
            "gross_return": gs.get("ann_return", np.nan),
            "gross_vol": gs.get("ann_vol", np.nan),
            "gross_maxdd": gs.get("max_drawdown", np.nan),
            "net_sharpe": ns.get("sharpe", np.nan),
            "net_return": ns.get("ann_return", np.nan),
            "spearman_rho_mean": float(rho_arr.mean()),
            "frac_tickers_positive_rho": float((rho_arr > 0).mean()),
            "n_tickers": len(index_panel),
            "n_weeks": len(gross_series),
        }
        rows.append(row)

        done = gi + 1
        print(f"  [{done}/{total}] g={gamma:.2f} pm={prob_mass:.2f} "
              f"-> gross={gs.get('sharpe', 0):.3f} net={ns.get('sharpe', 0):.3f} "
              f"frac_rho+={ (rho_arr > 0).mean():.0%} tickers={len(index_panel)}",
              flush=True)

    return pd.DataFrame(rows)


def _nan_row(gamma: float, prob_mass: float) -> dict:
    return {
        "gamma": gamma,
        "prob_mass": prob_mass,
        "gross_sharpe": np.nan,
        "gross_return": np.nan,
        "gross_vol": np.nan,
        "gross_maxdd": np.nan,
        "net_sharpe": np.nan,
        "net_return": np.nan,
        "spearman_rho_mean": np.nan,
        "frac_tickers_positive_rho": np.nan,
        "n_tickers": 0,
        "n_weeks": 0,
    }


def plateau_diagnosis(grid: pd.DataFrame) -> dict:
    """Диагностика: плато или острый пик по gross_sharpe.

    Сравнивает median vs peak. Если Q25 > 0.5*peak — плато.
    """
    col = "gross_sharpe"
    if grid.empty or grid[col].isna().all():
        return {"verdict": "NO DATA"}

    s = grid[col].dropna()
    peak = s.max()
    median = s.median()
    q25 = s.quantile(0.25)
    q75 = s.quantile(0.75)

    plateau_ratio = q25 / peak if peak != 0 else np.nan

    return {
        "peak_gross_sharpe": peak,
        "median_gross_sharpe": median,
        "q25_gross_sharpe": q25,
        "q75_gross_sharpe": q75,
        "plateau_ratio_q25_peak": plateau_ratio,
        "n_configs": len(s),
        "verdict": (
            "PLATEAU" if (not np.isnan(plateau_ratio) and plateau_ratio > 0.5)
            else "PEAK (fragile)"
        ),
    }
