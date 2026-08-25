"""Портфельный бэктест STTM-индексов (Этап 6).

Правило из конфига/статьи: каждую пятницу ранжируем тикеры по индексу недели,
длинные позиции — топ top_pct (20%), равные веса внутри корзины; удержание до
следующей ребалансировки, доходность реализуется на следующей неделе
(w_t · r_{t+1}). Оборот считается только по фактически изменённым весам
(эквивалент rebalance_on_change_only: тот же набор имён ⇒ нулевой оборот),
издержки = ставка × Σ|Δw|. Sharpe без rf — как в статье (гросс-чекпойнт).
"""
import numpy as np
import pandas as pd


def _select(signal_row: pd.Series, prev_cols: list[str] | None,
            top_pct: float, min_names: int) -> pd.Series | None:
    """Равновзвешенная корзина топ-top_pct по сигналу; None если вселенная мала."""
    sig = signal_row.dropna()
    if len(sig) < min_names:
        return None
    k = max(1, int(round(top_pct * len(sig))))
    # стабильная сортировка: при равенстве сигналов порядок детерминирован
    names = list(sig.sort_values(ascending=False, kind="stable").index[:k])
    w = pd.Series(1.0 / len(names), index=names)
    if prev_cols is not None and set(names) == set(prev_cols):
        return w.reindex(prev_cols)  # идентичная корзина: веса без изменений
    return w


def backtest(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    top_pct: float = 0.20,
    min_names: int = 10,
    rates: tuple[float, ...] = (0.0,),
    slippage: float = 0.0,
    mode: str = "level",
) -> pd.DataFrame:
    """Недельный бэктест по широким матрицам [недели × тикеры].

    signals: STTM-индексы (level) или их приращения (mode="delta").
    Возвращает DataFrame по неделям реализации t+1: колонка gross, turnover
    и net_{rate} для каждой ставки издержек.
    """
    if mode == "delta":
        # parquet может вернуть object-колонки с None → приводим к float/NaN
        signals = signals.astype(float).diff()
    else:
        signals = signals.astype(float)
    sig = signals.sort_index()
    rets = returns.sort_index()

    rows = []
    prev_w: pd.Series | None = None
    weeks = list(sig.index)
    for i, t in enumerate(weeks[:-1]):
        w = _select(sig.loc[t], None if prev_w is None else list(prev_w.index),
                    top_pct, min_names)
        if w is None:
            continue
        t_next = weeks[i + 1]
        r_next = rets.loc[t_next].reindex(w.index)
        valid = r_next.notna()
        if not valid.any():
            continue
        w_use = (w[valid] / w[valid].sum()) if not valid.all() else w
        gross = float((w_use * r_next[valid]).sum())
        turnover = 0.0 if prev_w is None else \
            float((w_use.subtract(prev_w, fill_value=0.0)).abs().sum())
        row = {"week": t_next, "n": int(valid.sum()), "gross": gross,
               "turnover": turnover}
        for rate in rates:
            row[f"net_{rate:g}"] = gross - (rate + slippage) * turnover
        rows.append(row)
        prev_w = w_use

    return pd.DataFrame(rows).set_index("week")


def summarize(port: pd.Series, periods_per_year: int = 52) -> dict:
    """Годовые метрики серии недельных доходностей (rf=0, как в статье)."""
    port = port.dropna()
    if len(port) < 2 or port.std(ddof=1) == 0:
        return {"n": len(port)}
    ann_ret = port.mean() * periods_per_year
    ann_vol = port.std(ddof=1) * np.sqrt(periods_per_year)
    equity = (1.0 + port).cumprod()
    max_dd = float((equity / equity.cummax() - 1.0).min())
    return {
        "n": len(port),
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol,
        "max_drawdown": max_dd,
    }
