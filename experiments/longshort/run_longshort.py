"""Long-Short decile test on Kommersant: предсказывает ли STTM кросс-секционный ранг?

Гипотеза: если STTM-индекс извлекает информационный сигнал из новостей, то
long-short (top-20% vs bottom-20%, dollar-neutral) должен давать значимый
Sharpe после издержек. Если Sharpe long-short ≈ 0 — весь «сигнал» в long-only
это рыночная экспозиция (beta), а не альфа из новостей.

Сравнение:
  1. Long-only top-20% (existing): baseline, на Kommersant даёт Sharpe 1.42
  2. Long-short top-20% vs bottom-20%: новый, dollar-neutral
  3. Long-short placebo (50 sims): shuffled returns, ожидаемо ≈ 0

Дизайн:
  - Weekly rebalance по STTM-индексу (level, не delta)
  - Длинная корзина: top-20% по индексу, +1/N_long каждый
  - Короткая корзина: bottom-20% по индексу, -1/N_short каждый
  - Sum of weights = 0 (dollar-neutral), gross exposure = 2.0
  - Издержки: cost_rate × Σ|Δw| за неделю (round-trip ≈ 2 × cost_rate при full rebalance)
  - Borrow cost: borrow_rate × |short notional| (≈ 1 в неделю)
  - Placebo: per-column shuffle, сохраняет mean каждого тикера
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from newsalpha.backtest.portfolio import backtest, summarize
from newsalpha.io.market import load_all_tickers


def longshort_weights(
    signal_row: pd.Series,
    long_pct: float = 0.20,
    short_pct: float = 0.20,
    min_names: int = 10,
) -> pd.Series | None:
    """Long top-pct, short bottom-pct, dollar-neutral, equal weight внутри корзины.

    Returns: Series[ticker -> weight], sum = 0 (если k_long = k_short).
    """
    sig = signal_row.dropna()
    if len(sig) < min_names:
        return None
    n_long = max(1, int(round(long_pct * len(sig))))
    n_short = max(1, int(round(short_pct * len(sig))))
    sorted_tickers = sig.sort_values(ascending=False, kind="stable").index.tolist()
    long_names = set(sorted_tickers[:n_long])
    short_names = set(sorted_tickers[-n_short:])

    w = pd.Series(0.0, index=sig.index)
    for t in long_names:
        w[t] = 1.0 / n_long
    for t in short_names:
        if t not in long_names:  # не пересекаются
            w[t] = -1.0 / n_short
    return w


def backtest_longshort(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    long_pct: float = 0.20,
    short_pct: float = 0.20,
    min_names: int = 10,
    cost_rate: float = 0.001,
    borrow_rate_weekly: float = 0.0003,
) -> pd.DataFrame:
    """Недельный long-short бэктест: top-pct long, bottom-pct short, dollar-neutral.

    signals: [weeks × tickers] STTM-индексы
    returns: [weeks × tickers] недельные доходности
    cost_rate: ставка за единицу |Δw| (round-trip ≈ 2 × cost_rate)
    borrow_rate_weekly: ставка borrow cost за неделю на short notional

    Возвращает: DataFrame[week → gross, turnover, cost, borrow, net]
    """
    sig = signals.astype(float).sort_index()
    rets = returns.astype(float).sort_index()

    rows = []
    weeks = list(sig.index)
    prev_w: pd.Series | None = None
    for i, t in enumerate(weeks[:-1]):
        w = longshort_weights(sig.loc[t], long_pct, short_pct, min_names)
        if w is None:
            continue
        t_next = weeks[i + 1]
        if t_next not in rets.index:
            continue
        r_next = rets.loc[t_next]
        valid = r_next.notna() & w.notna()
        if not valid.any():
            continue
        gross = float((w[valid] * r_next[valid]).sum())

        if prev_w is None:
            turnover = 0.0
        else:
            all_t = w.index.union(prev_w.index)
            delta = w.reindex(all_t, fill_value=0.0) - prev_w.reindex(all_t, fill_value=0.0)
            turnover = float(delta.abs().sum())

        # short notional = сумма |w| для отрицательных весов = ровно 1.0
        # (так как short basket = -1/k_short × k_short = -1)
        short_notional = 1.0
        cost = cost_rate * turnover
        borrow = borrow_rate_weekly * short_notional
        net = gross - cost - borrow

        rows.append({
            "week": t_next,
            "gross": gross,
            "turnover": turnover,
            "cost": cost,
            "borrow": borrow,
            "net": net,
        })
        prev_w = w

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["gross", "turnover", "cost", "borrow", "net"])
    return df.set_index("week")


def shuffle_returns_per_col(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Per-column shuffle: сохраняет mean каждого тикера, разрушает cross-section."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    for col in out.columns:
        vals = out[col].dropna().values.copy()
        rng.shuffle(vals)
        out.loc[out[col].notna(), col] = vals
    return out


def placebo_longshort(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    n_sims: int = 50,
    seed: int = 42,
    **bt_kwargs,
) -> dict:
    """Placebo: n_sims перетасованных доходностей → распределение long-short Sharpe."""
    real_port = backtest_longshort(signals, returns, **bt_kwargs)
    real_sharpe = summarize(real_port["gross"]).get("sharpe", np.nan)

    rng = np.random.default_rng(seed)
    placebo_sharpes = []
    placebo_nets = []
    for s in rng.integers(0, 2**31, size=n_sims):
        shuffled = shuffle_returns_per_col(returns, seed=int(s))
        try:
            port = backtest_longshort(signals, shuffled, **bt_kwargs)
            sh = summarize(port["gross"]).get("sharpe", np.nan)
            sh_net = summarize(port["net"]).get("sharpe", np.nan)
            if not np.isnan(sh):
                placebo_sharpes.append(sh)
            if not np.isnan(sh_net):
                placebo_nets.append(sh_net)
        except Exception:
            continue

    arr = np.array(placebo_sharpes)
    arr_net = np.array(placebo_nets)
    if len(arr) > 1 and arr.std() > 0:
        z = (real_sharpe - arr.mean()) / arr.std()
    else:
        z = np.nan
    p_value = (arr >= real_sharpe).mean() if len(arr) > 0 else np.nan

    return {
        "real_sharpe": real_sharpe,
        "real_sharpe_net": summarize(real_port["net"]).get("sharpe", np.nan),
        "placebo_mean": arr.mean() if len(arr) > 0 else np.nan,
        "placebo_std": arr.std() if len(arr) > 0 else np.nan,
        "placebo_median": np.median(arr) if len(arr) > 0 else np.nan,
        "placebo_95ci": (
            float(np.percentile(arr, 2.5)),
            float(np.percentile(arr, 97.5)),
        ) if len(arr) > 0 else (np.nan, np.nan),
        "placebo_net_mean": arr_net.mean() if len(arr_net) > 0 else np.nan,
        "z_score": z,
        "p_value": p_value,
        "n_sims": len(arr),
        "conclusion": "SIGNIFICANT" if (not np.isnan(z) and abs(z) > 2.0) else "NOT SIGNIFICANT",
    }


def main():
    cfg = yaml.safe_load(open(ROOT / "config/default.yaml", encoding="utf-8"))

    # Load STTM index and returns
    sttm_path = ROOT / "models" / "sttm_indices" / "sttm_index_kommersant.parquet"
    if not sttm_path.exists():
        print(f"ERROR: {sttm_path} not found. Run build_sttm_index.py first.")
        sys.exit(1)

    sttm = pd.read_parquet(sttm_path)
    sttm.index = pd.to_datetime(sttm.index)
    rets = load_all_tickers(ROOT / cfg["data"]["prices_dir"], cfg["tickers"])
    rets.index = pd.to_datetime(rets.index)

    common_tickers = sttm.columns.intersection(rets.columns)
    common_dates = sttm.index.intersection(rets.index)
    sttm = sttm.loc[common_dates, common_tickers]
    rets = rets.loc[common_dates, common_tickers]

    print("=" * 70)
    print("  LONG-SHORT DECILE TEST (Kommersant, weekly, 2013-2021)")
    print("=" * 70)
    print(f"  Период: {common_dates.min().date()} – {common_dates.max().date()}")
    print(f"  Недель: {len(common_dates)}")
    print(f"  Тикеров: {len(common_tickers)}")
    print()

    # ── 1. Long-only baseline ─────────────────────────────────────────
    rates = tuple(cfg["strategy"]["commission_scenarios"])
    long_only = backtest(sttm, rets, top_pct=0.20, rates=rates, mode="level")
    lo_gross = summarize(long_only["gross"])
    lo_net_05 = summarize(long_only["net_0.0005"])
    lo_net_15 = summarize(long_only["net_0.0015"])

    print("  [1] LONG-ONLY top-20% (baseline):")
    print(f"      Gross Sharpe:  {lo_gross['sharpe']:>7.3f}  | Ann Ret: {lo_gross['ann_return']*100:>+5.1f}% | Vol: {lo_gross['ann_vol']*100:>4.1f}%")
    print(f"      Net 0.05%:     {lo_net_05['sharpe']:>7.3f}  | Ann Ret: {lo_net_05['ann_return']*100:>+5.1f}%")
    print(f"      Net 0.15%:     {lo_net_15['sharpe']:>7.3f}  | Ann Ret: {lo_net_15['ann_return']*100:>+5.1f}%")
    print(f"      Mean turnover: {long_only['turnover'].mean():.3f}")
    print()

    # ── 2. Long-short (NEW) ───────────────────────────────────────────
    cost_rate = 0.001          # 10 bps per unit |Δw|, ~20 bps full rebalance
    borrow_rate = 0.0003       # ~1.5% annual, реалистично для liquid Russian blue chips

    ls = backtest_longshort(
        sttm, rets,
        long_pct=0.20, short_pct=0.20,
        cost_rate=cost_rate, borrow_rate_weekly=borrow_rate,
    )
    ls_gross = summarize(ls["gross"])
    ls_net = summarize(ls["net"])

    print("  [2] LONG-SHORT top-20% vs bottom-20% (dollar-neutral):")
    print(f"      Gross Sharpe:  {ls_gross['sharpe']:>7.3f}  | Ann Ret: {ls_gross['ann_return']*100:>+5.1f}% | Vol: {ls_gross['ann_vol']*100:>4.1f}%")
    print(f"      Net Sharpe:    {ls_net['sharpe']:>7.3f}  | Ann Ret: {ls_net['ann_return']*100:>+5.1f}% | MaxDD: {ls_net['max_drawdown']*100:>5.1f}%")
    print(f"      Mean turnover: {ls['turnover'].mean():.3f}")
    print(f"      Total return (gross):  {((1 + ls['gross']).prod() - 1)*100:>+.1f}%")
    print(f"      Total return (net):    {((1 + ls['net']).prod() - 1)*100:>+.1f}%")
    print()

    # ── 3. Placebo ────────────────────────────────────────────────────
    print("  [3] Placebo (50 sims, per-column shuffle)...")
    pbt = placebo_longshort(
        sttm, rets, n_sims=50,
        cost_rate=cost_rate, borrow_rate_weekly=borrow_rate,
    )
    print(f"      Real Sharpe (gross):  {pbt['real_sharpe']:>+7.3f}")
    print(f"      Real Sharpe (net):    {pbt['real_sharpe_net']:>+7.3f}")
    print(f"      Placebo mean:         {pbt['placebo_mean']:>+7.3f} ± {pbt['placebo_std']:.3f}")
    print(f"      Placebo median:       {pbt['placebo_median']:>+7.3f}")
    print(f"      Placebo 95% CI:       [{pbt['placebo_95ci'][0]:>+6.3f}, {pbt['placebo_95ci'][1]:>+6.3f}]")
    print(f"      z-score:              {pbt['z_score']:>+6.2f}")
    print(f"      p-value:              {pbt['p_value']:.3f}")
    print(f"      Conclusion:           {pbt['conclusion']}")
    print()

    # ── 4. Interpretation ─────────────────────────────────────────────
    print("=" * 70)
    print("  ИНТЕРПРЕТАЦИЯ")
    print("=" * 70)
    if not np.isnan(pbt['real_sharpe_net']) and pbt['real_sharpe_net'] > 0.3:
        if pbt['p_value'] < 0.05:
            print("  ✓ STTM ПРЕДСКАЗЫВАЕТ кросс-секционный ранг (long-short net > 0.3, p<0.05).")
            print("    Длинная корзина систематически обыгрывает короткую после издержек.")
            print("    Следовательно, новостной сигнал имеет информационную ценность.")
        else:
            print("  ~ Long-short net > 0.3, но placebo p>0.05 — сигнал слабый / ненадёжный.")
            print("    Требуется больше данных (мульти-источник, multi-period) для подтверждения.")
    elif not np.isnan(pbt['real_sharpe_net']) and pbt['real_sharpe_net'] > 0:
        print("  ~ Long-short net > 0 но < 0.3 — слабый сигнал, граничит со шумом.")
    else:
        print("  ✗ Long-short net ≤ 0 — STTM НЕ предсказывает кросс-секционный ранг.")
        print("    Вся «прибыль» long-only (Sharpe 1.42) = рыночная экспозиция (beta), не альфа.")
        print("    Это подтверждает, что в long-only сигнал STTM не добавляет ценности поверх рынка.")
    print()

    # ── 5. Save ───────────────────────────────────────────────────────
    out = {
        "long_only": {
            "gross_sharpe": lo_gross.get("sharpe"),
            "net_0.0005_sharpe": lo_net_05.get("sharpe"),
            "net_0.0015_sharpe": lo_net_15.get("sharpe"),
            "ann_return": lo_gross.get("ann_return"),
            "ann_vol": lo_gross.get("ann_vol"),
            "mean_turnover": float(long_only["turnover"].mean()),
        },
        "long_short": {
            "gross_sharpe": ls_gross.get("sharpe"),
            "net_sharpe": ls_net.get("sharpe"),
            "ann_return_gross": ls_gross.get("ann_return"),
            "ann_return_net": ls_net.get("ann_return"),
            "ann_vol": ls_gross.get("ann_vol"),
            "max_drawdown": ls_net.get("max_drawdown"),
            "mean_turnover": float(ls["turnover"].mean()),
        },
        "placebo": pbt,
        "config": {
            "long_pct": 0.20, "short_pct": 0.20,
            "cost_rate": cost_rate, "borrow_rate_weekly": borrow_rate,
            "n_sims": 50,
        },
    }
    out_path = ROOT / "findings" / "longshort_kommersant_2026-08-26.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Сохранено: {out_path}")


if __name__ == "__main__":
    main()
