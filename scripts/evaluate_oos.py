"""Evaluation OOS 2022-2026: direction accuracy, statistical tests, diagnostics."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    sttm = pd.read_parquet(ROOT / "models/sttm_indices/sttm_index_kommersant_full.parquet")
    sttm.index = pd.to_datetime(sttm.index)

    from newsalpha.io.market import load_all_tickers
    import yaml
    cfg = yaml.safe_load(open(ROOT / "config/default.yaml", encoding="utf-8"))
    rets = load_all_tickers(ROOT / cfg["data"]["prices_dir"], cfg["tickers"])
    rets.index = pd.to_datetime(rets.index)

    common = sttm.index.intersection(rets.index)
    sttm = sttm.loc[common]
    rets = rets.loc[common]
    print(f"Общие даты: {len(common)}\n")

    # ── 1. Direction Accuracy (per-ticker) ───────────────────────────
    print("=" * 60)
    print("  DIRECTION ACCURACY (predicting up/down)")
    print("=" * 60)

    all_dir_acc = []
    for t in cfg["tickers"]:
        aligned = pd.concat([sttm[t].rename("idx"), rets[t].rename("ret")], axis=1).dropna()
        if len(aligned) < 10:
            continue
        # Direction: sign(idx) vs sign(ret)
        correct = (np.sign(aligned["idx"]) == np.sign(aligned["ret"])).mean()
        all_dir_acc.append(correct)
        print(f"  {t:6s}: {correct*100:.1f}%  (n={len(aligned)})")

    mean_dir = np.mean(all_dir_acc)
    print(f"\n  MEAN Direction Accuracy: {mean_dir*100:.1f}%")
    print(f"  RANDOM baseline:         50.0%")
    print(f"  Improvement:             {(mean_dir-0.5)*100:+.1f}pp")

    # ── 2. Statistical significance of mean Spearman ─────────────────
    print("\n" + "=" * 60)
    print("  SPEARMAN CORRELATION STATISTICS")
    print("=" * 60)

    spears = []
    for t in cfg["tickers"]:
        aligned = pd.concat([sttm[t].rename("idx"), rets[t].rename("ret")], axis=1).dropna()
        if len(aligned) > 10:
            s = aligned["idx"].corr(aligned["ret"], method="spearman")
            spears.append(s)

    spears = np.array(spears)
    print(f"  N tickers:       {len(spears)}")
    print(f"  Mean Spearman:   {spears.mean():+.4f}")
    print(f"  Std Spearman:    {spears.std():.4f}")
    print(f"  Median Spearman: {np.median(spears):+.4f}")

    # One-sample t-test: H0: mean_spearman = 0
    t_stat, p_value = stats.ttest_1samp(spears, 0.0)
    print(f"\n  One-sample t-test (H0: mean=0):")
    print(f"    t-statistic:  {t_stat:.3f}")
    print(f"    p-value:      {p_value:.4f}")
    print(f"    Significant:  {'YES (p<0.05)' if p_value < 0.05 else 'NO (p>=0.05)'}")

    # ── 3. Portfolio-level direction accuracy ─────────────────────────
    print("\n" + "=" * 60)
    print("  PORTFOLIO-LEVEL ANALYSIS")
    print("=" * 60)

    # Construct portfolio return (top 20% by STTM index)
    portfolio_rets = []
    for date in sttm.index:
        row = sttm.loc[date].dropna()
        if len(row) < 10:
            continue
        n_select = max(int(len(row) * 0.20), 1)
        top_tickers = row.nlargest(n_select).index
        port_ret = rets.loc[date, top_tickers].mean()
        portfolio_rets.append({"date": date, "port_ret": port_ret})

    port_df = pd.DataFrame(portfolio_rets).set_index("date")
    print(f"  Portfolio weeks: {len(port_df)}")

    # Portfolio direction
    port_dir_correct = (port_df["port_ret"] > 0).mean()
    print(f"  Portfolio positive weeks: {port_dir_correct*100:.1f}%")
    print(f"  (RANDOM baseline: ~50%)")

    # ── 4. Rank correlation over time (rolling) ──────────────────────
    print("\n" + "=" * 60)
    print("  ROLLING SPEARMAN (26-week window)")
    print("=" * 60)

    # Compute weekly cross-sectional Spearman
    weekly_spear = []
    for date in sttm.index:
        row_s = sttm.loc[date].dropna()
        row_r = rets.loc[date].dropna()
        common_tickers = row_s.index.intersection(row_r.index)
        if len(common_tickers) < 10:
            continue
        s = row_s[common_tickers].corr(row_r[common_tickers], method="spearman")
        weekly_spear.append({"date": date, "spearman": s})

    ws_df = pd.DataFrame(weekly_spear).set_index("date")
    ws_df["rolling"] = ws_df["spearman"].rolling(26, min_periods=10).mean()

    print(f"  Weekly Spearman mean:  {ws_df['spearman'].mean():+.4f}")
    print(f"  Weekly Spearman std:   {ws_df['spearman'].std():.4f}")
    print(f"  Positive weeks:        {(ws_df['spearman'] > 0).mean()*100:.1f}%")
    print(f"  Rolling mean (26w):    {ws_df['rolling'].dropna().iloc[-1]:+.4f}")

    # ── 5. Calmar ratio ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RISK-ADJUSTED METRICS (Option B)")
    print("=" * 60)

    cumret = (1 + port_df["port_ret"]).cumprod()
    total_return = cumret.iloc[-1] - 1
    peak = cumret.cummax()
    drawdown = (cumret - peak) / peak
    max_dd = drawdown.min()
    years = len(port_df) / 52
    ann_return = (1 + total_return) ** (1 / years) - 1
    ann_vol = port_df["port_ret"].std() * np.sqrt(52)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    print(f"  Total Return:  {total_return*100:+.1f}%")
    print(f"  Ann Return:    {ann_return*100:+.1f}%")
    print(f"  Ann Vol:       {ann_vol*100:.1f}%")
    print(f"  MaxDD:         {max_dd*100:.1f}%")
    print(f"  Sharpe:        {sharpe:.3f}")
    print(f"  Calmar:        {calmar:.3f}")

    # ── 6. Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY: OOS 2022-2026")
    print("=" * 60)
    print(f"  Sharpe:             {sharpe:.3f}")
    print(f"  Direction Accuracy: {mean_dir*100:.1f}% (vs 50% random)")
    print(f"  Mean Spearman:      {spears.mean():+.4f} (p={p_value:.4f})")
    print(f"  Significant:        {'YES' if p_value < 0.05 else 'NO'}")
    print(f"\n  CONCLUSION: STTM signal is {'STATISTICALLY SIGNIFICANT' if p_value < 0.05 else 'NOT STATISTICALLY SIGNIFICANT'} on OOS")


if __name__ == "__main__":
    main()
