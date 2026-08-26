"""Бэктест OOS 2022-2026."""
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from newsalpha.backtest.portfolio import backtest, summarize
from newsalpha.io.market import load_all_tickers


def main():
    cfg = yaml.safe_load(open(ROOT / "config/default.yaml", encoding="utf-8"))
    tickers = cfg["tickers"]

    sttm = pd.read_parquet(ROOT / "models/sttm_indices/sttm_index_kommersant_oos.parquet")
    print(f"STTM index: {sttm.shape}")

    rets = load_all_tickers(ROOT / cfg["data"]["prices_dir"], tickers)
    rets.index = pd.to_datetime(rets.index)
    print(f"Weekly returns: {rets.shape}")

    sttm.index = pd.to_datetime(sttm.index)
    common_dates = sttm.index.intersection(rets.index)
    sttm = sttm.loc[common_dates]
    rets = rets.loc[common_dates]
    print(f"Common dates: {len(common_dates)}")

    port = backtest(sttm, rets, top_pct=0.20, min_names=10,
                    rates=(0.0, 0.0005, 0.0015), mode="level")
    m = summarize(port["gross"])

    print()
    print("=== OOS 2022-2026 RESULTS (Frozen LDA 2013-2021) ===")
    print(f"Gross Sharpe:   {m['sharpe']:.3f}")
    print(f"Ann Return:     {m['ann_return'] * 100:.1f}%")
    print(f"Ann Vol:        {m['ann_vol'] * 100:.1f}%")
    print(f"MaxDD:          {m['max_drawdown'] * 100:.1f}%")
    print(f"Weeks:          {m['n']}")
    print()
    for rate in [0.0, 0.0005, 0.0015]:
        col = f"net_{rate:g}"
        if col in port.columns:
            nm = summarize(port[col])
            print(f"net_{rate * 100:.2f}%: Sharpe={nm['sharpe']:.3f}")
    print()
    print(f"Mean turnover:  {port['turnover'].mean():.3f}")
    print(f"Mean n_stocks:  {port['n'].mean():.1f}")
    print(f"Min n_stocks:   {port['n'].min()}")
    print(f"Max n_stocks:   {port['n'].max()}")


if __name__ == "__main__":
    main()
