"""Бэктест STTM-портфеля по готовым индексам (Этап 6).

Пример:
    python scripts/run_backtest.py --source lenta
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from newsalpha.backtest.portfolio import backtest, summarize
from newsalpha.io.market import load_all_tickers


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--source", default="lenta")
    ap.add_argument("--mode", choices=["level", "delta"], default="level",
                    help="сигнал ранжирования: уровень индекса или прирост")
    ap.add_argument("--slippage", action="store_true",
                    help="добавить slippage из конфига к комиссиям (net-строго)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    st = cfg["strategy"]
    rates = tuple(st["commission_scenarios"])
    slip = st["slippage"] if args.slippage else 0.0

    idx = pd.read_parquet(
        ROOT / cfg["paths"]["sttm_indices"] / f"sttm_index_{args.source}.parquet")
    rets = load_all_tickers(cfg["data"]["prices_dir"], list(idx.columns))
    rets.index = pd.to_datetime(rets.index)

    res = backtest(idx, rets, top_pct=st["top_pct"],
                   rates=rates, slippage=slip, mode=args.mode)

    rows = []
    rows.append({"scenario": "gross", **summarize(res["gross"])})
    for rate in rates:
        rows.append({"scenario": f"net_{rate:g}{'+' + str(slip) if slip else ''}",
                     **summarize(res[f"net_{rate:g}"])})
    summary = pd.DataFrame(rows).set_index("scenario")

    out_path = ROOT / "models" / "backtest" / \
        f"backtest_{args.source}_{args.mode}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(out_path)

    print(f"источник: {args.source} | сигнал: {args.mode} | "
          f"top_pct={st['top_pct']} | недель: {len(res)} | "
          f"оборот ср.: {res['turnover'].mean():.3f}")
    print(summary.round(4).to_string())
    print(f"\nнедельные серии → {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
