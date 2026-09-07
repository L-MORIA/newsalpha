"""Базлайны: эндогенные модели (Этап 5).

Запуск:
  python -m scripts.run_baselines --source lenta
  python -m scripts.run_baselines --source kommersant
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from newsalpha.baselines.endogenous import (
    endogenous_baseline,
    compute_baseline_metrics,
)
from newsalpha.baselines.sestm import (
    screening,
    supervised_topic_estimate,
    compute_p_score,
    sestm_expanding,
)
from newsalpha.io.market import load_all_tickers


def load_doc_term(source: str):
    """Загрузить doc-topic матрицу из чекпойнтов LDA."""
    if source == "lenta":
        dt_path = ROOT / "models" / "doc_topic" / "doc_topic_lenta.parquet"
    elif source == "kommersant":
        dt_path = ROOT / "models" / "doc_topic" / "doc_topic_kommersant.parquet"
    else:
        raise ValueError(f"Unknown source: {source}")

    if not dt_path.exists():
        raise FileNotFoundError(f"Doc-topic matrix not found: {dt_path}")

    df = pd.read_parquet(dt_path)
    doc_term = df.values  # all columns are topic weights [n_docs x n_topics]
    doc_names = df.index.tolist()
    return doc_term, doc_names


def load_documents(source: str):
    """Загрузить документы с датами и тикерами из preprocessed parquet."""
    if source == "lenta":
        preproc_path = ROOT / "data" / "processed" / "news_lenta_preproc.parquet"
    elif source == "kommersant":
        preproc_path = ROOT / "data" / "processed" / "news_kommersant_preproc.parquet"
    else:
        raise ValueError(f"Unknown source: {source}")

    if not preproc_path.exists():
        raise FileNotFoundError(f"Preprocessed data not found: {preproc_path}")

    df = pd.read_parquet(preproc_path, columns=["date", "preproc"])
    dates = df["date"].astype(str).tolist()
    tickers = ["ALL"] * len(df)
    doc_names = df.index.astype(str).tolist()

    return doc_names, dates, tickers


def run_endogenous(source: str):
    """Запустить эндогенные базлайны."""
    print(f"Running endogenous baselines for {source}...")

    # Load data
    prices_dir = ROOT / "data" / "raw" / "prices"
    tickers = [p.name.replace("shares_TQBR_", "").replace(".csv", "")
               for p in sorted(prices_dir.glob("shares_TQBR_*.csv"))]
    returns = load_all_tickers(prices_dir, tickers)

    # Run baselines
    results = endogenous_baseline(returns, lags=5, horizon=1, min_train=104)

    # Save results
    output_dir = ROOT / "models" / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for name, result in results.items():
        metrics = result["metrics"]
        print(f"  {name}: Acc={metrics['accuracy']:.3f}, Spearman={metrics['spearman']:.3f}")

        summary[name] = {
            "accuracy": metrics["accuracy"],
            "spearman": metrics["spearman"],
            "n_tickers": metrics["n_tickers"],
        }

        if not result["signals"].empty:
            result["signals"].to_parquet(output_dir / f"signals_{name.lower()}.parquet")

    # Save summary
    with open(output_dir / "endogenous_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to {output_dir}")
    return summary


def run_sestm(source: str):
    """Запустить SESTM-базлайн."""
    print(f"Running SESTM baseline for {source}...")

    # Load data
    prices_dir = ROOT / "data" / "raw" / "prices"
    tickers = [p.name.replace("shares_TQBR_", "").replace(".csv", "")
               for p in sorted(prices_dir.glob("shares_TQBR_*.csv"))]
    returns = load_all_tickers(prices_dir, tickers)

    # Load documents
    doc_names, dates, doc_tickers = load_documents(source)

    # Load doc-term matrix
    doc_term, _ = load_doc_term(source)

    # Run SESTM
    result = sestm_expanding(
        returns_panel=returns,
        doc_term=doc_term,
        dates=pd.Series(dates),
        tickers=tickers,
        doc_tickers=np.array(doc_tickers),
        train_years=10,
        test_years=1,
        lam=1.0,
        min_articles=5,
        top_k=200,
    )

    # Save results
    output_dir = ROOT / "models" / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)

    result.to_parquet(output_dir / f"signals_sestm_{source}.parquet")

    # Compute metrics
    metrics = compute_baseline_metrics(result, returns, horizon=1)
    print(f"  SESTM: Acc={metrics['accuracy']:.3f}, Spearman={metrics['spearman']:.3f}")

    summary = {"SESTM": metrics}
    with open(output_dir / f"sestm_summary_{source}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to {output_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run baselines")
    parser.add_argument("--source", required=True, help="Data source: lenta or kommersant")
    parser.add_argument("--type", choices=["endogenous", "sestm", "all"], default="all",
                        help="Type of baseline to run")
    args = parser.parse_args()

    start_time = time.time()

    if args.type in ["endogenous", "all"]:
        run_endogenous(args.source)

    if args.type in ["sestm", "all"]:
        run_sestm(args.source)

    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
