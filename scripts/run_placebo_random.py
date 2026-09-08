"""Random-signal placebo: случайные портфели топ-20% на РЕАЛЬНЫХ доходностях.

Нуль-вопрос: даёт ли STTM-сигнал что-то сверх механики отбора топ-20%
на растущем рынке? Сигналы случайны, доходности настоящие.
Дешёвый тест (без перестройки LDA/индекса): 50 симов за минуты.

Заменяет невоспроизводимое число p=0.50 от удалённого temp-скрипта
(placebo_kommersant_2026-08-25.md): воспроизводимо кодом репозитория.

Пример:
    python scripts/run_placebo_random.py --source kommersant --n-sims 50 --seed 7
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from newsalpha.backtest.portfolio import backtest, summarize
from newsalpha.evaluation.placebo import _build_index_panel


def main() -> None:
    ap = argparse.ArgumentParser(description="Random-signal placebo for STTM.")
    ap.add_argument("--source", default="kommersant")
    ap.add_argument("--n-sims", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--top-pct", type=float, default=0.20)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "config/default.yaml", encoding="utf-8"))

    from newsalpha.io.market import load_all_tickers

    wret = load_all_tickers(str(ROOT / cfg["data"]["prices_dir"]), cfg["tickers"])
    wret.index = pd.DatetimeIndex(wret.index)

    from gensim.corpora import Dictionary
    from gensim.models import LdaMulticore

    lda_path = ROOT / "models" / "lda" / args.source
    best_k = json.loads((lda_path / "best.json").read_text(encoding="utf-8"))["best_k"]
    lda = LdaMulticore.load(str(lda_path / f"lda_k{best_k}.model"))
    dictionary = Dictionary.load(str(lda_path / "dictionary.dict"))
    vocab = dictionary.token2id

    from newsalpha.sttm.pipeline import build_streams, topic_word_lists

    tw = topic_word_lists(lda, topn=40)
    df_pre = pd.read_parquet(
        ROOT / cfg["data"]["processed_dir"] / f"news_{args.source}_preproc.parquet",
        columns=["date", "preproc"],
    )
    doc_topic = pd.read_parquet(
        ROOT / cfg["paths"]["doc_topic"] / f"doc_topic_{args.source}.parquet"
    ).to_numpy()
    df_pre = df_pre.iloc[: doc_topic.shape[0]]
    docs_tokens = [s.split() for s in df_pre["preproc"]]
    theta, c_words, cal_weeks = build_streams(doc_topic, docs_tokens, df_pre["date"], vocab)
    del docs_tokens

    sig_df = _build_index_panel(
        wret, theta, c_words, tw, vocab, cal_weeks,
        gamma=0.05, prob_mass=0.3, first_test_year=2015,
        initial_train_years=2, norm="sigmoid",
    )
    ret_df = wret[sig_df.columns]
    common = sig_df.index.intersection(ret_df.index)
    sig_df, ret_df = sig_df.loc[common], ret_df.loc[common]

    bt_real = backtest(sig_df, ret_df, top_pct=args.top_pct, rates=(0.0005,), mode="level")
    real = float(summarize(bt_real["gross"])["sharpe"])
    print(f"REAL gross Sharpe={real:.4f} n={len(common)}", flush=True)

    rng = np.random.default_rng(args.seed)
    sims: list[float] = []
    for i in range(args.n_sims):
        rnd = pd.DataFrame(
            rng.normal(size=sig_df.shape), index=sig_df.index, columns=sig_df.columns
        )
        bt = backtest(rnd, ret_df, top_pct=args.top_pct, rates=(0.0005,), mode="level")
        sims.append(float(summarize(bt["gross"])["sharpe"]))
        if (i + 1) % 10 == 0:
            print(f"  sim {i + 1}/{args.n_sims} last={sims[-1]:.3f}", flush=True)

    arr = np.array(sims)
    z = float((real - arr.mean()) / arr.std(ddof=1))
    try:
        from scipy.stats import norm  # type: ignore

        p = float(1.0 - norm.cdf(z))
    except ImportError:
        p = float("nan")
    conclusion = (
        "SIGNIFICANT (signal beats random baskets)"
        if p < 0.05
        else "NOT SIGNIFICANT (selection artifact)"
    )
    out = {
        "real_sharpe": real,
        "placebo_mean": float(arr.mean()),
        "placebo_std": float(arr.std(ddof=1)),
        "placebo_median": float(np.median(arr)),
        "z_score": z,
        "p_value": p,
        "n_sims": args.n_sims,
        "seed": args.seed,
        "sims": sims,
        "conclusion": conclusion,
    }
    print(
        f"RANDOM-SIGNAL n={args.n_sims} mean={arr.mean():.3f} "
        f"std={arr.std(ddof=1):.3f} z={z:.2f} p={p:.4f}",
        flush=True,
    )
    print("CONCLUSION:", conclusion, flush=True)

    dest = ROOT / "findings" / f"placebo_random_{args.source}_seed{args.seed}.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved -> {dest}", flush=True)


if __name__ == "__main__":
    main()
