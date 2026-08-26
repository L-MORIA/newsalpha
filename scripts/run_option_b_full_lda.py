"""Option B: LDA на полном корпусе 2013-2026, бэктест OOS.

Шаги:
1. Обучить LDA k=50 на ВСЕХ документах 2013-2026 (42 071)
2. Вычислить doc-topic для ВСЕХ документов
3. Построить STTM-индекс для OOS (first_test_year=2022)
4. Бэктест OOS 2022-2026

Сравнение с Option A (замороженная LDA 2013-2021 → Sharpe -0.32).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from newsalpha.io.market import load_all_tickers
from newsalpha.sttm.pipeline import (
    build_streams,
    sttm_expanding,
    topic_word_lists,
)
from newsalpha.backtest.portfolio import backtest, summarize


def main():
    cfg = yaml.safe_load(open(ROOT / "config/default.yaml", encoding="utf-8"))
    sttm_cfg = cfg["sttm"]
    seed = cfg["seed"]

    # ── 1. Load full preprocessed corpus ──────────────────────────────
    preproc_path = ROOT / cfg["data"]["processed_dir"] / "news_kommersant_preproc.parquet"
    df = pd.read_parquet(preproc_path)
    df["date"] = pd.to_datetime(df["date"])
    print(f"Документов: {len(df)}, период: {df['date'].min().date()} – {df['date'].max().date()}")

    texts = [s.split() for s in df["preproc"]]

    # ── 2. Train LDA on FULL corpus ──────────────────────────────────
    lda_dir = ROOT / "models/lda/kommersant_full"
    lda_dir.mkdir(parents=True, exist_ok=True)

    dict_path = lda_dir / "dictionary.dict"
    model_path = lda_dir / "lda_k50.model"

    if model_path.exists():
        print("LDA модель уже существует, загружаю...")
        model = LdaMulticore.load(str(model_path))
        dictionary = Dictionary.load(str(dict_path))
    else:
        print("Обучаю LDA k=50 на полном корпусе...")
        t0 = time.time()

        dictionary = Dictionary(texts)
        dictionary.filter_extremes(no_below=10, no_above=0.4, keep_n=None)
        dictionary.compactify()
        dictionary.save(str(dict_path))
        print(f"  Словарь: {len(dictionary)} слов ({time.time() - t0:.1f} сек)")

        corpus = [dictionary.doc2bow(t) for t in texts]
        model = LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=50,
            random_state=seed,
            passes=10,
            workers=4,
            iterations=50,
            chunksize=2000,
        )
        model.save(str(model_path))
        print(f"  LDA обучена за {time.time() - t0:.1f} сек")

    vocab = dictionary.token2id

    # ── 3. Compute doc-topic for ALL documents ───────────────────────
    doc_topic_path = ROOT / "models/doc_topic/doc_topic_kommersant_full.parquet"
    if doc_topic_path.exists():
        print("doc-topic уже существует, загружаю...")
        doc_topic = pd.read_parquet(doc_topic_path).to_numpy()
    else:
        print("Вычисляю doc-topic для всех документов...")
        t0 = time.time()
        corpus = [dictionary.doc2bow(t) for t in texts]
        rows = []
        for bow in corpus:
            topic_dist = model.get_document_topics(bow, minimum_probability=0.0)
            row = {t: p for t, p in topic_dist}
            rows.append(row)
        doc_topic = pd.DataFrame(rows).fillna(0.0).to_numpy()
        pd.DataFrame(doc_topic).to_parquet(doc_topic_path, index=False)
        print(f"  doc-topic: {doc_topic.shape} ({time.time() - t0:.1f} сек)")

    print(f"doc-topic shape: {doc_topic.shape}")

    # ── 4. Build STTM index for OOS ──────────────────────────────────
    print("\nСтрою STTM-индекс для OOS...")
    docs_tokens = [s.split() for s in df["preproc"]]
    theta, c_words, weeks = build_streams(
        doc_topic, docs_tokens, df["date"], vocab
    )
    del docs_tokens
    tw_lists = topic_word_lists(model)
    print(f"  Недель: {len(weeks)}, матрица слов: {c_words.shape}")

    tickers = cfg["tickers"]
    rets = load_all_tickers(ROOT / cfg["data"]["prices_dir"], tickers)

    out = {}
    for t in tickers:
        r = rets[t]
        r.index = pd.to_datetime(r.index)
        idx = sttm_expanding(
            r, theta, c_words, tw_lists, vocab, weeks,
            gamma=sttm_cfg["gamma"],
            prob_mass=sttm_cfg["prob_mass"],
            initial_train_years=2,
            norm=sttm_cfg["index_norm"],
            first_test_year=2022,
        )
        out[t] = idx
        aligned = pd.concat([idx.rename("idx"), r.rename("ret")], axis=1).dropna()
        spear = aligned["idx"].corr(aligned["ret"], method="spearman") if len(aligned) > 10 else np.nan
        print(f"  {t}: тестовых недель {len(idx)} | Spearman={spear:.3f}", flush=True)

    sttm_df = pd.DataFrame(out)
    sttm_path = ROOT / "models/sttm_indices/sttm_index_kommersant_full.parquet"
    sttm_path.parent.mkdir(parents=True, exist_ok=True)
    sttm_df.to_parquet(sttm_path)
    print(f"\nSTTM-индекс: {sttm_df.shape} → {sttm_path.name}")

    # ── 5. Backtest OOS ──────────────────────────────────────────────
    print("\nБэктест OOS 2022-2026 (Full LDA 2013-2026)...")
    sttm_df.index = pd.to_datetime(sttm_df.index)
    common_dates = sttm_df.index.intersection(rets.index)
    sttm_df = sttm_df.loc[common_dates]
    rets = rets.loc[common_dates]
    print(f"  Common dates: {len(common_dates)}")

    port = backtest(sttm_df, rets, top_pct=0.20, min_names=10,
                    rates=(0.0, 0.0005, 0.0015), mode="level")
    m = summarize(port["gross"])

    print()
    print("=" * 60)
    print("  OOS 2022-2026: OPTION B (Full LDA 2013-2026)")
    print("=" * 60)
    print(f"  Gross Sharpe:   {m['sharpe']:.3f}")
    print(f"  Ann Return:     {m['ann_return'] * 100:.1f}%")
    print(f"  Ann Vol:        {m['ann_vol'] * 100:.1f}%")
    print(f"  MaxDD:          {m['max_drawdown'] * 100:.1f}%")
    print(f"  Weeks:          {m['n']}")
    print()
    for rate in [0.0, 0.0005, 0.0015]:
        col = f"net_{rate:g}"
        if col in port.columns:
            nm = summarize(port[col])
            print(f"  net_{rate * 100:.2f}%: Sharpe={nm['sharpe']:.3f}")
    print()
    print(f"  Mean turnover:  {port['turnover'].mean():.3f}")
    print(f"  Mean n_stocks:  {port['n'].mean():.1f}")
    print(f"  Min n_stocks:   {port['n'].min()}")
    print(f"  Max n_stocks:   {port['n'].max()}")

    # ── 6. Spearman by ticker ────────────────────────────────────────
    print("\nSpearman(idx, ret) по тикерам:")
    spears = {}
    for t in tickers:
        aligned = pd.concat([sttm_df[t].rename("idx"), rets[t].rename("ret")], axis=1).dropna()
        if len(aligned) > 10:
            s = aligned["idx"].corr(aligned["ret"], method="spearman")
            spears[t] = s
    spears_sorted = sorted(spears.items(), key=lambda x: x[1], reverse=True)
    print("  TOP-10 positive:")
    for t, s in spears_sorted[:10]:
        print(f"    {t:6s}: {s:+.3f}")
    print("  BOTTOM-5 negative:")
    for t, s in spears_sorted[-5:]:
        print(f"    {t:6s}: {s:+.3f}")

    mean_spear = np.mean(list(spears.values()))
    print(f"\n  Mean Spearman:  {mean_spear:+.3f}")

    # ── 7. Compare with Option A ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("  COMPARISON: Option A vs Option B")
    print("=" * 60)
    print(f"  {'Metric':<20s} {'A (frozen)':>12s} {'B (full)':>12s}")
    print(f"  {'-'*44}")
    print(f"  {'Gross Sharpe':<20s} {'-0.318':>12s} {m['sharpe']:>12.3f}")
    print(f"  {'Ann Return':<20s} {'-8.5%':>12s} {m['ann_return']*100:>11.1f}%")
    print(f"  {'Ann Vol':<20s} {'26.8%':>12s} {m['ann_vol']*100:>11.1f}%")
    print(f"  {'MaxDD':<20s} {'-47.9%':>12s} {m['max_drawdown']*100:>11.1f}%")

    # Save results
    results = {
        "option": "B",
        "description": "LDA k=50 on full corpus 2013-2026",
        "gross_sharpe": m["sharpe"],
        "ann_return": m["ann_return"],
        "ann_vol": m["ann_vol"],
        "max_drawdown": m["max_drawdown"],
        "n_weeks": m["n"],
        "mean_turnover": port["turnover"].mean(),
        "mean_n_stocks": port["n"].mean(),
        "mean_spearman": mean_spear,
    }
    out_json = ROOT / "findings" / "oos_option_b_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nРезультаты сохранены → {out_json.name}")


if __name__ == "__main__":
    main()
