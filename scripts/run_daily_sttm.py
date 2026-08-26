"""Daily Frequency STTM: пересчёт на дневных данных.

Гипотеза: эффект новостей длится часы, а не недели.
Если метод работает — он должен работать лучше на дневных данных.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from newsalpha.io.market import load_all_tickers_daily
from newsalpha.sttm.pipeline import (
    build_streams_daily,
    sttm_expanding_daily,
    topic_word_lists,
)
from newsalpha.backtest.portfolio import backtest, summarize
from gensim.models import LdaMulticore
from gensim.corpora import Dictionary


def main():
    cfg = yaml.safe_load(open(ROOT / "config/default.yaml", encoding="utf-8"))
    sttm_cfg = cfg["sttm"]

    # ── 1. Load LDA model (reuse from in-sample) ─────────────────────
    lda_dir = ROOT / "models/lda/kommersant"
    import json
    best = json.loads((lda_dir / "best.json").read_text(encoding="utf-8"))
    best_k = int(best["best_k"])
    model = LdaMulticore.load(str(lda_dir / f"lda_k{best_k}.model"))
    dictionary = Dictionary.load(str(lda_dir / "dictionary.dict"))
    vocab = dictionary.token2id
    print(f"LDA k={best_k}, словарь: {len(vocab)}")

    # ── 2. Load preprocessed texts ───────────────────────────────────
    preproc_path = ROOT / cfg["data"]["processed_dir"] / "news_kommersant_preproc.parquet"
    df = pd.read_parquet(preproc_path)
    df["date"] = pd.to_datetime(df["date"])
    print(f"Документов: {len(df)}, период: {df['date'].min().date()} – {df['date'].max().date()}")

    # ── 3. Load doc-topic (reuse existing) ───────────────────────────
    dt_path = ROOT / "models/doc_topic/doc_topic_kommersant_oos.parquet"
    doc_topic = pd.read_parquet(dt_path).to_numpy()
    print(f"doc-topic: {doc_topic.shape}")

    # ── 4. Build DAILY streams ───────────────────────────────────────
    print("\nСтрою дневные потоки...")
    t0 = time.time()
    docs_tokens = [s.split() for s in df["preproc"]]
    theta, c_words, days = build_streams_daily(
        doc_topic, docs_tokens, df["date"], vocab
    )
    del docs_tokens
    print(f"  Дней: {len(days)}, theta: {theta.shape}, c_words: {c_words.shape}")
    print(f"  Время: {time.time() - t0:.1f} сек")

    # ── 5. Load daily prices ─────────────────────────────────────────
    print("\nЗагружаю дневные цены...")
    tickers = cfg["tickers"]
    rets = load_all_tickers_daily(ROOT / cfg["data"]["prices_dir"], tickers)
    print(f"Daily returns: {rets.shape}")
    print(f"Период: {rets.index.min().date()} – {rets.index.max().date()}")

    # ── 6. Build daily STTM index ────────────────────────────────────
    print("\nСтрою STTM-индекс (дневная частота)...")
    tw_lists = topic_word_lists(model)

    out = {}
    for t in tickers:
        r = rets[t]
        r.index = pd.to_datetime(r.index)
        idx = sttm_expanding_daily(
            r, theta, c_words, tw_lists, vocab, days,
            gamma=sttm_cfg["gamma"],
            prob_mass=sttm_cfg["prob_mass"],
            initial_train_years=2,
            norm=sttm_cfg["index_norm"],
            first_test_year=2015,
        )
        out[t] = idx
        aligned = pd.concat([idx.rename("idx"), r.rename("ret")], axis=1).dropna()
        spear = aligned["idx"].corr(aligned["ret"], method="spearman") if len(aligned) > 10 else np.nan
        print(f"  {t}: дней {len(idx)} | Spearman={spear:.3f}", flush=True)

    sttm_df = pd.DataFrame(out)
    print(f"\nSTTM-индекс: {sttm_df.shape}")

    sttm_df.index = pd.to_datetime(sttm_df.index)
    rets.index = pd.to_datetime(rets.index)

    # ── 7. Backtest (in-sample: 2015-2021) ───────────────────────────
    print("\n" + "=" * 60)
    print("  BACKTEST IN-SAMPLE (2015-2021), DAILY FREQUENCY")
    print("=" * 60)

    is_mask = sttm_df.index.year <= 2021
    sttm_is = sttm_df.loc[is_mask]
    rets_is = rets.loc[rets.index.year <= 2021]

    common_is = sttm_is.index.intersection(rets_is.index)
    sttm_is = sttm_is.loc[common_is]
    rets_is = rets_is.loc[common_is]
    print(f"  Common dates: {len(common_is)}")

    port_is = backtest(sttm_is, rets_is, top_pct=0.20, min_names=10,
                       rates=(0.0, 0.0005, 0.0015), mode="level")
    m_is = summarize(port_is["gross"])

    print(f"\n  Gross Sharpe:   {m_is['sharpe']:.3f}")
    print(f"  Ann Return:     {m_is['ann_return'] * 100:.1f}%")
    print(f"  Ann Vol:        {m_is['ann_vol'] * 100:.1f}%")
    print(f"  MaxDD:          {m_is['max_drawdown'] * 100:.1f}%")
    print(f"  Days:           {m_is['n']}")
    print(f"  Mean turnover:  {port_is['turnover'].mean():.3f}")
    print(f"  Mean n_stocks:  {port_is['n'].mean():.1f}")

    # ── 8. Backtest (OOS: 2022-2026) ─────────────────────────────────
    print("\n" + "=" * 60)
    print("  BACKTEST OOS (2022-2026), DAILY FREQUENCY")
    print("=" * 60)

    oos_mask = sttm_df.index.year >= 2022
    sttm_oos = sttm_df.loc[oos_mask]
    rets_oos = rets.loc[rets.index.year >= 2022]

    common_oos = sttm_oos.index.intersection(rets_oos.index)
    sttm_oos = sttm_oos.loc[common_oos]
    rets_oos = rets_oos.loc[common_oos]
    print(f"  Common dates: {len(common_oos)}")

    port_oos = backtest(sttm_oos, rets_oos, top_pct=0.20, min_names=10,
                        rates=(0.0, 0.0005, 0.0015), mode="level")
    m_oos = summarize(port_oos["gross"])

    print(f"\n  Gross Sharpe:   {m_oos['sharpe']:.3f}")
    print(f"  Ann Return:     {m_oos['ann_return'] * 100:.1f}%")
    print(f"  Ann Vol:        {m_oos['ann_vol'] * 100:.1f}%")
    print(f"  MaxDD:          {m_oos['max_drawdown'] * 100:.1f}%")
    print(f"  Days:           {m_oos['n']}")
    print(f"  Mean turnover:  {port_oos['turnover'].mean():.3f}")
    print(f"  Mean n_stocks:  {port_oos['n'].mean():.1f}")

    # ── 9. Comparison ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  COMPARISON: DAILY vs WEEKLY")
    print("=" * 60)
    print(f"  {'Metric':<20s} {'Weekly IS':>12s} {'Daily IS':>12s} {'Daily OOS':>12s}")
    print(f"  {'-'*56}")
    print(f"  {'Sharpe':<20s} {'+1.42':>12s} {m_is['sharpe']:>12.3f} {m_oos['sharpe']:>12.3f}")
    print(f"  {'Ann Return':<20s} {'+25.4%':>12s} {m_is['ann_return']*100:>11.1f}% {m_oos['ann_return']*100:>11.1f}%")
    print(f"  {'Ann Vol':<20s} {'17.9%':>12s} {m_is['ann_vol']*100:>11.1f}% {m_oos['ann_vol']*100:>11.1f}%")
    print(f"  {'MaxDD':<20s} {'-22.0%':>12s} {m_is['max_drawdown']*100:>11.1f}% {m_oos['max_drawdown']*100:>11.1f}%")

    # ── 10. Direction accuracy ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DIRECTION ACCURACY (daily)")
    print("=" * 60)

    all_dir = []
    for t in tickers:
        aligned = pd.concat([pd.Series(sttm_df[t].values, index=pd.to_datetime(sttm_df.index), name="idx"),
                             pd.Series(rets[t].values, index=pd.to_datetime(rets.index), name="ret")], axis=1).dropna()
        if len(aligned) > 10:
            correct = (np.sign(aligned["idx"]) == np.sign(aligned["ret"])).mean()
            all_dir.append(correct)

    mean_dir = np.mean(all_dir)
    print(f"  Mean Direction Accuracy: {mean_dir*100:.1f}%")
    print(f"  RANDOM baseline:         50.0%")
    print(f"  Improvement:             {(mean_dir-0.5)*100:+.1f}pp")

    # ── 11. Statistical significance ─────────────────────────────────
    from scipy import stats
    spears = []
    for t in tickers:
        aligned = pd.concat([pd.Series(sttm_df[t].values, index=pd.to_datetime(sttm_df.index), name="idx"),
                             pd.Series(rets[t].values, index=pd.to_datetime(rets.index), name="ret")], axis=1).dropna()
        if len(aligned) > 10:
            s = aligned["idx"].corr(aligned["ret"], method="spearman")
            spears.append(s)
    spears = np.array(spears)
    t_stat, p_value = stats.ttest_1samp(spears, 0.0)
    print(f"\n  Mean Spearman:  {spears.mean():+.4f}")
    print(f"  t-test p-value: {p_value:.4f}")
    print(f"  Significant:    {'YES' if p_value < 0.05 else 'NO'}")


if __name__ == "__main__":
    main()
