"""STTM-индексы для OOS (2022–2026) по замороженной LDA 2013–2021.

Использует merged doc-topic (train+OOS), first_test_year=2022.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from gensim.models import LdaMulticore
from gensim.corpora import Dictionary

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from newsalpha.io.market import load_all_tickers
from newsalpha.sttm.pipeline import (
    build_streams,
    sttm_expanding,
    topic_word_lists,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--tickers", default=None)
    ap.add_argument("--first-test-year", type=int, default=2022)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    sttm_cfg = cfg["sttm"]

    lda_dir = ROOT / cfg["paths"]["models_lda"] / "kommersant"
    best = json.loads((lda_dir / "best.json").read_text(encoding="utf-8"))
    best_k = int(best["best_k"])
    model = LdaMulticore.load(str(lda_dir / f"lda_k{best_k}.model"))
    dictionary = Dictionary.load(str(lda_dir / "dictionary.dict"))
    vocab = dictionary.token2id

    preproc_path = Path(cfg["data"]["processed_dir"]) / "news_kommersant_preproc.parquet"
    df = pd.read_parquet(preproc_path, columns=["date", "preproc"])

    dt_path = ROOT / cfg["paths"]["doc_topic"] / "doc_topic_kommersant_oos.parquet"
    doc_topic = pd.read_parquet(dt_path).to_numpy()
    print(f"доков: {len(df)} | тем: {doc_topic.shape[1]}", flush=True)

    docs_tokens = [s.split() for s in df["preproc"]]
    theta, c_words, weeks = build_streams(doc_topic, docs_tokens, df["date"], vocab)
    del docs_tokens
    tw_lists = topic_word_lists(model)
    print(f"недель: {len(weeks)} | матрица слов: {c_words.shape}", flush=True)

    tickers = args.tickers.split(",") if args.tickers else cfg["tickers"]
    rets = load_all_tickers(cfg["data"]["prices_dir"], tickers)

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
            first_test_year=args.first_test_year,
        )
        out[t] = idx
        aligned = pd.concat([idx.rename("idx"), r.rename("ret")], axis=1).dropna()
        spear = aligned["idx"].corr(aligned["ret"], method="spearman") if len(aligned) > 10 else np.nan
        print(f"{t}: тестовых недель {len(idx)} | Spearman(idx, ret)={spear:.3f}",
              flush=True)

    res = pd.DataFrame(out)
    out_path = ROOT / cfg["paths"]["sttm_indices"] / "sttm_index_kommersant_oos.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(out_path)
    print(f"сохранено → {out_path.name}: {res.shape}", flush=True)


if __name__ == "__main__":
    main()
