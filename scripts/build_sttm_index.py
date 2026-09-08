"""STTM-индексы по всем тикерам (Этап 4).

Примеры:
    python scripts/build_sttm_index.py --tickers SBER,GAZP   # smoke
    python scripts/build_sttm_index.py                        # все из конфига
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from gensim.models import LdaMulticore

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
    ap.add_argument("--source", default="lenta")
    ap.add_argument("--tickers", default=None, help="через запятую; по умолчанию все")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    sttm_cfg = cfg["sttm"]
    eval_cfg = cfg["evaluation"]

    lda_dir = ROOT / cfg["paths"]["models_lda"] / args.source
    best = json.loads((lda_dir / "best.json").read_text(encoding="utf-8"))
    model = LdaMulticore.load(str(lda_dir / f"lda_k{best['best_k']}.model"))
    from gensim.corpora import Dictionary

    dictionary = Dictionary.load(str(lda_dir / "dictionary.dict"))
    vocab = dictionary.token2id

    preproc_path = Path(cfg["data"]["processed_dir"]) / f"news_{args.source}_preproc.parquet"
    df = pd.read_parquet(preproc_path, columns=["date", "preproc"])
    doc_topic = pd.read_parquet(
        ROOT / cfg["paths"]["doc_topic"] / f"doc_topic_{args.source}.parquet"
    ).to_numpy()
    print(f"доков: {len(df)} | тем: {doc_topic.shape[1]}", flush=True)

    docs_tokens = [s.split() for s in df["preproc"]]
    theta, c_words, weeks = build_streams(doc_topic, docs_tokens, df["date"], vocab)
    del docs_tokens
    tw_lists = topic_word_lists(model)
    print(f"недель: {len(weeks)} | матрица слов: {c_words.shape}", flush=True)

    tickers = args.tickers.split(",") if args.tickers else cfg["tickers"]
    rets = load_all_tickers(cfg["data"]["prices_dir"], tickers)

    out = {}
    # глобальный календарь протокола статьи: protocol_start + initial_train_years
    data_start_year = int(pd.Timestamp(eval_cfg["protocol_start"]).year)
    fty_global = data_start_year + int(eval_cfg["initial_train_years"])
    for t in tickers:
        r = rets[t]
        r.index = pd.to_datetime(r.index)
        idx = sttm_expanding(
            r, theta, c_words, tw_lists, vocab, weeks,
            gamma=sttm_cfg["gamma"],
            prob_mass=sttm_cfg["prob_mass"],
            initial_train_years=eval_cfg["initial_train_years"],
            norm=sttm_cfg["index_norm"],
            first_test_year=fty_global,
        )
        out[t] = idx
        aligned = pd.concat([idx.rename("idx"), r.rename("ret")], axis=1, sort=False).dropna()
        spear = aligned["idx"].corr(aligned["ret"], method="spearman") if len(aligned) > 10 else np.nan
        print(f"{t}: тестовых недель {len(idx)} | Spearman(idx, ret)={spear:.3f}",
              flush=True)

    res = pd.DataFrame(out)
    out_path = ROOT / cfg["paths"]["sttm_indices"] / f"sttm_index_{args.source}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(out_path)
    print(f"сохранено → {out_path.name}: {res.shape}", flush=True)


if __name__ == "__main__":
    main()
