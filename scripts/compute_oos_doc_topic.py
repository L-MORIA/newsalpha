"""Вычисление doc-topic для OOS (2022–2026) по замороженной LDA 2013–2021.

Использует ту же модель и словарь, что и in-sample.
"""
import sys
from pathlib import Path

import pandas as pd
from gensim.models import LdaMulticore
from gensim.corpora import Dictionary

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    lda_dir = ROOT / "models/lda/kommersant"

    # 1. Load frozen model and dictionary
    import json
    best = json.loads((lda_dir / "best.json").read_text(encoding="utf-8"))
    best_k = int(best["best_k"])
    model = LdaMulticore.load(str(lda_dir / f"lda_k{best_k}.model"))
    dictionary = Dictionary.load(str(lda_dir / "dictionary.dict"))
    print(f"Загружена LDA k={best_k}, словарь: {len(dictionary)}")

    # 2. Load preprocessed texts
    preproc_path = ROOT / "data/processed/news_kommersant_preproc.parquet"
    df = pd.read_parquet(preproc_path)
    print(f"Всего документов: {len(df)}")

    # 3. Split by date
    df["date"] = pd.to_datetime(df["date"])
    oos_mask = df["date"] >= "2022-01-01"
    oos_df = df[oos_mask].copy()
    train_df = df[~oos_mask].copy()
    print(f"Train (2012–2021): {len(train_df)}, OOS (2022–2026): {len(oos_df)}")

    # 4. Load existing doc-topic for train
    dt_train = pd.read_parquet(ROOT / "models/doc_topic/doc_topic_kommersant.parquet")
    print(f"doc-topic train: {dt_train.shape}")

    # 5. Compute doc-topic for OOS
    oos_texts = [s.split() for s in oos_df["preproc"]]
    oos_corpus = [dictionary.doc2bow(t) for t in oos_texts]

    rows = []
    for bow in oos_corpus:
        topic_dist = model.get_document_topics(bow, minimum_probability=0.0)
        row = {t: p for t, p in topic_dist}
        rows.append(row)

    dt_oos = pd.DataFrame(rows).fillna(0.0)
    print(f"doc-topic OOS: {dt_oos.shape}")

    # 6. Merge
    dt_full = pd.concat([dt_train, dt_oos], ignore_index=True)
    print(f"doc-topic merged: {dt_full.shape}")

    # 7. Save
    out_path = ROOT / "models/doc_topic/doc_topic_kommersant_oos.parquet"
    dt_full.to_parquet(out_path, index=False)
    print(f"Сохранено → {out_path.name}")


if __name__ == "__main__":
    main()
