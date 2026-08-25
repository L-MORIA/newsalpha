"""LDA-моделирование и подбор числа тем по coherence C_v (Этап 3).

Соответствие статье STTM: грид n_topics 2..50 шаг 3, ожидаемый оптимум ~20 тем,
фиксированный seed для воспроизводимости. Словарь чистится фильтрами
no_below/no_above (краевые мусорные токены препроцессинга).
"""
import json
from pathlib import Path

import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaMulticore


def load_preproc_texts(path: str | Path) -> list[list[str]]:
    """preproc-колонка parquet → список токен-списков."""
    df = pd.read_parquet(path, columns=["preproc"])
    return [s.split() for s in df["preproc"]]


def build_dictionary(texts: list[list[str]], no_below: int, no_above: float) -> Dictionary:
    d = Dictionary(texts)
    d.filter_extremes(no_below=no_below, no_above=no_above, keep_n=None)
    d.compactify()
    return d


def train_lda(
    corpus: list[list[tuple[int, int]]],
    dictionary: Dictionary,
    n_topics: int,
    seed: int,
    passes: int,
    workers: int,
) -> LdaMulticore:
    return LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=n_topics,
        random_state=seed,
        passes=passes,
        workers=workers,
        iterations=50,
        chunksize=2000,
    )


def coherence_cv(
    model: LdaMulticore,
    texts: list[list[str]],
    dictionary: Dictionary,
    processes: int = 1,
) -> float:
    cm = CoherenceModel(
        model=model,
        texts=texts,
        dictionary=dictionary,
        coherence="c_v",
        processes=processes,
    )
    return float(cm.get_coherence())


def grid_search(
    texts: list[list[str]],
    dictionary: Dictionary,
    ks: list[int],
    seed: int,
    passes: int,
    workers: int,
    log_path: str | Path,
) -> tuple[int, LdaMulticore, list[dict]]:
    """Фит по гриду k; прогресс в JSONL; возврат (лучшее k, модель, лог)."""
    corpus = [dictionary.doc2bow(t) for t in texts]
    results: list[dict] = []
    best_k, best_model, best_c = None, None, -1.0
    for k in ks:
        model = train_lda(corpus, dictionary, k, seed, passes, workers)
        c = coherence_cv(model, texts, dictionary)
        results.append({"n_topics": k, "coherence_cv": c})
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(results[-1]) + "\n")
        print(f"k={k}: C_v={c:.4f}", flush=True)
        if c > best_c:
            best_k, best_model, best_c = k, model, c
    return best_k, best_model, results


def doc_topic_matrix(
    corpus: list[list[tuple[int, int]]], model: LdaMulticore, minimum_probability: float = 0.0
) -> pd.DataFrame:
    rows = []
    for bow in corpus:
        dense = [0.0] * model.num_topics
        for tid, p in model.get_document_topics(bow, minimum_probability=minimum_probability):
            dense[tid] = p
        rows.append(dense)
    cols = [f"topic_{i}" for i in range(model.num_topics)]
    return pd.DataFrame(rows, columns=cols)


def ks_from_config(cfg_search: dict) -> list[int]:
    return list(range(cfg_search["start"], cfg_search["stop"] + 1, cfg_search["step"]))
