"""Обучение LDA + подбор числа тем по C_v (Этап 3).

Примеры:
    python scripts/train_topics.py --source lenta --grid 5 --limit 3000   # smoke
    python scripts/train_topics.py --source lenta                          # полный грид
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from newsalpha.topics.lda_model import (
    build_dictionary,
    coherence_cv,
    doc_topic_matrix,
    grid_search,
    ks_from_config,
    load_preproc_texts,
    train_lda,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--source", default="lenta")
    ap.add_argument("--grid", default=None, help="через запятую; по умолчанию из конфига")
    ap.add_argument("--single", type=int, default=None, help="обучить только это k")
    ap.add_argument("--passes", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    seed = cfg["seed"]
    passes = args.passes or cfg["topics"]["passes"]
    workers = max(1, (os.cpu_count() or 4) - 2)
    filt = cfg["topics"]["dictionary_filters"]

    src_dir = ROOT / cfg["paths"]["models_lda"] / args.source
    src_dir.mkdir(parents=True, exist_ok=True)
    preproc_path = Path(cfg["data"]["processed_dir"]) / f"news_{args.source}_preproc.parquet"

    print(f"загрузка {preproc_path.name}...", flush=True)
    texts = load_preproc_texts(preproc_path)
    if args.limit:
        texts = texts[: args.limit]
    dictionary = build_dictionary(texts, no_below=filt["no_below"], no_above=filt["no_above"])
    corpus = [dictionary.doc2bow(t) for t in texts]
    print(f"доков: {len(texts)} | словарь: {len(dictionary)} | passes: {passes} "
          f"| workers: {workers}", flush=True)

    if args.single:
        t0 = time.time()
        model = train_lda(corpus, dictionary, args.single, seed, passes, workers)
        c = coherence_cv(model, texts, dictionary, processes=1)
        model.save(str(src_dir / f"lda_k{args.single}.model"))
        dt = time.time() - t0
        print(f"k={args.single}: C_v={c:.4f} за {dt:.0f} c "
              f"→ полный грид из N фитов ≈ {dt * len(ks_from_config(cfg['topics']['n_topics_search'])) / 60:.0f} мин",
              flush=True)
        return

    ks = [int(x) for x in args.grid.split(",")] if args.grid else \
        ks_from_config(cfg["topics"]["n_topics_search"])
    log_path = src_dir / "coherence_log.jsonl"
    log_path.write_text("", encoding="utf-8")

    t0 = time.time()
    best_k, best_model, _ = grid_search(
        texts, dictionary, ks, seed, passes, workers, log_path
    )
    print(f"грид занял {(time.time() - t0) / 60:.1f} мин | лучшее k={best_k}", flush=True)

    dictionary.save(str(src_dir / "dictionary.dict"))
    best_model.save(str(src_dir / f"lda_k{best_k}.model"))
    (src_dir / "best.json").write_text(
        json.dumps({"best_k": best_k, "seed": seed, "passes": passes}), encoding="utf-8"
    )

    dt_path = ROOT / cfg["paths"]["doc_topic"] / f"doc_topic_{args.source}.parquet"
    dt_path.parent.mkdir(parents=True, exist_ok=True)
    doc_topic_matrix(corpus, best_model).to_parquet(dt_path, index=False)
    print(f"doc-topic матрица → {dt_path.name}", flush=True)


if __name__ == "__main__":
    main()
