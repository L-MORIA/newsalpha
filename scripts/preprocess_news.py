"""Препроцессинг новостей до preproc-колонки + idf-словарь (Этап 2).

Примеры:
    python scripts/preprocess_news.py --sources lenta --limit 2000   # smoke
    python scripts/preprocess_news.py --sources lenta,kommersant     # полный
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from newsalpha.io.kommersant import load_days_to_df
from newsalpha.text.preprocess import idf_vocab, preprocess_parallel


def resolve_n_jobs(v: int) -> int:
    cpu = os.cpu_count() or 4
    return max(1, cpu - 2) if v < 0 else min(v, cpu)


def load_source_df(name: str, cfg: dict) -> pd.DataFrame:
    if name == "kommersant":
        return load_days_to_df(Path(cfg["data"]["news_dir"]) / "kommersant")
    parts = sorted(Path(cfg["data"]["interim_dir"]).glob(f"news_{name}_part*.parquet"))
    if not parts:
        raise FileNotFoundError(f"нет interim-партов для '{name}'")
    return pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--sources", default="lenta,kommersant")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    out_dir = Path(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    n_jobs = resolve_n_jobs(args.n_jobs)

    for name in [s.strip() for s in args.sources.split(",") if s.strip()]:
        print(f"=== {name} ===", flush=True)
        df = load_source_df(name, cfg)
        if args.limit:
            df = df.head(args.limit)
        print(f"документов: {len(df)} | процессов: {n_jobs}", flush=True)

        t0 = time.time()
        df["preproc"] = preprocess_parallel(
            df["text"].tolist(), ner_glue=True, n_jobs=n_jobs
        )
        print(f"препроцессинг: {time.time() - t0:.0f} c", flush=True)

        out_suffix = f"_limit{args.limit}" if args.limit else ""
        if args.limit:
            print(
                f"ВНИМАНИЕ: smoke-режим --limit={args.limit}: боевой файл НЕ тронут, "
                f"пишу в news_{name}_preproc{out_suffix}.parquet",
                flush=True,
            )
        out_path = out_dir / f"news_{name}_preproc{out_suffix}.parquet"
        df.to_parquet(out_path, index=False)

        vocab, stats = idf_vocab(df["preproc"].tolist())
        (out_dir / f"vocab_{name}{out_suffix}.json").write_text(
            json.dumps({"stats": stats, "vocab": vocab}, ensure_ascii=False),
            encoding="utf-8",
        )
        empty = int((df["preproc"].str.len() == 0).sum())
        print(f"[{name}] сохранено {len(df)} → {out_path.name} | "
              f"словарь: {stats['words_total']} → {stats['words_kept']} | "
              f"пустых preproc: {empty}", flush=True)


if __name__ == "__main__":
    main()
