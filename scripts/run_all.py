"""Полный пайплайн STTM одной командой (Этап 8).

Использование:
    python scripts/run_all.py --source lenta
    python scripts/run_all.py --source kommersant
    python scripts/run_all.py --source lenta --skip-preproc  # если данные уже обработаны

Порядок:
    1. preprocess_news   — лемматизация/NER → news_{src}_preproc.parquet
    2. train_topics      — LDA + грид по C_v → models/lda/{src}/
    3. build_sttm_index  — per-ticker STTM → models/sttm_indices/sttm_index_{src}.parquet
    4. run_backtest      — кросс-секционный backtest → models/backtest/
    5. run_evaluation    — Granger + Direction + FDR + Sensitivity

Все стадии используют один seed из config/default.yaml.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_stage(name: str, cmd: list[str], skip: bool = False) -> bool:
    """Запуск одной стадии пайплайна. Возвращает True при успехе."""
    if skip:
        print(f"\n{'='*60}\n  SKIP: {name}\n{'='*60}", flush=True)
        return True

    print(f"\n{'='*60}\n  START: {name}\n{'='*60}", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=False)
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n  OK: {name} ({elapsed:.0f}s)", flush=True)
        return True
    else:
        print(f"\n  FAILED: {name} (exit {result.returncode}, {elapsed:.0f}s)", flush=True)
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--source", default="lenta",
                    help="lenta | kommersant")
    ap.add_argument("--skip-preproc", action="store_true",
                    help="пропустить preprocess_news (если данные уже обработаны)")
    ap.add_argument("--skip-topics", action="store_true",
                    help="пропустить train_topics (если LDA уже обучена)")
    ap.add_argument("--skip-index", action="store_true",
                    help="пропустить build_sttm_index")
    ap.add_argument("--skip-backtest", action="store_true",
                    help="пропустить run_backtest")
    ap.add_argument("--skip-eval", action="store_true",
                    help="пропустить run_evaluation")
    args = ap.parse_args()

    python = sys.executable
    src = args.source
    cfg = args.config

    print(f"{'#'*60}")
    print(f"  newsalpha — полный пайплайн STTM")
    print(f"  source: {src} | config: {cfg}")
    print(f"  seed: проверьте config/default.yaml → seed")
    print(f"{'#'*60}")

    t_start = time.time()
    stages = [
        ("preprocess_news", [python, "scripts/preprocess_news.py",
                             "--config", cfg, "--source", src], args.skip_preproc),
        ("train_topics", [python, "scripts/train_topics.py",
                          "--config", cfg, "--source", src], args.skip_topics),
        ("build_sttm_index", [python, "scripts/build_sttm_index.py",
                              "--config", cfg, "--source", src], args.skip_index),
        ("run_backtest", [python, "scripts/run_backtest.py",
                          "--config", cfg, "--source", src], args.skip_backtest),
        ("run_evaluation", [python, "scripts/run_evaluation.py",
                            "--config", cfg, "--source", src], args.skip_eval),
    ]

    results = {}
    for name, cmd, skip in stages:
        ok = run_stage(name, cmd, skip)
        results[name] = ok
        if not ok:
            print(f"\n  ПАЙПЛАЙН ОСТАНОВЛЕН на {name}")
            break

    elapsed_total = time.time() - t_start
    print(f"\n{'#'*60}")
    print(f"  ИТОГИ ({elapsed_total:.0f}s):")
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"    {name}: {status}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
