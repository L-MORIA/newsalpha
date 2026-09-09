"""Чекпоинт-драйвер эндогенных базлайнов: те же вызовы и параметры, что
scripts/run_baselines.py::run_endogenous (lags=5, horizon=1, min_train=104),
но по одной модели за запуск + инкрементальный summary.

Промотирован 2026-09-09 из data/logs/ (gitignored) для provenance SVM-refresh:
SVM здесь — CalibratedClassifierCV(SVC-rbf) per fix 657caa2 (sklearn-proof).

Использование:
    python scripts/run_endogenous_ckpt.py --only SVM
    python scripts/run_endogenous_ckpt.py --only RidgeClassifier
    ... (по очереди; пропуск уже готовых — по файлам сигналов)
    python scripts/run_endogenous_ckpt.py --finalize   # собрать models/baselines/endogenous_summary.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from newsalpha.baselines.endogenous import (
    build_lag_features,
    compute_baseline_metrics,
    expanding_cv_signals,
)
from newsalpha.io.market import load_all_tickers

MODEL_ORDER = [
    "RidgeClassifier",
    "LogisticRegression",
    "RandomForest",
    "GradientBoosting",
    "SVM",
]

SUMMARY_NEW = ROOT / "data" / "logs" / "endo_summary_new.json"
OUT_DIR = ROOT / "models" / "baselines"


def get_models():
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.svm import SVC

    return {
        "LogisticRegression": (LogisticRegression, {"max_iter": 500, "C": 1.0}),
        "RidgeClassifier": (RidgeClassifier, {"alpha": 1.0}),
        "RandomForest": (RandomForestClassifier, {"n_estimators": 100, "max_depth": 5, "random_state": 11}),
        "GradientBoosting": (GradientBoostingClassifier, {"n_estimators": 100, "max_depth": 3, "random_state": 11}),
        "SVM": (
            CalibratedClassifierCV,
            {"estimator": SVC(kernel="rbf", C=1.0), "ensemble": False},
        ),
    }


def load_returns():
    prices_dir = ROOT / "data" / "raw" / "prices"
    tickers = [p.name.replace("shares_TQBR_", "").replace(".csv", "")
               for p in sorted(prices_dir.glob("shares_TQBR_*.csv"))]
    return load_all_tickers(prices_dir, tickers)


def signal_path(name):
    return OUT_DIR / f"signals_{name.lower()}.parquet"


def run_one(name):
    models = get_models()
    cls, params = models[name]
    print(f"[{name}] build panel...", flush=True)
    returns = load_returns()
    panel = build_lag_features(returns, lags=5, horizon=1)
    print(f"[{name}] expanding CV fit...", flush=True)
    t0 = time.time()
    signals = expanding_cv_signals(panel, cls, params, min_train=104, lags=5)
    dt = time.time() - t0
    metrics = compute_baseline_metrics(signals, returns, 1) if not signals.empty else {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    signals.to_parquet(signal_path(name))

    def clean(v):
        return None if isinstance(v, float) and np.isnan(v) else v

    entry = {
        "accuracy": clean(metrics.get("accuracy")),
        "spearman": clean(metrics.get("spearman")),
        "n_tickers": metrics.get("n_tickers"),
        "time_sec": round(dt, 1),
    }
    prev = json.loads(SUMMARY_NEW.read_text(encoding="utf-8")) if SUMMARY_NEW.exists() else {}
    prev[name] = entry
    SUMMARY_NEW.write_text(json.dumps(prev, indent=2), encoding="utf-8")
    print(f"[{name}] Acc={entry['accuracy']:.3f} Spearman={entry['spearman']:.3f} "
          f"n={entry['n_tickers']} time={dt:.0f}s -> {signal_path(name).name}", flush=True)


def finalize():
    prev = json.loads(SUMMARY_NEW.read_text(encoding="utf-8"))
    missing = [m for m in MODEL_ORDER if m not in prev]
    if missing:
        print(f"FINALIZE REFUSED: missing {missing}")
        sys.exit(1)
    # MERGE, а не перезапись: закоммиченный summary — склейка из нескольких
    # прогонов (ключи SVM-RBF/SESTM-*/STTM-*, которых run_endogenous не пишет).
    # Обновляем только 5 моделей текущего скрипта, остальное сохраняем.
    dest = OUT_DIR / "endogenous_summary.json"
    base = json.loads(dest.read_text(encoding="utf-8")) if dest.exists() else {}
    for m in MODEL_ORDER:
        base[m] = prev[m]
    dest.write_text(json.dumps(base, indent=2), encoding="utf-8")
    print(f"finalized {dest} (merged {len(MODEL_ORDER)} models, kept extra keys)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(get_models()), default=None)
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()
    if args.finalize:
        finalize()
        return
    if args.only:
        if signal_path(args.only).exists() and SUMMARY_NEW.exists() \
                and args.only in json.loads(SUMMARY_NEW.read_text(encoding="utf-8")):
            print(f"[{args.only}] already done, skip")
            return
        run_one(args.only)
        return
    for m in MODEL_ORDER:
        if not (signal_path(m).exists() and SUMMARY_NEW.exists()
                and m in json.loads(SUMMARY_NEW.read_text(encoding="utf-8"))):
            run_one(m)


if __name__ == "__main__":
    main()
