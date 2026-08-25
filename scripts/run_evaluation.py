"""Прогон полной оценки STTM-индекса (Этап 6).

Использование:
    python scripts/run_evaluation.py --source lenta
    python scripts/run_evaluation.py --source lenta --placebo-sims 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import yaml

from newsalpha.evaluation.granger import evaluate_granger
from newsalpha.evaluation.direction import direction_table
from newsalpha.evaluation.fdr import fdr_analysis
from newsalpha.evaluation.sensitivity import sensitivity_grid, plateau_diagnosis
from newsalpha.sttm.pipeline import build_streams, topic_word_lists


def load_config(path: str = "config/default.yaml") -> dict:
    with open(ROOT / path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="STTM evaluation (Etape 6)")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--source", default="lenta")
    parser.add_argument("--placebo-sims", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--prob-mass", type=float, default=0.3)
    args = parser.parse_args()

    cfg = load_config(args.config)
    source = args.source
    print(f"=== Evaluation: {source} ===\n")

    # --- Load artifacts ---
    idx_path = ROOT / "models" / "sttm_indices" / f"sttm_index_{source}.parquet"
    if not idx_path.exists():
        print(f"ERROR: {idx_path} not found. Run build_sttm_index.py first.")
        sys.exit(1)

    sttm_idx = pd.read_parquet(idx_path)
    print(f"STTM index: {sttm_idx.shape} (weeks x tickers)")

    # --- Load returns ---
    from newsalpha.io.market import load_all_tickers
    tickers_list = cfg["tickers"]
    wret = load_all_tickers(str(ROOT / cfg["data"]["prices_dir"]), tickers_list)
    wret.index = pd.DatetimeIndex(wret.index)
    # Align
    common_tickers = sttm_idx.columns.intersection(wret.columns)
    sttm_idx = sttm_idx[common_tickers]
    wret = wret[common_tickers]

    # --- Stock index (aggregate across tickers) ---
    stock_idx = sttm_idx.mean(axis=1)
    stock_ret = wret.mean(axis=1)

    # --- 1. Granger ---
    print("\n--- 1. Granger Causality ---")
    gr = evaluate_granger(stock_idx, stock_ret, maxlag=5)
    print(f"ADF index: p={gr['adf_index']['pvalue']:.4f} ({'stationary' if gr['adf_index']['stationary'] else 'non-stationary'})")
    print(f"ADF returns: p={gr['adf_returns']['pvalue']:.4f} ({'stationary' if gr['adf_returns']['stationary'] else 'non-stationary'})")
    for lag, v in gr["granger"]["results"].items():
        print(f"  lag {lag}: F={v['f_stat']:.3f}, p={v['pvalue']:.4f}")
    print(f"  => {gr['conclusion']}")

    # --- 2. Direction metrics ---
    print("\n--- 2. Direction Metrics ---")
    dt = direction_table(stock_idx, stock_ret)
    for _, row in dt.iterrows():
        print(f"  h={int(row['horizon'])}w: Acc={row['accuracy']:.3f}, "
              f"F1={row['f1']:.3f}, AUC={row['auc']:.3f}, "
              f"ρ={row['spearman_rho']:.3f} (p={row['spearman_pvalue']:.4f})")

    # --- 3. FDR per ticker ---
    print("\n--- 3. FDR (Benjamini-Hochberg) ---")
    fdr = fdr_analysis(sttm_idx, wret)
    print(f"  {fdr['conclusion']}")
    per_t = fdr["per_ticker"].sort_values("pvalue")
    for _, row in per_t.head(5).iterrows():
        print(f"  {row['ticker']}: ρ={row['rho']:.3f}, p={row['pvalue']:.4f}")

    # --- Load streams (shared by sensitivity + placebo) ---
    lda_path = ROOT / "models" / "lda" / source
    preproc_path = Path(cfg["data"]["processed_dir"]) / f"news_{source}_preproc.parquet"
    doc_topic_path = ROOT / cfg["paths"]["doc_topic"] / f"doc_topic_{source}.parquet"
    theta = c_words = tw = vocab = cal_weeks = None

    if ((lda_path / "lda_k32.model").exists()
            and preproc_path.exists()
            and doc_topic_path.exists()):
        from gensim.models import LdaMulticore
        from gensim.corpora import Dictionary
        lda = LdaMulticore.load(str(lda_path / "lda_k32.model"))
        dictionary = Dictionary.load(str(lda_path / "dictionary.dict"))
        vocab = dictionary.token2id
        tw = topic_word_lists(lda, topn=40)
        df_pre = pd.read_parquet(preproc_path, columns=["date", "preproc"])
        doc_topic = pd.read_parquet(doc_topic_path).to_numpy()
        docs_tokens = [s.split() for s in df_pre["preproc"]]
        theta, c_words, cal_weeks = build_streams(doc_topic, docs_tokens, df_pre["date"], vocab)
        del docs_tokens

    # --- 4. Sensitivity grid ---
    print("\n--- 4. Sensitivity Grid ---")
    if theta is not None:
        cost_val = cfg.get("strategy", {}).get("commission_scenarios", [0.0005])[0]
        grid = sensitivity_grid(
            wret, theta, c_words, vocab, cal_weeks,
            topic_word_lists_override=tw,
            gammas=[0.01, 0.03, 0.05, 0.08, 0.10],
            prob_masses=[0.15, 0.20, 0.30, 0.40],
            first_test_year=2015,
            cost=cost_val,
            initial_train_years=cfg.get("evaluation", {}).get("initial_train_years", 2),
            index_norm=cfg.get("sttm", {}).get("index_norm", "sigmoid"),
        )
        diag = plateau_diagnosis(grid)
        print(f"  Peak gross Sharpe: {diag.get('peak_gross_sharpe', 'N/A')}")
        print(f"  Median gross Sharpe: {diag.get('median_gross_sharpe', 'N/A')}")
        print(f"  Q25: {diag.get('q25_gross_sharpe', 'N/A')}")
        print(f"  => {diag['verdict']}")
        top5 = grid.nlargest(5, "gross_sharpe", keep="first")
        print("  Top-5 configs:")
        for _, row in top5.iterrows():
            print(f"    g={row['gamma']:.2f}, pm={row['prob_mass']:.2f} "
                  f"-> gross={row['gross_sharpe']:.3f} "
                  f"net={row['net_sharpe']:.3f} "
                  f"frac_rho+={row.get('frac_tickers_positive_rho', 0):.0%}")
        (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
        grid.to_csv(ROOT / "data" / "processed" / f"sensitivity_grid_{source}.csv", index=False)
    else:
        print("  Skipped (LDA model or preproc data not found)")

    # --- 5. Placebo ---
    print(f"\n--- 5. Placebo ({args.placebo_sims} sims) ---")
    if theta is not None:
        from newsalpha.evaluation.placebo import placebo_test
        pbt = placebo_test(
            wret, theta, c_words, tw, vocab, cal_weeks,
            n_sims=args.placebo_sims,
            gamma=args.gamma,
            prob_mass=args.prob_mass,
            first_test_year=2015,
            initial_train_years=cfg.get("evaluation", {}).get("initial_train_years", 2),
            top_pct=0.20,
            cost=cost_val,
            index_norm=cfg.get("sttm", {}).get("index_norm", "sigmoid"),
        )
        print(f"  Real Sharpe: {pbt['real_sharpe']:.3f}")
        print(f"  Placebo mean: {pbt['placebo_mean']:.3f} ± {pbt['placebo_std']:.3f}")
        print(f"  z-score: {pbt['z_score']:.2f}, p-value: {pbt['p_value']:.4f}")
        print(f"  => {pbt['conclusion']}")
    else:
        print("  Skipped (LDA streams not loaded)")

    # --- Save report ---
    report = {
        "source": source,
        "granger": {k: v for k, v in gr.items() if k != "results"},
        "direction": dt.to_dict(orient="records"),
        "fdr": fdr["fdr"],
    }
    report_path = ROOT / "data" / "processed" / f"evaluation_{source}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n=== Report saved: {report_path} ===")


if __name__ == "__main__":
    main()
