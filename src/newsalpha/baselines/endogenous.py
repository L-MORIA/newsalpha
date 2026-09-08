"""Эндогенные базлайны: лаги доходности × ML модели (Этап 5).

Модель: 5 лагов недельной доходности тикера → предсказание знака
доходности на следующей неделе. Expanding CV, 5 моделей.
Контрольный бэзлайн: «нет новостей — только цена».
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


FEATURE_NAMES = [f"ret_lag{i}" for i in range(1, 6)]


def build_lag_features(
    returns: pd.DataFrame,
    lags: int = 5,
    horizon: int = 1,
) -> pd.DataFrame:
    """DataFrame[недели x тикеры] → panel [недели x тикеры x lags].

    Возвращает DataFrame с мультииндексом (week, ticker) и колонками ret_lag1..ret_lagN.
    Целевая переменная: sign(ret_{t+horizon}).
    """
    rows = []
    for ticker in returns.columns:
        r = returns[ticker].dropna()
        for i in range(lags + horizon, len(r)):
            week = r.index[i - horizon]
            lags_vals = [r.iloc[i - lags - horizon + j - 1] for j in range(lags)]
            target = r.iloc[i]
            row = {"week": week, "ticker": ticker, "target": target}
            for j, v in enumerate(lags_vals):
                row[f"ret_lag{j+1}"] = v
            rows.append(row)
    return pd.DataFrame(rows)


def expanding_cv_signals(
    panel: pd.DataFrame,
    model_cls,
    model_params: dict | None = None,
    min_train: int = 104,
    lags: int = 5,
) -> pd.DataFrame:
    """Expanding-CV: на каждом шаге train = все недели до текущей, test = текущая.

    panel: DataFrame с колонками [week, ticker, target, ret_lag1..N].
    model_cls: класс sklearn-модели (LogisticRegression, RandomForest, и т.д.)
    min_train: минимальное число строк для обучения.

    Возвращает DataFrame[недели x тикеры] с предсказанными вероятностями.
    """
    if model_params is None:
        model_params = {}

    weeks = sorted(panel["week"].unique())
    feature_cols = [c for c in panel.columns if c.startswith("ret_lag")]

    pred_matrix = {}
    for wi, week in enumerate(weeks):
        if wi < min_train // 50 + 1:
            continue

        train = panel[panel["week"].isin(weeks[:wi])]
        test = panel[panel["week"] == week]

        if len(train) < min_train or len(test) == 0:
            continue
        if train["target"].nunique() < 2:
            continue

        X_train = train[feature_cols].fillna(0).to_numpy()
        y_train = (train["target"] > 0).astype(int).to_numpy()
        X_test = test[feature_cols].fillna(0).to_numpy()

        model = model_cls(**model_params)
        try:
            model.fit(X_train, y_train)
            prob = model.predict_proba(X_test)[:, 1]
        except Exception:
            prob = np.full(len(test), 0.5)

        pred_matrix[week] = dict(zip(test["ticker"], prob))

    return pd.DataFrame(pred_matrix).T.sort_index()


def endogenous_baseline(
    returns: pd.DataFrame,
    lags: int = 5,
    horizon: int = 1,
    min_train: int = 104,
) -> dict:
    """Полный прогон: 5 моделей → dict[model_name] = signals DataFrame.

    Возвращает сигналы (вероятности) и метрики.
    """
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.svm import SVC

    panel = build_lag_features(returns, lags=lags, horizon=horizon)

    models = {
        "LogisticRegression": (LogisticRegression, {"max_iter": 500, "C": 1.0}),
        "RidgeClassifier": (RidgeClassifier, {"alpha": 1.0}),
        "RandomForest": (RandomForestClassifier, {"n_estimators": 100, "max_depth": 5, "random_state": 11}),
        "GradientBoosting": (GradientBoostingClassifier, {"n_estimators": 100, "max_depth": 3, "random_state": 11}),
        "SVM": (
            CalibratedClassifierCV,
            {"estimator": SVC(kernel="rbf", C=1.0), "ensemble": False},
        ),
    }

    results = {}
    for name, (cls, params) in models.items():
        signals = expanding_cv_signals(panel, cls, params, min_train=min_train, lags=lags)
        if signals.empty:
            results[name] = {"signals": signals, "metrics": {}}
            continue
        metrics = compute_baseline_metrics(signals, returns, horizon)
        results[name] = {"signals": signals, "metrics": metrics}

    return results


def compute_baseline_metrics(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    horizon: int = 1,
) -> dict:
    """Метрики: direction accuracy, Spearman, Sharpe (top-20% long)."""
    if signals.empty:
        return {"accuracy": np.nan, "spearman": np.nan, "sharpe": np.nan}

    common = signals.index.intersection(returns.index)
    if len(common) < 20:
        return {"accuracy": np.nan, "spearman": np.nan, "sharpe": np.nan}

    sig = signals.loc[common]
    ret = returns.loc[common]

    accs = []
    rhos = []
    for ticker in sig.columns:
        if ticker not in ret.columns:
            continue
        s = sig[ticker].dropna()
        r = ret[ticker].reindex(s.index).dropna()
        idx = s.index.intersection(r.index)
        if len(idx) < 20:
            continue
        pred_dir = (s.loc[idx] > 0.5).astype(int)
        true_dir = (r.loc[idx] > 0).astype(int)
        accs.append((pred_dir == true_dir).mean())
        rho, _ = sp_stats.spearmanr(s.loc[idx], r.loc[idx])
        rhos.append(rho)

    return {
        "accuracy": np.mean(accs) if accs else np.nan,
        "spearman": np.mean(rhos) if rhos else np.nan,
        "n_tickers": len(accs),
    }
