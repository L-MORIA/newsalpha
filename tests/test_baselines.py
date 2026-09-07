"""Тесты эндогенных базлайнов (Этап 5)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from newsalpha.baselines.endogenous import (
    build_lag_features,
    compute_baseline_metrics,
)
from newsalpha.baselines.sestm import (
    compute_p_score,
    screening,
    sestm_expanding,
    supervised_topic_estimate,
)


# ── endogenous ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_returns():
    np.random.seed(42)
    dates = pd.date_range("2015-01-05", periods=200, freq="W-MON")
    tickers = ["SBER", "GAZP"]
    data = np.random.randn(200, 2) * 0.02
    return pd.DataFrame(data, index=dates, columns=tickers)


def test_build_lag_features_shapes(sample_returns):
    panel = build_lag_features(sample_returns, lags=5, horizon=1)
    assert "ret_lag1" in panel.columns
    assert "ret_lag5" in panel.columns
    assert "target" in panel.columns
    assert len(panel) > 0


def test_build_lag_features_no_nan(sample_returns):
    panel = build_lag_features(sample_returns, lags=5, horizon=1)
    assert not panel[["ret_lag1", "ret_lag2", "ret_lag3", "ret_lag4", "ret_lag5"]].isna().any().any()


def test_build_lag_features_target_sign(sample_returns):
    panel = build_lag_features(sample_returns, lags=5, horizon=1)
    assert (panel["target"].abs() > 0).any()


def test_compute_baseline_metrics_empty():
    signals = pd.DataFrame()
    metrics = compute_baseline_metrics(signals, pd.DataFrame(), horizon=1)
    assert np.isnan(metrics["accuracy"])


def test_compute_baseline_metrics_basic(sample_returns):
    from sklearn.linear_model import LogisticRegression
    from newsalpha.baselines.endogenous import build_lag_features, expanding_cv_signals

    panel = build_lag_features(sample_returns, lags=5, horizon=1)
    signals = expanding_cv_signals(panel, LogisticRegression, {"max_iter": 100}, min_train=50, lags=5)
    if not signals.empty:
        metrics = compute_baseline_metrics(signals, sample_returns, horizon=1)
        assert 0 <= metrics["accuracy"] <= 1 or np.isnan(metrics["accuracy"])


# ── seSTM ───────────────────────────────────────────────────────────────────

@pytest.fixture
def doc_term_matrix():
    np.random.seed(42)
    return np.random.randint(0, 3, size=(100, 50))


@pytest.fixture
def returns_array():
    np.random.seed(42)
    return np.random.randn(100) * 0.02


def test_screening_returns_subset(doc_term_matrix, returns_array):
    words = [f"w{i}" for i in range(50)]
    selected, f_dict = screening(doc_term_matrix, returns_array, words, top_k=20)
    assert len(selected) <= 40
    assert isinstance(f_dict, dict)


def test_screening_filters_low_count(doc_term_matrix, returns_array):
    words = [f"w{i}" for i in range(50)]
    selected, f_dict = screening(doc_term_matrix, returns_array, words, min_articles_per_word=100, top_k=5)
    assert len(f_dict) == 0


def test_supervised_topic_estimate_shape(doc_term_matrix, returns_array):
    words = [f"w{i}" for i in range(50)]
    selected, _ = screening(doc_term_matrix, returns_array, words, top_k=20)
    selected_idx = [words.index(w) for w in selected]
    O = supervised_topic_estimate(doc_term_matrix[:, selected_idx], returns_array, lam=1.0)
    assert O.shape == (len(selected_idx), 2)


def test_supervised_topic_estimate_nonneg(doc_term_matrix, returns_array):
    words = [f"w{i}" for i in range(50)]
    selected, _ = screening(doc_term_matrix, returns_array, words, top_k=20)
    selected_idx = [words.index(w) for w in selected]
    O = supervised_topic_estimate(doc_term_matrix[:, selected_idx], returns_array, lam=1.0)
    assert (O >= 0).all()


def test_compute_p_score_range(doc_term_matrix, returns_array):
    words = [f"w{i}" for i in range(50)]
    selected, _ = screening(doc_term_matrix, returns_array, words, top_k=20)
    selected_idx = [words.index(w) for w in selected]
    O = supervised_topic_estimate(doc_term_matrix[:, selected_idx], returns_array, lam=1.0)
    p_scores = compute_p_score(doc_term_matrix[:, selected_idx], O)
    assert p_scores.shape == (100,)
    assert (p_scores >= 0).all() and (p_scores <= 1).all()


def test_sestm_expanding_unattributed_corpus_raises():
    """Корпус без привязки статей к тикерам ("ALL") — громкая ошибка, а не тихий пустой фрейм."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2015-01-05", periods=600, freq="W-MON")
    returns_panel = pd.DataFrame(
        rng.normal(0, 0.02, (600, 2)), index=dates, columns=["SBER", "GAZP"]
    )
    doc_term = rng.integers(0, 3, size=(200, 50))
    art_dates = pd.Series(pd.date_range("2015-01-06", periods=200, freq="D"))
    with pytest.raises(ValueError, match="не привязана"):
        sestm_expanding(
            returns_panel, doc_term, art_dates, ["SBER", "GAZP"],
            np.array(["ALL"] * 200), train_years=2, test_years=1,
        )
