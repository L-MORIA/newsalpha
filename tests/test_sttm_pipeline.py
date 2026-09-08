"""Тесты сборки STTM-индекса (pipeline.sttm_expanding) на синтетике.

Ключевой тест — анти-утечка: индекс тестового года обязан быть НЕЧУВСТВИТЕЛЬНЫМ
к пертурбации доходностей этого же года (тональности оцениваются только по
train-неделям) и чувствительным к пертурбации train-доходностей.

Фикстура: topic_0 — слова с положительной корреляцией (+20*signal),
topic_1 — слова с отрицательной корреляцией (-20*signal).
prob_mass=0.3 охватывает ≥2 слов на тему → f_T₀>0, f_T₁<0.
"""
import numpy as np
import pandas as pd
import pytest

from newsalpha.sttm.pipeline import build_streams, build_streams_daily, sttm_expanding

_WORDS = [f"w{i}" for i in range(20)]
VOCAB = {w: i for i, w in enumerate(_WORDS)}
TW = [
    [("w0", 0.15), ("w2", 0.14), ("w4", 0.13), ("w6", 0.12), ("w8", 0.11),
     ("w10", 0.10), ("w12", 0.09), ("w14", 0.08), ("w16", 0.05), ("w18", 0.03)],
    [("w1", 0.15), ("w3", 0.14), ("w5", 0.13), ("w7", 0.12), ("w9", 0.11),
     ("w11", 0.10), ("w13", 0.09), ("w15", 0.08), ("w17", 0.05), ("w19", 0.03)],
]


@pytest.fixture
def world():
    rng = np.random.default_rng(11)
    weeks = list(pd.date_range("2020-01-03", periods=210, freq="W-FRI"))
    t = len(weeks)
    years = np.array([w.year for w in weeks])
    rets = pd.Series(rng.normal(0, 0.02, t), index=pd.DatetimeIndex(weeks))
    train_mask = years < years.max()
    signal = np.where(train_mask, np.sign(rets.to_numpy()), 0.0)
    rows = []
    for i in range(20):
        base = np.abs(rng.normal(10, 2, t))
        if i % 2 == 0:
            rows.append(base + signal)
        else:
            rows.append(base - signal)
    c_words = np.vstack(rows)
    theta = rng.dirichlet(np.ones(2), size=t).T
    return rets, theta, c_words, weeks


def test_index_only_for_post_train_years(world):
    rets, theta, c_words, weeks = world
    idx = sttm_expanding(rets, theta, c_words, TW, VOCAB, weeks,
                         first_test_year=2022)
    first_idx_year = idx.index.min().year
    assert first_idx_year == 2022


def test_no_leakage_from_test_returns(world):
    """Пертурбация года P: индекс года C инвариантен <=> P >= C (двойной цикл).

    Ловит и прямой leak y->idx[y], и off-by-one train_mask («<=» вместо «<»),
    и замороженные тональности (чувствительность ветки P<C).
    """
    rets, theta, c_words, weeks = world
    idx = sttm_expanding(rets, theta, c_words, TW, VOCAB, weeks,
                         first_test_year=weeks[0].year + 2)
    years = sorted(set(idx.index.year))
    for p in years:
        flipped = rets.copy()
        m = flipped.index.year == p
        flipped[m] = -flipped[m]
        idx_flipped = sttm_expanding(flipped, theta, c_words, TW, VOCAB, weeks,
                                     first_test_year=weeks[0].year + 2)
        for c in years:
            a = idx[idx.index.year == c].to_numpy()
            b = idx_flipped[idx_flipped.index.year == c].to_numpy()
            if p >= c:
                np.testing.assert_array_equal(a, b)
            else:
                assert not np.allclose(a, b), (p, c)


def test_train_perturbation_changes_index(world):
    rets, theta, c_words, weeks = world
    idx = sttm_expanding(rets, theta, c_words, TW, VOCAB, weeks, first_test_year=2022)
    pert = rets.copy()
    head = pert.index.year == weeks[0].year
    pert[head] = -pert[head]
    idx_pert = sttm_expanding(pert, theta, c_words, TW, VOCAB, weeks, first_test_year=2022)
    assert not np.allclose(idx.to_numpy(), idx_pert.to_numpy())


def test_nan_weeks_excluded_from_training(world):
    """Заполнение NaN-недель мусором меняет тон => они реально исключаются."""
    rets, theta, c_words, weeks = world
    gappy = rets.copy()
    gappy[gappy.index.dayofyear % 7 == 0] = np.nan
    idx_gaps = sttm_expanding(gappy, theta, c_words, TW, VOCAB, weeks, first_test_year=2022)
    filled = gappy.fillna(0.999)
    idx_filled = sttm_expanding(filled, theta, c_words, TW, VOCAB, weeks, first_test_year=2022)
    assert not np.allclose(idx_gaps.to_numpy(), idx_filled.to_numpy())


def test_index_in_unit_range(world):
    rets, theta, c_words, weeks = world
    idx = sttm_expanding(rets, theta, c_words, TW, VOCAB, weeks, first_test_year=2022)
    assert ((idx > 0) & (idx < 1)).all()


def test_build_streams_rejects_mismatched_inputs():
    """preproc расширен OOS-периодом, а doc_topic — нет: громкая ошибка, а не cryptic broadcast."""
    rng = np.random.default_rng(7)
    n_docs, n_topics = 60, 2
    doc_topic = rng.dirichlet(np.ones(n_topics), size=n_docs)
    docs_tokens = [["w0", "w1"] for _ in range(n_docs + 10)]
    dates = pd.Series(pd.date_range("2020-01-06", periods=n_docs + 10, freq="D"))
    with pytest.raises(ValueError, match="несогласованные входы"):
        build_streams(doc_topic, docs_tokens, dates, VOCAB)


def test_build_streams_daily_rejects_mismatched_inputs():
    """preproc обрезан smoke-прогоном --limit: громкая ошибка, а не cryptic broadcast."""
    rng = np.random.default_rng(7)
    n_docs, n_topics = 60, 2
    doc_topic = rng.dirichlet(np.ones(n_topics), size=n_docs)
    docs_tokens = [["w0", "w1"] for _ in range(n_docs - 10)]
    dates = pd.Series(pd.date_range("2020-01-06", periods=n_docs - 10, freq="D"))
    with pytest.raises(ValueError, match="несогласованные входы"):
        build_streams_daily(doc_topic, docs_tokens, dates, VOCAB)
