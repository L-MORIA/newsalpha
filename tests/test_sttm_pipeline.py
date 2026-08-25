"""Тесты сборки STTM-индекса (pipeline.sttm_expanding) на синтетике.

Ключевой тест — анти-утечка: индекс тестового года обязан быть НЕЧУВСТВИТЕЛЬНЫМ
к пертурбации доходностей этого же года (тональности оцениваются только по
train-неделям) и чувствительным к пертурбации train-доходностей.
"""
import numpy as np
import pandas as pd
import pytest

from newsalpha.sttm.pipeline import sttm_expanding

VOCAB = {"альфа": 0, "бета": 1, "гамма": 2}
TW = [[("альфа", 0.4), ("бета", 0.3), ("гамма", 0.3)],
      [("бета", 0.5), ("гамма", 0.5)]]


@pytest.fixture
def world():
    rng = np.random.default_rng(11)
    weeks = list(pd.date_range("2020-01-03", periods=210, freq="W-FRI"))
    t = len(weeks)
    years = np.array([w.year for w in weeks])
    rets = pd.Series(rng.normal(0, 0.02, t), index=pd.DatetimeIndex(weeks))
    # альфа коррелирует с доходностью train-лет (годы < последнего)
    train_mask = years < years.max()
    signal = np.where(train_mask, np.sign(rets.to_numpy()), 0.0)
    c_words = np.vstack([
        np.abs(rng.normal(10, 3, t)) + 20 * signal,
        np.abs(rng.normal(5, 2, t)),
        np.abs(rng.normal(7, 2, t)),
    ])
    theta = rng.dirichlet(np.ones(2), size=t).T  # [темы × недели]
    return rets, theta, c_words, weeks


def test_index_only_for_post_train_years(world):
    rets, theta, c_words, weeks = world
    idx = sttm_expanding(rets, theta, c_words, TW, VOCAB, weeks,
                         initial_train_years=2)
    first_idx_year = idx.index.min().year
    assert first_idx_year == weeks[0].year + 2


def test_no_leakage_from_test_returns(world):
    """Переворот доходностей года Y не меняет индекс года Y (каждый Y отдельно)."""
    rets, theta, c_words, weeks = world
    idx = sttm_expanding(rets, theta, c_words, TW, VOCAB, weeks)
    assert len(idx) > 50
    for y in sorted(set(idx.index.year)):
        flipped = rets.copy()
        m = flipped.index.year == y
        flipped[m] = -flipped[m]
        idx_flipped = sttm_expanding(flipped, theta, c_words, TW, VOCAB, weeks)
        np.testing.assert_array_equal(
            idx[idx.index.year == y].to_numpy(),
            idx_flipped[idx.index.year == y].to_numpy(),
        )


def test_train_perturbation_changes_index(world):
    rets, theta, c_words, weeks = world
    idx = sttm_expanding(rets, theta, c_words, TW, VOCAB, weeks)
    pert = rets.copy()
    head = pert.index.year == weeks[0].year
    pert[head] = -pert[head]
    idx_pert = sttm_expanding(pert, theta, c_words, TW, VOCAB, weeks)
    assert not np.allclose(idx.to_numpy(), idx_pert.to_numpy())


def test_nan_weeks_excluded_from_training(world):
    """Заполнение NaN-недель мусором меняет тон ⇒ они реально исключаются."""
    rets, theta, c_words, weeks = world
    gappy = rets.copy()
    gappy[gappy.index.dayofyear % 7 == 0] = np.nan
    idx_gaps = sttm_expanding(gappy, theta, c_words, TW, VOCAB, weeks)
    filled = gappy.fillna(0.999)
    idx_filled = sttm_expanding(filled, theta, c_words, TW, VOCAB, weeks)
    assert not np.allclose(idx_gaps.to_numpy(), idx_filled.to_numpy())


def test_index_in_unit_range(world):
    rets, theta, c_words, weeks = world
    idx = sttm_expanding(rets, theta, c_words, TW, VOCAB, weeks)
    assert ((idx > 0) & (idx < 1)).all()
