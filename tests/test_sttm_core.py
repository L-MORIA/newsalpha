"""Тесты ядра STTM: каждая формула на синтетике (PLAN, чекпойнт Этапа 4)."""
import numpy as np
import pytest

from newsalpha.sttm.core import (
    stock_index,
    topic_stream,
    topic_tone,
    tts,
    word_stream,
    word_tone,
)


def test_word_stream_counts():
    vocab = {"рост": 0, "падение": 1}
    docs = [["рост", "рост", "падение"], ["рост"], []]
    weeks = np.array([0, 0, 1])
    c = word_stream(docs, vocab, n_weeks=2, doc_week=weeks)
    assert c.shape == (2, 2)
    assert c[0, 0] == 3 and c[0, 1] == 0 and c[1, 0] == 1
    # слово вне словаря игнорируется
    docs2 = [["неизвестное"]]
    assert word_stream(docs2, vocab, 1, np.array([0])).sum() == 0


def test_topic_stream_sums_by_week():
    theta_docs = np.array([[0.8, 0.2], [0.6, 0.4], [1.0, 0.0]])
    weeks = np.array([0, 0, 1])
    th = topic_stream(theta_docs, weeks, n_weeks=2)
    assert th[0, 0] == pytest.approx(1.4)
    assert th[1, 0] == pytest.approx(0.6)
    assert th[0, 1] == pytest.approx(1.0)


def test_word_tone_matrix_matches_scalar():
    from newsalpha.sttm.core import word_tone_matrix

    rng = np.random.default_rng(11)
    c = rng.poisson(3.0, size=(50, 80)).astype(float)
    r = rng.normal(size=80)
    fast = word_tone_matrix(c, r, gamma=0.05)
    slow = word_tone(c, r, gamma=0.05)
    np.testing.assert_allclose(fast, slow, atol=1e-9)


def test_word_tone_matrix_zero_variance_row():
    from newsalpha.sttm.core import word_tone_matrix

    c = np.ones((2, 30))
    c[1] = np.arange(30) * 1.0
    r = np.sin(np.arange(30))
    f = word_tone_matrix(c, r, gamma=0.05)
    assert f[0] == 0.0  # константный поток не может быть значимым


def test_word_tone_sign_and_gamma_zeroing():
    rng = np.random.default_rng(11)
    r = rng.normal(size=60)
    # идеальный положительный сигнал → значимый положительный тон
    x_pos = r + rng.normal(scale=0.01, size=60)
    # чистый шум → почти наверняка обнулится при gamma=0.05
    x_noise = rng.normal(size=60)
    c = np.vstack([x_pos, x_noise])
    f = word_tone(c, r, gamma=0.05)
    assert f[0] > 0.9
    assert f[1] == 0.0 or abs(f[1]) < 0.5


def test_word_tone_nan_weeks_excluded():
    r = np.array([np.nan, 1.0, 2.0, 3.0, 4.0])
    c = np.array([[1, 1, 2, 3, 4]], dtype=float)
    f = word_tone(c, r, gamma=0.05)
    assert f[0] > 0.99  # NaN-неделя отброшена, остальное идеально


def test_topic_tone_prob_mass_truncation():
    # тема: два слова по 0.4 и одно 0.2; prob_mass=0.5 → только первые два
    topic_words = [("a", 0.4), ("b", 0.4), ("c", 0.2)]
    vocab = {"a": 0, "b": 1, "c": 2}
    f_w = np.array([1.0, -1.0, 10.0])  # 'c' значим, но не входит в массу 0.5
    ft = topic_tone(topic_words, f_w, vocab, prob_mass=0.5)
    assert ft == pytest.approx(0.4 - 0.4)


def test_topic_tone_zero_without_significant_words():
    topic_words = [("a", 0.5), ("b", 0.5)]
    vocab = {"a": 0, "b": 1}
    f_w = np.array([0.0, 0.0])  # ничего не значимо
    assert topic_tone(topic_words, f_w, vocab) == 0.0


def test_tts_and_index():
    # Θ [2 темы × 2 недели]
    theta = np.array([[0.7, 0.2], [0.3, 0.8]])
    f_topics = np.array([1.0, -0.5])
    t = tts(theta, f_topics)
    assert t.shape == (2, 2)  # [недели × темы]
    assert t[0, 0] == pytest.approx(0.7)
    assert t[1, 0] == pytest.approx(0.2)
    assert t[1, 1] == pytest.approx(-0.4)

    idx = stock_index(t, norm="sigmoid")
    assert idx.shape == (2,)
    assert (idx > 0).all() and (idx < 1).all()
    # неделя с суммарным TTS > 0 даёт индекс > 0.5
    assert idx[0] > 0.5 > idx[1]


def test_stock_index_monotonic():
    t = np.linspace(-2, 2, 21).reshape(-1, 1)
    idx = stock_index(t)
    assert (np.diff(idx.flatten()) > 0).all()
