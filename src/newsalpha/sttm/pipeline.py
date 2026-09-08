"""Сборка STTM-индекса из doc-topic матрицы и доходностей (Этап 4).

Схема статьи: потоки тем Θ[j,t] и слов c[w,t] глобальные (вся рубрика),
тональности слов f_w и тем f_T — свои для каждого тикера (корреляция потока
с его доходностью) и пересчитываются на каждом expanding-сплите только по
train-неделям. Индекс тикера: TTS[t,j]=Θ·f_T[j] → Σ_j → сигмоид → [0,1].
"""
import numpy as np
import pandas as pd

from newsalpha.sttm.core import (
    stock_index,
    topic_stream,
    topic_tone,
    tts,
    word_stream,
    word_tone_matrix,
)


def friday_of(date_series: pd.Series) -> pd.Series:
    """Дата → пятница её недели (метки совпадают с бинами W-FRI рыночного модуля)."""
    dt = pd.to_datetime(date_series)
    return dt + pd.to_timedelta((4 - dt.dt.dayofweek) % 7, unit="D")


def build_streams(
    doc_topic: np.ndarray,
    docs_tokens: list[list[str]],
    dates: pd.Series,
    vocab: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, list[pd.Timestamp]]:
    """Θ[темы × недели], c[слова × недели], список пятниц (календарь потоков)."""
    if doc_topic.shape[0] != len(docs_tokens) or len(docs_tokens) != len(dates):
        raise ValueError(
            "build_streams: несогласованные входы: "
            f"doc_topic={doc_topic.shape}, docs={len(docs_tokens)}, dates={len(dates)} — "
            "preproc и doc_topic должны описывать одни и те же документы в том же порядке "
            "(частый случай: preproc расширен OOS-периодом, а doc_topic — нет)"
        )
    fridays = friday_of(dates)
    labels = sorted(pd.unique(fridays))
    week_idx = fridays.map({w: i for i, w in enumerate(labels)}).to_numpy()
    theta = topic_stream(doc_topic, week_idx, len(labels))
    c = word_stream(docs_tokens, vocab, len(labels), week_idx)
    return theta, c, [pd.Timestamp(w) for w in labels]


def topic_word_lists(model, topn: int = 40) -> list[list[tuple[str, float]]]:
    """Топ-слова каждой темы с запасом topn (обрезка по массе — в topic_tone)."""
    return [model.show_topic(j, topn=topn) for j in range(model.num_topics)]


def _topic_tones(tw_lists, f_w, vocab, prob_mass):
    return np.array(
        [topic_tone(tw, f_w, vocab, prob_mass=prob_mass) for tw in tw_lists]
    )


def build_streams_daily(
    doc_topic: np.ndarray,
    docs_tokens: list[list[str]],
    dates: pd.Series,
    vocab: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, list[pd.Timestamp]]:
    """Θ[темы × дни], c[слова × дни], список дат (календарь потоков, дневной)."""
    dates_pd = pd.to_datetime(dates)
    labels = sorted(pd.unique(dates_pd))
    day_idx = dates_pd.map({d: i for i, d in enumerate(labels)}).to_numpy()
    theta = topic_stream(doc_topic, day_idx, len(labels))
    c = word_stream(docs_tokens, vocab, len(labels), day_idx)
    return theta, c, [pd.Timestamp(d) for d in labels]


def sttm_expanding_daily(
    returns: pd.Series,
    theta: np.ndarray,
    c_words: np.ndarray,
    tw_lists: list[list[tuple[str, float]]],
    vocab: dict[str, int],
    days: list[pd.Timestamp],
    gamma: float = 0.05,
    prob_mass: float = 0.3,
    initial_train_years: int = 2,
    norm: str = "sigmoid",
    first_test_year: int = 2015,
) -> pd.Series:
    """Expanding-CV индекс одного тикера (дневная частота)."""
    ret_by_day = returns.to_dict()
    r_vec = np.array([ret_by_day.get(d, np.nan) for d in days])
    day_years = np.array([d.year for d in days])

    out = {}
    for test_year in range(first_test_year, day_years.max() + 1):
        train_mask = (day_years < test_year) & ~np.isnan(r_vec)
        if train_mask.sum() < 20:
            continue
        f_w = word_tone_matrix(c_words[:, train_mask], r_vec[train_mask], gamma=gamma)
        f_topics = _topic_tones(tw_lists, f_w, vocab, prob_mass)
        test_mask = day_years == test_year
        idx = stock_index(tts(theta[:, test_mask], f_topics), norm=norm)
        for d, v in zip(np.array(days)[test_mask], idx):
            out[d] = v
    return pd.Series(out, name="sttm_index").sort_index()


def sttm_expanding(
    returns: pd.Series,
    theta: np.ndarray,
    c_words: np.ndarray,
    tw_lists: list[list[tuple[str, float]]],
    vocab: dict[str, int],
    weeks: list[pd.Timestamp],
    gamma: float = 0.05,
    prob_mass: float = 0.3,
    initial_train_years: int = 2,
    norm: str = "sigmoid",
    first_test_year: int = 2015,
) -> pd.Series:
    """Expanding-CV индекс одного тикера.

    На каждом календарном году Y ≥ first_test_year тональности оцениваются
    только по неделям с годом < Y и валидной доходностью; индекс тестового
    года считается на этих тональностях.

    first_test_year ОБЯЗАТЕЛЕН явным числом: глобальный календарь протокола
    статьи = год начала данных + initial_train_years (2013+2=2015). Поздно
    листингованные тикеры отсекаются гейтом min train-недель, а не сдвигом
    календаря. Дефолт-вычисление от потока маскировало регрессию (recheck
    Mavis 2026-08-25, находка №3).
    Возврат: Series[week] = индекс тестовых недель.
    """
    ret_by_week = returns.to_dict()
    r_vec = np.array([ret_by_week.get(w, np.nan) for w in weeks])
    week_years = np.array([w.year for w in weeks])

    out = {}
    for test_year in range(first_test_year, week_years.max() + 1):
        train_mask = (week_years < test_year) & ~np.isnan(r_vec)
        if train_mask.sum() < 8:
            continue
        f_w = word_tone_matrix(c_words[:, train_mask], r_vec[train_mask], gamma=gamma)
        f_topics = _topic_tones(tw_lists, f_w, vocab, prob_mass)
        test_mask = week_years == test_year
        # tts даёт [недели × темы]; агрегация Σ_j внутри stock_index идёт по оси 1
        idx = stock_index(tts(theta[:, test_mask], f_topics), norm=norm)
        for w, v in zip(np.array(weeks)[test_mask], idx):
            out[w] = v
    return pd.Series(out, name="sttm_index").sort_index()
