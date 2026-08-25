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
    first_test_year: int | None = None,
) -> pd.Series:
    """Expanding-CV индекс одного тикера.

    На каждом календарном году Y тональности оцениваются только по неделям
    с годом < Y и валидной доходностью; индекс тестового года считается на
    этих тональностях. first_test_year задаёт нижнюю границу тестовых лет
    (иначе первый год потока + initial_train_years).
    Возврат: Series[week] = индекс тестовых недель.
    """
    ret_by_week = returns.to_dict()
    r_vec = np.array([ret_by_week.get(w, np.nan) for w in weeks])
    week_years = np.array([w.year for w in weeks])

    start = first_test_year or int(week_years.min() + initial_train_years)
    out = {}
    for test_year in range(start, week_years.max() + 1):
        train_mask = (week_years < test_year) & ~np.isnan(r_vec)
        if train_mask.sum() < 8:
            continue
        f_w = word_tone_matrix(c_words[:, train_mask], r_vec[train_mask], gamma=gamma)
        f_topics = _topic_tones(tw_lists, f_w, vocab, prob_mass)
        test_mask = week_years == test_year
        # [недели × темы]: агрегация Σ_j внутри stock_index идёт по оси 1
        idx = stock_index((theta[:, test_mask] * f_topics[:, None]).T, norm=norm)
        for w, v in zip(np.array(weeks)[test_mask], idx):
            out[w] = v
    return pd.Series(out, name="sttm_index").sort_index()
