"""SESTM-базлайн: Supervised Sentiment Topic Model (Этап 5).

Реализация по MATLAB-коду Ke/Kelly/Xiu (JASA 2026 supplement).
Ключевые шаги:
  1. Screening: f_i = доля позитивных статей, порог count >= thresh * log(N)
  2. Supervised topic model: ровно 2 темы (bull/bear), оценка регрессией на рангах
  3. P-score: вероятность «позитивной темы» для каждой статьи
  4. Агрегация по тикеру → торговый сигнал

Адаптация под российский рынок: недельный горизонт, top-20% long-only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def screening(
    doc_term: np.ndarray,
    returns: np.ndarray,
    words: list[str],
    min_articles_per_word: int = 5,
    top_k: int = 200,
) -> tuple[list[str], dict[str, float]]:
    """Screening слов по статье Ke/Kelly/Xiu §3.2.

    doc_term: [articles x words] бинарная или частотная матрица
    returns:  [articles] нормированные доходности
    words:    список слов (длины doc_term.shape[1])

    Возвращает: (отобранные_слова, f_i_dict)
    """
    n_articles = doc_term.shape[0]
    median_ret = np.median(returns)
    pos_mask = returns > median_ret

    f_i = np.zeros(doc_term.shape[1])
    article_count = np.zeros(doc_term.shape[1])

    for w in range(doc_term.shape[1]):
        appears = doc_term[:, w] > 0
        article_count[w] = appears.sum()
        if article_count[w] > 0:
            f_i[w] = pos_mask[appears].mean()

    log_n = np.log(max(n_articles, 1))
    threshold = min_articles_per_word * log_n
    valid = article_count >= threshold

    f_dict = {words[i]: f_i[i] for i in range(len(words)) if valid[i]}
    sorted_words = sorted(f_dict.items(), key=lambda x: x[1], reverse=True)

    bullish = [w for w, _ in sorted_words[:top_k]]
    bearish = [w for w, _ in sorted_words[-top_k:]]
    selected = bullish + bearish

    return selected, f_dict


def supervised_topic_estimate(
    doc_term_selected: np.ndarray,
    returns: np.ndarray,
    lam: float = 1.0,
) -> np.ndarray:
    """Оценка supervised topic model (2 темы).

    doc_term_selected: [articles x selected_words] — матрица документов по отобранным словам
    returns: [articles] — нормированные доходности
    lam: штраф (Beta-приор, тянет веса к 0.5)

    Возвращает: O [words x 2] — матрица тем (bull, bear)
    """
    n_articles, n_words = doc_term_selected.shape
    if n_articles == 0 or n_words == 0:
        return np.zeros((max(n_words, 1), 2))

    # Нормировка документов
    col_sums = doc_term_selected.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    H = doc_term_selected / col_sums

    # Ранговые веса
    ranks = sp_stats.rankdata(returns)
    w1 = ranks / (n_articles + 1)
    w2 = 1 - w1

    # Целевая матрица рангов
    W = np.column_stack([w1, w2])

    # Линейная регрессия с регуляризацией: O = H^+ @ W
    # (H'H + lam*I)^{-1} H'W
    HtH = H.T @ H + lam * np.eye(n_words)
    HtW = H.T @ W
    O = np.linalg.solve(HtH, HtW)

    # Проекция на симплекс: O >= 0, sum = 1 по темам
    O = np.maximum(O, 0)
    row_sums = O.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    O = O / row_sums

    return O


def compute_p_score(
    doc_term_selected: np.ndarray,
    O: np.ndarray,
) -> np.ndarray:
    """P-score для каждой статьи: доля «позитивной темы».

    doc_term_selected: [articles x words]
    O: [words x 2]

    Возвращает: [articles] p-score в [0, 1]
    """
    H = doc_term_selected.astype(float)
    col_sums = H.sum(axis=1, keepdims=True)
    col_sums[col_sums == 0] = 1
    H = H / col_sums

    scores = H @ O
    total = scores.sum(axis=1, keepdims=True)
    total[total == 0] = 1
    p_scores = scores[:, 0] / total[:, 0]

    return np.clip(p_scores, 0, 1)


def sestm_expanding(
    returns_panel: pd.DataFrame,
    doc_term: np.ndarray,
    dates: pd.Series,
    tickers: list[str],
    doc_tickers: np.ndarray,
    train_years: int = 10,
    test_years: int = 1,
    lam: float = 1.0,
    min_articles: int = 5,
    top_k: int = 200,
) -> pd.DataFrame:
    """SESTM expanding CV: train-years лет → test-years лет.

    returns_panel: DataFrame[недели x тикеры]
    doc_term: [articles x words] матрица документов
    dates: Series с датами статей
    tickers: список тикеров статей (параллельно doc_term)
    doc_tickers: массив тикеров для каждой статьи

    Возвращает: DataFrame[недели x тикеры] с p-score агрегированными по неделям.
    """
    week_labels = sorted(returns_panel.index)
    article_dates = pd.to_datetime(dates)

    words = [f"w{i}" for i in range(doc_term.shape[1])]

    doc_set = set(np.asarray(doc_tickers).tolist())
    if not (doc_set & set(returns_panel.columns)):
        raise ValueError(
            "sestm_expanding: ни одна статья не привязана к тикерам из returns_panel "
            f"(уникальных doc_tickers: {len(doc_set)}, пример: {list(doc_set)[:5]}); "
            "per-ticker сигналы построить нельзя — нужна разметка статей по тикерам"
        )

    all_weeks = []
    all_tickers = []
    all_scores = []

    for test_start_idx in range(
        train_years * 52, len(week_labels), test_years * 52
    ):
        test_end_idx = min(test_start_idx + test_years * 52, len(week_labels))
        if test_start_idx >= len(week_labels):
            break

        train_start = max(0, test_start_idx - train_years * 52)
        train_end = test_start_idx

        train_weeks = week_labels[train_start:train_end]
        test_weeks = week_labels[test_start_idx:test_end_idx]

        if len(train_weeks) < 52 or len(test_weeks) == 0:
            continue

        train_start_date = pd.Timestamp(train_weeks[0])
        train_end_date = pd.Timestamp(train_weeks[-1])
        test_start_date = pd.Timestamp(test_weeks[0])

        train_mask_dates = (
            (article_dates >= train_start_date) & (article_dates <= train_end_date)
        )

        train_doc = doc_term[train_mask_dates]
        train_returns_dates = article_dates[train_mask_dates]
        train_tickers_arr = doc_tickers[train_mask_dates]

        if train_doc.shape[0] < min_articles * 2:
            continue

        train_rets_for_screening = np.zeros(train_doc.shape[0])
        for i in range(train_doc.shape[0]):
            d = train_returns_dates.iloc[i] if hasattr(train_returns_dates, 'iloc') else pd.Timestamp(train_returns_dates[i])
            t = train_tickers_arr.iloc[i] if hasattr(train_tickers_arr, 'iloc') else train_tickers_arr[i]
            if t in returns_panel.columns:
                week_of = returns_panel.index[returns_panel.index.searchsorted(d, side='right') - 1]
                if week_of in returns_panel.index:
                    train_rets_for_screening[i] = returns_panel.loc[week_of, t] if not np.isnan(returns_panel.loc[week_of, t]) else 0

        selected, f_dict = screening(
            train_doc, train_rets_for_screening, words,
            min_articles_per_word=min_articles, top_k=top_k,
        )

        if len(selected) < 10:
            continue

        selected_idx = [words.index(w) for w in selected if w in words]
        train_doc_sel = train_doc[:, selected_idx]

        O = supervised_topic_estimate(train_doc_sel, train_rets_for_screening, lam=lam)

        test_mask_dates = article_dates >= test_start_date
        test_doc = doc_term[test_mask_dates]
        test_tickers_arr = doc_tickers[test_mask_dates]
        test_dates_arr = article_dates[test_mask_dates]

        if test_doc.shape[0] == 0:
            continue

        test_doc_sel = test_doc[:, selected_idx]
        p_scores = compute_p_score(test_doc_sel, O)

        for wi, week in enumerate(test_weeks):
            week_start = pd.Timestamp(week)
            week_end = week_start + pd.Timedelta(days=7)
            in_week = (test_dates_arr >= week_start) & (test_dates_arr < week_end)
            if in_week.sum() == 0:
                continue

            for t in tickers:
                t_mask = in_week & (test_tickers_arr == t)
                if t_mask.sum() > 0:
                    all_weeks.append(week)
                    all_tickers.append(t)
                    all_scores.append(p_scores[t_mask].mean())

    if not all_weeks:
        return pd.DataFrame(index=returns_panel.index, columns=tickers, dtype=float)

    result = pd.DataFrame({
        "week": all_weeks,
        "ticker": all_tickers,
        "p_score": all_scores,
    })
    pivot = result.pivot_table(index="week", columns="ticker", values="p_score")
    pivot = pivot.reindex(returns_panel.index).reindex(columns=tickers)
    return pivot
