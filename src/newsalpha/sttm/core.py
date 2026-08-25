"""Ядро STTM: тональности слов и тем, TTS, индекс (Этап 4).

Спецификация — статья STTM (PeerJ CS 2022) + PLAN.md Этап 4:
1. поток тем Θ[j,t]: сумма вероятностей темы j по документам недели t;
2. поток слова c[w,t]: суммарная частота слова w в неделях t;
3. тон слова f_w = Пирсон r(c_w, r) при p < γ, иначе 0;
4. тон темы: топ-слова до массы prob_mass от суммы вероятностей темы,
   f_T = pProb − nProb (0, если значимых слов нет);
5. TTS[t,j] = Θ[j,t] · f_T[j];
6. индекс[t] = Σ_j TTS → нормировка сигмоидом в [0,1].

Все потоки строятся ТОЛЬКО на train-окне (анти-утечка, красный пункт №2).
"""
import numpy as np
from scipy import stats


def word_stream(docs_tokens: list[list[str]], vocab: dict[str, int], n_weeks: int,
                doc_week: np.ndarray) -> np.ndarray:
    """c[w, t]: частоты слов по неделям.

    docs_tokens — токены документа; doc_week[i] — номер недели документа i.
    """
    c = np.zeros((len(vocab), n_weeks))
    for toks, t in zip(docs_tokens, doc_week):
        for tok in toks:
            w = vocab.get(tok)
            if w is not None:
                c[w, t] += 1
    return c


def topic_stream(doc_topic: np.ndarray, doc_week: np.ndarray, n_weeks: int) -> np.ndarray:
    """Θ[j, t]: сумма вероятностей темы j по документам недели t."""
    theta = np.zeros((doc_topic.shape[1], n_weeks))
    np.add.at(theta.T, doc_week, doc_topic)
    return theta


def word_tone(c_words: np.ndarray, returns: np.ndarray, gamma: float = 0.05) -> np.ndarray:
    """f_w = pearson r(c_w, r), обнуляется при p >= gamma.

    Недели без сделок (NaN-доходность) исключаются попарно по каждому слову.
    """
    mask = ~np.isnan(returns)
    r = returns[mask]
    f = np.zeros(c_words.shape[0])
    for w in range(c_words.shape[0]):
        x = c_words[w, mask]
        if x.std() == 0 or len(x) < 3:
            continue
        corr = stats.pearsonr(x, r)
        if corr.pvalue < gamma:
            f[w] = corr.statistic
    return f


def word_tone_matrix(c_words: np.ndarray, returns: np.ndarray,
                     gamma: float = 0.05) -> np.ndarray:
    """Векторизованный аналог word_tone для матрицы [слова × недели] без NaN.

    Совпадает с попарным scipy.pearsonr до ~1e-9; слова с нулевой дисперсией
    и недели с NaN заранее исключены вызывающим кодом.
    """
    mask = ~np.isnan(returns)
    x = c_words[:, mask].astype(np.float64)
    r = returns[mask].astype(np.float64)
    n = len(r)
    if n < 3:
        return np.zeros(c_words.shape[0])
    xc = x - x.mean(axis=1, keepdims=True)
    rc = r - r.mean()
    dx = np.sqrt((xc**2).sum(axis=1))
    dr = np.sqrt((rc**2).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where((dx > 0) & (dr > 0), (xc @ rc) / (dx * dr), 0.0)
    dof = n - 2
    tstat = np.abs(corr) * np.sqrt(dof / np.maximum(1.0 - corr**2, 1e-12))
    pval = 2.0 * stats.t.sf(tstat, dof)
    return np.where(pval < gamma, corr, 0.0)


def topic_tone(topic_words: list[tuple[str, float]], f_w: np.ndarray,
               vocab: dict[str, int], prob_mass: float = 0.3) -> float:
    """f_T = pProb − nProb по топ-словам темы до кумулятивной массы prob_mass.

    Масса считается от суммы вероятностей ВСЕХ слов темы (как в ф. 5.3 статьи).
    Если среди отобранных нет ни одного значимого слова — 0.
    """
    total_p = sum(p for _, p in topic_words)
    if total_p <= 0:
        return 0.0
    p_prob = n_prob = 0.0
    any_significant = False
    cum = 0.0
    for word, p in topic_words:
        idx = vocab.get(word)
        fw = f_w[idx] if idx is not None else 0.0
        cum += p
        if fw > 0:
            p_prob += p
            any_significant = True
        elif fw < 0:
            n_prob += p
            any_significant = True
        if cum / total_p >= prob_mass:
            break
    return (p_prob - n_prob) if any_significant else 0.0


def tts(theta: np.ndarray, f_topics: np.ndarray) -> np.ndarray:
    """TTS[t, j] = Θ[j, t] · f_T[j]; на входе Θ [темы × недели], выход [недели × темы]."""
    return (theta * f_topics[:, None]).T


def stock_index(tts_week: np.ndarray, norm: str = "sigmoid") -> np.ndarray:
    """Индекс недели: агрегация Σ_j TTS → нормировка в [0,1]."""
    agg = tts_week.sum(axis=1)
    if norm == "sigmoid":
        return 1.0 / (1.0 + np.exp(-agg))
    raise ValueError(f"неизвестная нормировка: {norm}")
