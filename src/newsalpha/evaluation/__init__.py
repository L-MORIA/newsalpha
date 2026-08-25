"""Статистическая оценка STTM-индекса (Этап 6).

Модули:
    granger   — Грейнджер-причинность (maxlag=5) + ADF стационарности
    placebo   — placebo ±k недель (перетасовка доходностей)
    direction — Acc/F1/AUC/Spearman для предсказания направления
    sensitivity — мини-грид {n_topics}×{prob_mass}×{γ}
"""
