# experiments/longshort

## Гипотеза

Если STTM-индекс извлекает **информационный сигнал** из новостей, то
**long-short** (top-20% long vs bottom-20% short, dollar-neutral) должен давать
значимый Sharpe после реалистичных издержек.

Если long-short Sharpe ≈ 0 (и равен placebo) — значит вся «прибыль»
long-only (Sharpe 1.42 на Kommersant) — это **рыночная экспозиция (beta)**, а не
альфа из новостей. Это ключевая проверка гипотезы метода.

## Дизайн

- **Сигнал:** `models/sttm_indices/sttm_index_kommersant.parquet` (level, [0, 1])
- **Universe:** 39 TQBR-тикеров, weekly
- **Период:** 2013–2021 (in-sample)
- **Long-basket:** top 20% по STTM-индексу, equal weight, +1/k
- **Short-basket:** bottom 20%, equal weight, −1/k
- **Net exposure:** 0 (dollar-neutral)
- **Gross exposure:** 2.0 (1 long + 1 short)
- **Costs:**
  - `cost_rate = 0.001` (10 bps per unit |Δw|) → ~20 bps full rebalance
  - `borrow_rate_weekly = 0.0003` (~1.5% annual) → реалистично для liquid TQBR
- **Placebo:** per-column shuffle, 50 симуляций

## Запуск

```powershell
& "F:\newsalpha\.venv\Scripts\python.exe" -m experiments.longshort.run_longshort
```

## Результаты

См. `findings/longshort_kommersant_2026-08-26.md` после прогона.

## Что НЕ делает

- ❌ Не тестирует intraday или 4h частоту (→ `scripts/run_daily_sttm.py`)
- ❌ Не использует другой k для LDA (→ `findings/analysis_and_plan_2026-08-26.md` Option B)
- ❌ Не сравнивает с другими сигналами (SESTM, ruBERT) — пока только STTM vs placebo
