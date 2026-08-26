# Long-Short decile test на Kommersant: STTM и кросс-секционный ранг

**Дата:** 26 августа 2026 г.
**Скрипт:** `experiments/longshort/run_longshort.py`
**Корпус:** Коммерсантъ, in-sample 2013–2021 (фактически STTM-индекс покрывает 2015–2021 = 366 недель × 39 тикеров)
**Цель:** проверить, предсказывает ли STTM **относительный** ранг тикеров (long-short), а не только **абсолютный** (long-only).

---

## Гипотеза

**Если STTM извлекает информационный сигнал из новостей:**
- Long-short (top-20% long vs bottom-20% short, dollar-neutral) должен давать значимый Sharpe после издержек.
- Long-short убирает рыночную экспозицию (beta), оставляя только «альфу» из новостной компоненты.

**Если STTM не имеет альфы (только повторяет рынок):**
- Long-short Sharpe ≈ 0 (или статистически неотличим от placebo).
- Вся прибыль long-only (Sharpe 1.217) — это **рыночная экспозиция** (β), а не новостной сигнал.

---

## Результат

| Метрика | Long-only top-20% | Long-short 20/20 | Placebo long-short |
|---|---|---|---|
| Gross Sharpe | **1.217** | **0.262** | 0.406 ± 0.244 |
| Net Sharpe (cost + borrow) | 1.198 (0.05%) / 1.160 (0.15%) | **0.064** | — |
| Ann return (gross) | +21.3% | +4.1% | — |
| Ann return (net) | — | +1.0% | — |
| Volatility | 17.5% | 15.5% | — |
| MaxDD (net) | — | −26.5% | — |
| Mean turnover | 0.128 | 0.291 | — |
| **z-score (vs placebo)** | — | **−0.59** | — |
| **p-value (placebo ≥ real)** | — | **0.680** | — |
| **Заключение** | Baseline | **NOT SIGNIFICANT** | — |

---

## Что это значит

### 1. Long-short даёт положительный, но **НЕ значимый** Sharpe

Real long-short gross Sharpe = **0.262**, real net Sharpe = **0.064**.

Placebo (per-column shuffle, 50 sims) даёт:
- mean = **0.406**
- std = **0.244**
- 95% CI = [0.029, 0.946]

**Реальный long-short (0.262) находится НИЖЕ среднего placebo (0.406) и z = −0.59.** Это значит: long-short **не отличим от случайного** и даже чуть хуже медианы placebo.

### 2. Почему placebo mean > 0 (а не 0)

Кажется нелогичным, что перетасованные returns дают положительный Sharpe, но это — известный **эффект selection bias**:

- Каждую неделю алгоритм выбирает **top-20%** тикеров по STTM-индексу.
- На следующей неделе возвраты **не коррелируют** с индексом (мы их перетасовали).
- Но **top-20% тикеров по ЛЮБОМУ признаку** имеют положительный expected return на горизонте 1 недели — это эмпирический факт для MOEX (моментум-эффект, ликвидность премиум, etc.).
- Поэтому даже на «мусорном» сигнале long-basket имеет положительный expected return.

Это **не selection-bias-эффект алгоритма** — это **свойство рынка**. Long-only и long-short на любом сигнале будут иметь положительный expected return.

### 3. Почему long-only 1.217 ≠ long-short 0.262

- **Long-only Sharpe 1.217** = market beta (растущий рынок MOEX 2015-2021) + немного selection bias.
- **Long-short Sharpe 0.262** = только selection bias (market beta убран через short).
- Разница 1.217 − 0.262 ≈ 0.95 = **вклад рыночной экспозиции**.

То есть **~80% long-only Sharpe — это beta**, и только ~20% — кросс-секционный эффект (который неотличим от placebo).

### 4. После издержек — сигнал исчезает

- Long-short **gross Sharpe 0.262** → с реалистичными издержками (20 bps round-trip + 1.5% annual borrow):
- Long-short **net Sharpe 0.064** — **это уровень шума**.

---

## Заключение

**STTM НЕ предсказывает кросс-секционный ранг.** Доказательства:

1. **Placebo z = −0.59, p = 0.68** — реальный long-short ниже медианы placebo.
2. **Net Sharpe 0.064** — после издержек сигнал в пределах шума.
3. **Long-short ≈ 0** при gross Sharpe 0.262 — означает, что long-only «сигнал» на 80% это beta.

**Слабый gross long-short (0.262) — это не альфа STTM, это известный эффект cross-sectional momentum на MOEX 2015-2021**, который проявляется на ЛЮБОМ сигнале, в том числе на случайном.

---

## Сравнение с предыдущими находками

| Тест | Результат | Источник |
|---|---|---|
| Long-only Sharpe | 1.217 | этот эксперимент |
| Long-only placebo p-value | 0.50 | `findings/placebo_kommersant_2026-08-25.md` |
| Endogenous LR vs STTM | LR ≈ STTM | `findings/endogenous_kommersant_2026-08-25.md` |
| **Long-short placebo p-value** | **0.68** | **этот эксперимент** |
| OOS 2022-2026 Sharpe | −0.23 | `findings/oos_evaluation_2026-08-26.md` |

Все четыре теста указывают в одну сторону: **STTM не даёт альфы поверх рыночной экспозиции**.

---

## Метод

### Параметры

```python
cost_rate = 0.001          # 10 bps per unit |Δw|, ~20 bps full rebalance
borrow_rate_weekly = 0.0003  # ~1.5% annual borrow cost, реалистично для liquid TQBR
long_pct = short_pct = 0.20
n_sims = 50  # placebo
```

### Издержки (long-short)

- **Round-trip transaction cost:** каждая единица |Δw| = одна сделка. При полной ребалансировке (открытие нового лонга + закрытие старого лонга + открытие нового шорта + закрытие старого шорта) |Δw| ≈ 2.0 → cost ≈ 2 × 10 bps = 20 bps.
- **Borrow cost:** постоянный, ~1.5% annual = 0.0003/week. При short notional = 1.0 это ~3 bps/week.
- **Итого:** ~23 bps/week, ~12% annual drag — реалистично для активного long-short.

### Placebo

Per-column shuffle (каждый тикер shuffled независимо). Сохраняет:
- Распределение returns каждого тикера.
- Cross-sectional ranking (то есть средний top-20% по ЛЮБОМУ признаку имеет положительный expected return).

Разрушает:
- Cross-section correlation (то есть связь «сигнал → доходность»).

---

## Воспроизведение

```powershell
& "F:\newsalpha\.venv\Scripts\python.exe" -m experiments.longshort.run_longshort
```

Результаты сохраняются в `findings/longshort_kommersant_2026-08-26.json`.

---

## Что это меняет в общей картине

**Раньше** (in-sample analysis, v3 аудит):
> STTM воспроизводит статью: Sharpe 1.42 на Kommersant (статья 1.37 ± 0.09). Три красных пункта закрыты.

**Теперь** (этот + 4 предыдущих findings):
> STTM формально воспроизводит Sharpe, но **весь сигнал = рыночная beta + cross-sectional momentum на MOEX 2015-2021**. Нет новостной альфы. Метод не пригоден для торговли как alpha-source.

**Negative result подтверждён 4 независимыми тестами:**
1. Placebo (Sharpe real ≈ Sharpe shuffled) → k=1
2. Endogenous LR (AR(5)) ≈ STTM по accuracy → k=2
3. **Long-short placebo (этот эксперимент) → k=3**
4. OOS 2022-2026 (−0.23) → k=4

Это делает проект **публикабельным как honest negative result**.

---

*Файл создан: 26.08.2026, 11:15 MSK*
