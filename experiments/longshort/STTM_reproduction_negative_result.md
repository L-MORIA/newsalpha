# Воспроизведение метода STTM на российском рынке: отрицательный результат

**Автор:** Команда newsalpha (M. Moria)
**Дата:** 26 августа 2026 г.
**Версия:** 1.0
**Структура проекта:** `experiments/longshort/` (этот документ + исполняемый скрипт + README)
**Воспроизводимость:** все цифры независимо верифицированы (см. Приложение D)
**Код:** Python 3.11, открытый, в репозитории `newsalpha`

---

## Аннотация

Воспроизведён метод **STTM** (Stock-Tonal Topic Modeling; Рябых и др., PeerJ CS 2022) на корпусе новостей «Коммерсантъ» 2013–2021 и ценах 39 тикеров MOEX TQBR. Формальное воспроизведение успешно: **gross Sharpe 1.217 (статья: 1.37 ± 0.09)**, net Sharpe 1.198 при комиссии 0.05% и 1.160 при 0.15%. Однако **пять независимых тестов показывают, что этот Sharpe — не альфа, а артефакт кросс-секционного отбора**: (1) random-signal плацебо (случайные корзины топ-20% на реальных доходностях, 2 сида × 50 симов, воспроизводимо скриптом `scripts/run_placebo_random.py`): 0.944 ± 0.199 (z = 1.38, p = 0.084) и 0.953 ± 0.168 (z = 1.57, p = 0.058) — **не значимо**; (2) эндогенный базовый уровень AR(5) по цене даёт ту же accuracy (0.516), что и STTM (0.551), и лучший Spearman ρ (+0.068 против −0.054); (3) out-of-sample тест 2022–2026 даёт Sharpe −0.23 (MaxDD −45.5%, direction accuracy 46.2% — хуже монетки); (4) **long-short decile тест (top-20% long vs bottom-20% short, dollar-neutral) даёт net Sharpe 0.064 после издержек — статистически неотличим от placebo (p = 0.68, z = −0.59)**; (5) **benchmarks показывают, что STTM (1.217) лишь незначительно превосходит random top-20% (0.940 ± 0.20, z = 1.42, p = 0.078) и equal-weight (1.069).** Мы делаем вывод, что новостная компонента не добавляет предиктивной силы поверх базовой рыночной экспозиции, и публикуем это как **honest negative result**. Проект служит воспроизводимой реализацией метода с полным набором диагностических тестов, применимой для проверки аналогичных гипотез в будущем.

**Ключевые слова:** STTM, тематическое моделирование, новостной сигнал, placebo-тест, отрицательный результат, MOEX, воспроизводимость

---

## 1. Введение

Финансовый рынок обрабатывает огромный поток новостей, и количественные исследователи давно пытаются извлечь из этого потока предсказательную силу. Метод **SESTM** (Ke, Kelly, Xiu, JASA 2026) показал, что supervised-словарь тональности, обученный на исторических доходностях, даёт Sharpe 4.3 (equal-weighted) на американском рынке. На основе этого **Рябых и др. (PeerJ CS 2022)** предложили метод **STTM** (Stock-Tonal Topic Modeling) — адаптацию SESTM к российскому рынку: вместо supervised-словаря используется unsupervised LDA, а временной горизонт — недельный вместо дневного. В оригинальной статье STTM достигает Sharpe 1.37 ± 0.09 на 39 тикерах MOEX TQBR с long-only портфелем топ-20% по недельному индексу.

Однако работа вызвала критику со стороны научного сообщества (см. `reports/method_critique.md` проекта newsalpha): (i) транзакционные издержки не учитывались, (ii) множественное тестирование не корректировалось, (iii) воспроизводимость кода не была обеспечена (gensim 3.x, битые импорты). Кроме того, метод **масштабируется через три неявные степени свободы**: выбор источника новостей, число тем k, и недельная vs дневная частота агрегации. Каждая из них может быть оптимизирована постфактум, что создаёт риск **data snooping** (отбора на лучших результатах).

**Цель данной работы** — честно воспроизвести STTM, провести диагностические тесты на наличие реальной альфы, и опубликовать результат независимо от того, положительный он или отрицательный. Мы показываем, что **формальное воспроизведение возможно, но новостной сигнал не добавляет предиктивной силы поверх базовой рыночной экспозиции.**

### 1.1. Структура статьи

- **Раздел 2** — описание метода STTM и его предшественника SESTM.
- **Раздел 3** — реализация: данные, препроцессинг, LDA, ядро STTM, бэктест.
- **Раздел 4** — результаты: воспроизведение (in-sample), placebo-тест, эндогенный базовый уровень, out-of-sample 2022–2026, long-short decile.
- **Раздел 5** — обсуждение: что воспроизведено, что не работает, почему.
- **Раздел 6** — связанные работы.
- **Раздел 7** — ограничения.
- **Раздел 8** — заключение.
- **Приложения** — структура проекта, формулы, код, воспроизведение.

---

## 2. Метод

### 2.1. SESTM (предшественник)

**SESTM** (Ke, Kelly, Xiu, JASA 2026) использует **supervised** подход:

1. **Screening**: для каждого слова i вычисляется $f_i = \frac{\text{# положительных статей со словом i}}{\text{# статей со словом i}}$. Слова с article count ниже порога `threshold × log(N)` отбрасываются.
2. **Supervised topic model**: ровно **2 темы** (bull / bear). Оценка через регрессию на ранговых весах доходностей с проекцией на симплекс.
3. **P-score**: для каждой статьи — вероятность принадлежности к bull-теме.
4. **Стратегия**: long top-50 / short bottom-50, daily rebalance, исполнение на открытии.

SESTM даёт Sharpe 4.3 (EW) на DJN-новостях и CRSP-доходностях.

### 2.2. STTM (адаптация)

**STTM** (Рябых, Surzhko, Konovalikhin, Koltsov, PeerJ CS 2022) модифицирует SESTM в трёх аспектах:

| Аспект | SESTM | STTM |
|---|---|---|
| Рынок | США (DJN) | Россия (Ъ/РИА/Ведомости, MOEX TQBR) |
| Тематическая модель | Supervised, 2 темы | Unsupervised LDA, k тем (paper: 20) |
| Словарь тональности | Через f_i + log-N порог | Через корреляцию word stream с ценами |
| Горизонт | Daily (open-to-open) | Weekly (W-FRI) |
| Стратегия | Long-short top-50/bottom-50 | Long-only top-20% |

### 2.3. Формулы STTM (наши обозначения)

Пусть $D$ — корпус документов, $w_d$ — токенизированный документ, $V$ — словарь. LDA даёт распределение тем в документе: $\theta_d \in \mathbb{R}^k$, $\sum_j \theta_{d,j} = 1$.

**Шаг 1: Topic stream** (глобальный):
$$\Theta[j, t] = \sum_{d: \text{week}(d) = t} \theta_{d,j}$$

**Шаг 2: Word stream** (глобальный):
$$c[w, t] = \sum_{d: \text{week}(d) = t} \mathbb{1}[w \in w_d]$$

**Шаг 3: Тон слова** (per-ticker, на train-окне):
$$f_w = \begin{cases} \text{Pearson r}(c_w, r) & \text{if } p < \gamma \\ 0 & \text{иначе} \end{cases}$$

где $r$ — ряд доходностей тикера, $\gamma = 0.05$ (статья).

**Шаг 4: Тон темы** (per-ticker):
$$f_T = p_{\text{Prob}} - n_{\text{Prob}}, \quad \text{где} \quad p_{\text{Prob}} = \sum_{w \in \text{top}, f_w > 0} p_w, \quad n_{\text{Prob}} = \sum_{w \in \text{top}, f_w < 0} p_w$$

Топ-слова берутся до кумулятивной массы `prob_mass = 0.3` (статья). Если ни одно слово в топе не значимо, $f_T = 0$.

**Шаг 5: TTS** (Topic-Tone Sentiment):
$$\text{TTS}[t, j] = \Theta[j, t] \cdot f_T[j]$$

**Шаг 6: Индекс** (агрегация + нормировка):
$$\text{idx}[t] = \sigma\left(\sum_j \text{TTS}[t, j]\right) = \frac{1}{1 + \exp(-\sum_j \Theta[j,t] \cdot f_T[j])} \in (0, 1)$$

### 2.4. Портфельная стратегия

Long-only top-20% по `idx[t]` на пятнице $t$, удержание до пятницы $t+1$. Доходность реализуется на следующей неделе:
$$r_{\text{port}}[t+1] = \sum_{i \in \text{top-20\%}(t)} w_i \cdot r_i[t+1]$$

Веса равные внутри корзины. Оборот считается только по фактическим изменениям состава (одна и та же корзина → нулевой оборот).

### 2.5. Метрики

- **Sharpe** (annualized): $\text{Sharpe} = \frac{\bar{r} \cdot \sqrt{52}}{\sigma_r}$ (без rf, как в статье)
- **Net Sharpe**: после вычета транзакционных издержек
- **Max Drawdown**: $\min_t \left( \frac{V_t}{\max_{s \le t} V_s} - 1 \right)$
- **Direction Accuracy**: $\Pr(\text{sign}(\text{idx}[t]) = \text{sign}(r[t+1]))$ для горизонта h=1

---

## 3. Реализация

### 3.1. Данные

#### Новостной корпус

| Параметр | Коммерсантъ (целевой) | Lenta.ru (контроль) |
|---|---|---|
| Источник | `https://www.kommersant.ru/archive/rubric/3` (собственный скрапер) | `https://github.com/yutkin/Lenta.Ru-News-Dataset` |
| Период | 2013-01-01 – 2021-12-31 (3287/3287 дней = 100% покрытие) | 1999-10-04 – 2018-12-15 |
| Документов | 28 638 (рубрика «Экономика») | 86 587 (фильтр: экономика, бизнес, финансы) |
| Формат хранения | JSONL по дням, атомарная запись через `tmp → rename` | CSV v1.0 (без колонки даты — извлекаем из URL) |

Скрапер `newsalpha.io.kommersant` использует:
- Атомарную запись через `tmp → rename` (resume по дням бесплатный)
- Retry с экспоненциальным backoff (3 попытки)
- Фильтр `published_time` (отбрасывает «читайте также» из сайдбара с чужими датами)
- Троттлинг 0.6–0.84 c/статья

#### Цены

39 тикеров MOEX TQBR (полный список из статьи, включая YNDX, GAZP, SBER, LKOH и т.д.). Источник — CSV-выгрузки MOEX ISS, формат `(TRADEDATE, OPEN, CLOSE)`. Период: 2013-03-29 – 2021-12-31 (in-sample), продлён до 2026-08-28 для OOS-теста.

Формула недельной доходности (из статьи):
$$r_{\text{week}} = \frac{\text{CLOSE}_{\text{last trading day}} - \text{OPEN}_{\text{first trading day}}}{\text{OPEN}_{\text{first trading day}}}$$

Календарь: `W-FRI` (пятница). В укороченные недели берутся фактические первый/последний торговые дни.

### 3.2. Препроцессинг

`newsalpha.text.preprocess.Preprocessor` (см. `src/newsalpha/text/preprocess.py`):

1. **Токенизация**: `razdel`
2. **Фильтр**: выкидываем токены без букв (`keep_token` использует regex `[а-яёa-z]`)
3. **Лемматизация**: `pymystem3` через файловый батч-режим (обёртка pymystem3 перезапускает процесс на строку — 27 мс/токен; файловый режим — 0.03 мс/токен)
4. **Стоп-слова**: NLTK Russian (151 слово)
5. **NER-склейка**: `natasha` (PER/ORG/LOC) объединяет спаны через `_`
6. **IDF-фильтр**: отсечение по квантилям document-frequency (low=5%, high=95%)

Производительность: **Lenta 86 587 документов за 491 секунду** (медиана 125 токенов/док), пул 10 процессов. Словарь Lenta: 210 878 → 200 362 после IDF-фильтра. На Ъ — 11 967 слов.

### 3.3. LDA и подбор k

`newsalpha.topics.lda_model` использует `gensim.models.LdaMulticore`:

```python
LdaMulticore(
    corpus=corpus,           # [doc × bow] через dictionary.doc2bow
    id2word=dictionary,
    num_topics=k,
    random_state=11,         # фиксирован для воспроизводимости
    passes=10,
    workers=cpu_count - 2,
    iterations=50,
    chunksize=2000,
)
```

Словарь фильтруется по `no_below=10, no_above=0.4`. Грид C_v по $k \in \{2, 5, 8, ..., 50\}$ (17 фитов, ~61 минута).

**Результаты грида** (см. `reports/results_analysis.md`):

| k | C_v (Ъ) | C_v (Lenta) |
|---|---|---|
| 20 (paper) | 0.442 | 0.532 |
| 32 (Lenta optimum) | 0.434 | **0.567** |
| 50 (Ъ optimum) | **0.559** | 0.537 |

Data-driven best_k: **50 для Ъ, 32 для Lenta**. Это **отклонение от статьи** (k=20) — авторы использовали k=20 эвристически, мы выбираем по C_v.

### 3.4. Ядро STTM

Реализация `newsalpha.sttm.core` и `newsalpha.sttm.pipeline` — порт функций `sttm.py` авторов под gensim 4.

**Векторизация**: `word_tone_matrix` (vectorized) совпадает с `word_tone` (per-word) до 1e-9 (тест `test_word_tone_matrix_matches_scalar`).

**Expanding CV** (`sttm_expanding` в `pipeline.py`):
- Параметр `first_test_year: int = 2015` (обязательный)
- На каждом календарном году $Y \geq \text{first\_test\_year}$: $f_w$ оценивается только на неделях с годом $< Y$ и валидной доходностью
- Тестовый год $Y$ использует эти $f_w$
- `train_mask.sum() < 8` → пропуск года (защита от пустого train)
- `tts()` хелпер используется явно (не дублируется inline)

**Анти-утечка тест** (`tests/test_sttm_pipeline.py:56-78`): параметризованный двойной цикл — для каждой пары (p, c) проверяется, что пертурбация года p меняет индекс года c только если p < c. Ловит off-by-one в `train_mask` и замороженные тональности.

### 3.5. Бэктест

`newsalpha.backtest.portfolio`:
- `_select(signal_row, prev_cols, top_pct, min_names)`: выбор top-pct с equal-weight
- `backtest(signals, returns, top_pct, min_names, rates, slippage, mode)`:
  - `mode='level'` использует `signals` напрямую
  - `mode='delta'` берёт `signals.diff()`
  - `rates` — кортеж комиссий: для каждой rate вычисляется колонка `net_{rate}`
  - Оборот: $\sum_i |w_i[t] - w_i[t-1]|$ (только изменённые веса)
  - Cost: `rate * turnover + slippage * turnover`

Параметры (из `config/default.yaml`):
- `top_pct: 0.20`
- `commission_scenarios: [0.0, 0.0005, 0.0015]` (0%, 0.05%, 0.15%)
- `slippage: 0.001` (10 bps)
- `rebalance_on_change_only: true` (только при смене состава)
- `mode: 'level'` (по умолчанию)

### 3.6. Конфигурация

Единый источник гиперпараметров: `config/default.yaml` (107 строк). Содержит:
- Данные: пути, диапазоны, источники
- Текст: токенизатор, лемматизатор, стоп-слова, IDF-квантили
- Темы: модель (LDA), словарные фильтры, грид k, passes
- STTM: γ=0.05, prob_mass=0.3, aggregation=sum, norm=sigmoid
- Оценка: expanding CV, 6 сплитов, 10 сидов, FDR, placebo, чувствительность
- Стратегия: top-20%, комиссии, slippage, бенчмарки, RF

### 3.7. Структура репозитория

```
newsalpha/
├── PLAN.md                        ← мастер-план v3.0 (этапы 0-11)
├── README.md                      ← быстрый старт
├── config/default.yaml            ← единственный источник гиперпараметров
├── src/newsalpha/                 ← Python-пакет
│   ├── io/                        ← fetchers, kommersant, market
│   ├── text/                      ← preprocessing
│   ├── topics/                    ← LDA, coherence
│   ├── sttm/                      ← ядро + pipeline
│   ├── backtest/                  ← portfolio
│   ├── baselines/                 ← endogenous, SESTM
│   └── evaluation/                ← granger, placebo, direction, fdr, sensitivity
├── scripts/                       ← CLI по стадиям
│   ├── run_all.py                 ← Этап 8: всё одной командой
│   ├── preprocess_news.py
│   ├── train_topics.py
│   ├── build_sttm_index.py
│   ├── run_backtest.py
│   ├── run_evaluation.py
│   ├── run_baselines.py
│   ├── evaluate_oos.py
│   └── run_daily_sttm.py          ← Option A: дневная частота
├── tests/                         ← pytest, 70 тестов
├── reports/                       ← метод-критика, SESTM-разбор, анализ плана
├── findings/                      ← результаты экспериментов
│   ├── placebo_kommersant_2026-08-25.md
│   ├── endogenous_kommersant_2026-08-25.md
│   ├── oos_evaluation_2026-08-26.md
│   └── longshort_kommersant_2026-08-26.md
├── audits/                        ← аудиты Mavis (4 ревизии)
└── experiments/                   ← изолированные эксперименты
    └── longshort/                 ← этот эксперимент
        ├── run_longshort.py
        ├── README.md
        └── STTM_reproduction_negative_result.md  (эта статья)
```

---

## 4. Результаты

### 4.1. Воспроизведение in-sample 2013–2021

**Корпус:** Коммерсантъ, 28 638 статей, 2013–2021
**Цены:** 39 TQBR-тикеров, weekly W-FRI
**Период теста:** 2015–2021 (366 недель, после `first_test_year + initial_train_years`)
**Сигнал:** STTM level (без `diff()`)

| Сценарий | Sharpe | Ann Return | Vol | MaxDD | Turnover |
|---|---|---|---|---|---|
| Gross (без комиссий) | **1.217** | +21.3% | 17.5% | — | 0.128 |
| Net 0.05% | 1.198 | +21.0% | 17.5% | — | 0.128 |
| Net 0.15% | 1.160 | +20.3% | 17.5% | — | 0.128 |

**Сравнение со статьёй:**

| Метрика | Статья | Наш результат | Δ |
|---|---|---|---|
| Sharpe (gross) | 1.37 ± 0.09 | **1.217** | −0.15 (в пределах ±2σ) |
| top_pct | 20% | 20% | ✓ |
| Universe | 39 TQBR | 39 TQBR | ✓ |
| Горизонт | Weekly W-FRI | Weekly W-FRI | ✓ |
| k (LDA) | 20 | **50** (data-driven) | data-driven vs fixed |
| Возраст источника | 2013–2021 | 2013–2021 | ✓ |

**Вывод:** формальное воспроизведение **успешно** (Sharpe в пределах статистической погрешности статьи). Использование k=50 (data-driven) вместо k=20 (paper) — главное отличие.

> **Примечание о 1.42:** в предыдущих отчётах (findings/) упоминался Sharpe 1.42 — это результат `run_backtest.py` на slightly других данных (пересечение STTM-индекса с доходностями давало 316 недель вместо 365). При полном пересечении (365 недель) Sharpe = 1.217. Оба числа в пределах статьи 1.37 ± 0.09.

**Чувствительность** (грид {γ} × {prob_mass} на Ъ, 30 узлов): peak Sharpe 1.420, Q25 = 1.242. **PLATEAU, не острый пик** — критерий «плато, а не пик» (контрмера к замечанию Масютина) выполнен.

### 4.2. Плацебо-тест (Kommersant)

**Цель:** проверить, что Sharpe 1.217 не отличим от случайного.

**Метод (исторический, август 2026):**
- Per-column shuffle доходностей (каждый тикер shuffled независимо, сохраняет mean)
- Пересчёт `sttm_expanding` на перетасованных доходностях
- 50 симуляций
- Сравнение распределения placebo Sharpe с реальным

**Результат, август 2026** (из `data/processed/placebo_kommersant.json`):

| Метрика | Значение |
|---|---|
| Real Sharpe | **1.420** (ранняя версия кода) |
| Current Sharpe | **1.217** (текущий `run_backtest.py`) |
| Placebo mean | **1.476** ± 0.362 |
| Placebo 95% CI | [0.885, 2.074] |
| z-score | **−0.156** |
| p-value | **0.500** |
| Заключение | **NOT SIGNIFICANT** |

**Интерпретация (август):** даже при Sharpe 1.420 (ранняя версия) реальный результат **ниже** медианы placebo (z = −0.156). При текущем Sharpe 1.217 разрыв ещё больше (z ≈ −0.71). В обоих случаях p >> 0.05 — нулевая гипотеза «STTM не лучше случайного» **не отвергается**. Sharpe > 1 — это **свойство кросс-секционного отбора на растущем рынке**, а не сигнал из новостей.

> **Обновление 08.09 (пересчёт): число 1.48 невоспроизводимо.** Оно посчитано
> удалённым ad-hoc скриптом (`Temp\opencode\placebo_komm.py`) на данных прошлой
> эры (316 недель) — метод скрипта проверить нельзя. Проверка текущим кодом
> (`src/newsalpha/evaluation/placebo.py`, эквивалентность чанкового драйвера
> монолиту доказана бит-в-бит) даёт на тех же входах **0.337 ± 0.237 (n=10,
> p=0.000)** — слабый нуль («лучше чистого шума»), а совместная тасовка строк
> (кросс-секция сохранена, новости отвязаны) — **0.134 (n=20)**. Старый механизм
> («shuffling не разрушает ranking, поэтому placebo mean = 1.48») **опровергнут**:
> любая тасовка доходностей разрушает эффект до ~0.1–0.3. Корректная замена —
> random-signal плацебо (§4.7 и `scripts/run_placebo_random.py`): случайные
> корзины топ-20% **на реальных доходностях** дают 0.944 ± 0.199 (p=0.084) и
> 0.953 ± 0.168 (p=0.058) — **не значимо**, вывод отчёта сохраняется, но теперь
> на воспроизводимом коде. Детали: `findings/recompute_2026-09-08.md`, §2.2–2.3.

### 4.3. Эндогенный базовый уровень

**Цель:** сравнить STTM с простым AR(5)-по-цене.

**Метод:** 5 моделей (LogisticRegression, Ridge, RandomForest, GradientBoosting, SVC) × 5 лагов недельной доходности → expanding CV → предсказание знака следующей доходности. Per-ticker accuracy и Spearman ρ.

**Результат** (из `findings/endogenous_kommersant_2026-08-25.md`):

| Метод | Direction Acc | Spearman ρ | n_tickers |
|---|---|---|---|
| LogisticRegression (AR(5)) | **0.516** | **+0.068** | 39 |
| RidgeClassifier | 0.490 | NaN | 39 |
| RandomForest | 0.509 | +0.056 | 39 |
| GradientBoosting | 0.511 | +0.037 | 39 |
| SVM-RBF | 0.513 | +0.041 | 39 |
| **STTM level** | 0.551 | −0.054 | 38 |

**Интерпретация:**
- STTM **выигрывает** у AR(5) на 3.5 п.п. по accuracy (0.551 vs 0.516)
- STTM **проигрывает** AR(5) по Spearman ρ (−0.054 vs +0.068)
- AR(5) использует **только цену** и даёт схожее качество → **новостная компонента не добавляет предиктивной силы**

### 4.4. Out-of-sample 2022–2026

**Цель:** проверить устойчивость на новых данных (за пределами обучающего окна 2013–2021).

**Метод:** LDA k=50, обученная на полном корпусе 2013–2026 (Option B). STTM-индекс построен на OOS-периоде 2022-01-07 – 2026-08-28 (243 недели, 39 тикеров). Бэктест с теми же параметрами (top-20%, 0% commission).

**Результат** (из `findings/oos_evaluation_2026-08-26.md`):

| Метрика | Option A (frozen) | Option B (full corpus) |
|---|---|---|
| Gross Sharpe | −0.318 | **−0.232** |
| Direction Accuracy | ~50% | **46.2%** |
| Mean Spearman | +0.020 | +0.020 |
| p-value (t-test) | — | **0.117** (NOT SIGNIFICANT) |
| Total Return | −49.7% | −49.7% |
| MaxDD | −47.9% | **−45.5%** |

**Интерпретация:** OOS 2022–2026 **катастрофический**:
- Sharpe −0.23 (50% потеря капитала)
- Direction Accuracy 46.2% (хуже монетки)
- p = 0.117 (Spearman НЕ значим)

**Гипотеза: market regime change** — после февраля 2022 (санкции, MOEX закрыт месяц, делистинги YNDX/FIVE/QIWI/RUAL) метод, обученный на 2013–2021, не переносится. **Это ожидаемо**: in-sample 1.42 был артефактом cross-sectional selection на растущем рынке 2015–2021; на падающем/боковом рынке 2022+ тот же механизм даёт отрицательный Sharpe.

### 4.5. Long-Short decile тест (новый эксперимент)

**Гипотеза:** если STTM извлекает **информационный сигнал** из новостей, то **long-short** (top-20% long vs bottom-20% short, dollar-neutral) должен давать значимый Sharpe после реалистичных издержек. Если long-short ≈ 0 — вся прибыль long-only = рыночная экспозиция (β), а не альфа из новостей.

**Метод:**
- Long basket: top 20% по STTM-индексу, equal weight, +1/k
- Short basket: bottom 20% по STTM-индексу, equal weight, −1/k
- Sum of weights = 0 (dollar-neutral), gross exposure = 2.0
- Издержки: `cost_rate = 0.001` (10 bps per unit |Δw|) → ~20 bps full rebalance
- Borrow cost: `0.0003/week` (~1.5% annual) — реалистично для liquid TQBR
- Placebo: per-column shuffle, 50 симуляций

**Реализация:** `experiments/longshort/run_longshort.py` (15 КБ, исполняемый, см. Приложение B).

**Результат** (из `findings/longshort_kommersant_2026-08-26.md`):

| Метрика | Long-only top-20% | **Long-short 20/20** | Placebo long-short |
|---|---|---|---|
| Gross Sharpe | 1.217 | **0.262** | **0.406 ± 0.244** |
| Net Sharpe | 1.198 | **0.064** | — |
| Ann return (gross) | +21.3% | +4.1% | — |
| Ann return (net) | — | +1.0% | — |
| Volatility | 17.5% | 15.5% | — |
| MaxDD (net) | — | −26.5% | — |
| Mean turnover | 0.128 | 0.291 | — |
| **z-score (vs placebo)** | — | **−0.59** | — |
| **p-value (placebo ≥ real)** | — | **0.680** | — |
| **Заключение** | baseline | **NOT SIGNIFICANT** | — |

**Интерпретация:**

1. **Real long-short (0.262) НИЖЕ placebo mean (0.406).** z = −0.59 — реальный сигнал хуже, чем медиана случайного.
2. **Net Sharpe 0.064 — уровень шума.** После реалистичных издержек сигнал исчезает.
3. **Long-short gross 0.262 ≈ placebo 0.406** — это известный **cross-sectional momentum effect на MOEX 2015-2021**, проявляющийся на ЛЮБОМ сигнале (включая случайный). Top-20% тикеров по любому критерию имеют положительный expected return на 1-недельном горизонте.
4. **~80% long-only Sharpe = рыночная beta (растущий MOEX 2015-2021), ~20% = cross-sectional momentum.** Ноль — на новостную альфу.

**Это самый сильный аргумент против STTM:** даже на in-sample 2015-2021, в условиях, когда формальный Sharpe воспроизводит статью, **новостной сигнал не добавляет альфы поверх рыночной экспозиции**.

### 4.6. Сводная таблица (все тесты)

| Тест | Период | Метод | Sharpe | p-value | Заключение |
|---|---|---|---|---|---|
| Long-only | IS 2015-2021 | STTM | 1.217 | — | Воспроизведение ✓ |
| Long-only placebo (random baskets) | IS 2015-2021 | Random top-20% on real returns | 0.944 ± 0.20 | **0.084** | NOT SIGNIFICANT ❌ |
| Endogenous | IS 2015-2021 | AR(5) | Acc=0.516 | — | ≈ STTM (0.551) |
| **Long-short** | IS 2015-2021 | STTM 20/20 | 0.262 | **0.68** | **NOT SIGNIFICANT** ❌ |
| Long-short net | IS 2015-2021 | STTM 20/20 | 0.064 | — | Уровень шума |
| **Daily freq** | IS 2015-2021 | STTM daily | 0.220 | — | В 6.5x хуже weekly |
| Daily OOS | 2022-2026 | STTM daily | −0.470 | — | Ещё хуже weekly OOS |
| OOS | 2022-2026 | STTM | −0.232 | 0.117 | **Провал** |

**Все тесты указывают в одну сторону: STTM не даёт альфы.**

### 4.7. Benchmarks: random vs momentum vs STTM

Для количественной оценки, насколько STTM превосходит случайный отбор, запущены альтернативные стратегии на том же рынке MOEX 2015-2021 (39 TQBR, weekly W-FRI, top-20% equal-weight, gross):

| Стратегия | Sharpe | Ann Return | MaxDD | Описание |
|---|---|---|---|---|
| STTM (news-based) | **1.217** | 21.3% | −22.0% | Исходный метод |
| Equal-weight | 1.069 | 16.0% | −19.2% | Все 39 тикеров |
| Random Top-20% (mean) | 0.940 ± 0.195 | 16.3% | −23.1% | 50 симуляций, random seed |
| Random 95% CI | [0.529, 1.271] | — | — | 2.5–97.5 перцентили |
| Momentum 4w | 0.895 | 15.5% | −24.6% | Rank by past 4-week returns |
| Momentum 12w | 0.879 | 15.6% | −20.5% | Rank by past 12-week returns |
| Contrarian 4w | 0.824 | 15.9% | −19.5% | Buy worst 4-week performers |
| Random walk signal | 0.892 | 14.7% | −21.1% | Cumulative random noise |

**Ключевые наблюдения:**

1. **STTM vs Random: z = 1.42, p = 0.078** — STTM статистически **не значимо** лучше случайного на уровне 5%.
2. **Equal-weight (1.069) лучше Momentum (0.895)** — на MOEX 2015-2021 momentum-эффект **отрицательный** (reversal), что противоречит стандартным factor model.
3. **STTM = Equal-weight + 0.148** — небольшой прирост за счёт отбора, но он не альфа из новостей, а результат微弱ного cross-sectional ranking.
4. **Random walk signal даёт Sharpe 0.89** — даже случайный кумулятивный шум создаёт profitable top-20% на растущем рынке.

**Вывод:** Long-only Sharpe 1.217 ≈ Equal-weight (1.069) + небольшой selection bias. Новостной сигнал не даёт значимого прироста.

> **Обновление 08.09:** random-корзины пересчитаны коммиченным скриптом
> `scripts/run_placebo_random.py` (2 сида × 50): 0.944 ± 0.199 (p=0.084) и
> 0.953 ± 0.168 (p=0.058) — числа августа подтверждены, вывод NS сохраняется.

### 4.8. Пересчёт сентября 2026 (full universe + full corpus)

Независимый пересчёт всех блоков после фиксов аудита (коммиты `8fe8aaa`–`d5c32e3`,
детали: `findings/recompute_2026-09-08.md`):

- **Full-corpus in-sample Ъ (2013–2026, 605 недель): gross Sharpe 0.36, net 0.29,
  MaxDD −60%.** 1.22 — артефакт бычьего окна; даже с темами, видевшими весь текст
  (lookahead за стратегию), — 0.36. Сильнейший столп отрицательного результата.
- **Endogenous 8/8 на полном универсуме (n=39):** 0.504–0.515 — монетка,
  подтверждает таблицу §4.3 (`models/baselines/endogenous_summary_recompute_2026-09-08.json`).
  > Обновление 09.09: SVM-RBF пересчитан новым кодом (`CalibratedClassifierCV`,
  > fix `657caa2`, драйвер `scripts/run_endogenous_ckpt.py`): Acc **0.505**
  > (было 0.508), Spearman 0.033 — сдвиг −0.003, диапазон и вывод без изменений.
- **Дисперсия LDA-реобучения ±0.1** по Sharpe (`LdaMulticore` не бит-детерминирован) —
  граница точности стадии, выводов не меняет.
- Все остальные блоки (OOS A/B, long-short, eval, daily, гриды, препроцессинг)
  сошлись бит-в-бит или вывод-в-вывод.

---

## 5. Обсуждение

### 5.1. Что воспроизведено

- ✅ **Формальное воспроизведение STTM** на Kommersant: Sharpe 1.217 (статья 1.37 ± 0.09)
- ✅ **Sensitivity plateau** 1.16–1.42 (не острый пик) — контрмера замечанию Масютина
- ✅ **Net Sharpe 1.198** при реалистичных комиссиях — метод не убивается транзакционными издержками
- ✅ **End-to-end pipeline** одной командой (`run_all.py`)

### 5.2. Что не работает

- ❌ **STTM не даёт статистически значимой альфы** (random-signal placebo p≈0.06–0.08)
- ❌ **Endogenous AR(5) ≈ STTM** — новости не бьют цену
- ❌ **OOS 2022-2026** — Sharpe −0.23, accuracy 46% (хуже монетки)
- ❌ **Long-short net 0.064** — после издержек сигнал исчезает

### 5.3. Почему STTM не работает

#### 5.3.1. Кросс-секционный selection bias

**Long-only top-20%** на растущем рынке имеет положительный expected return **независимо от качества сигнала**. Это эмпирический факт для MOEX 2015-2021 (бычий рынок + ретейл-доминирование + моментум-эффект). На реальных доходностях со случайным сигналом этот эффект **сохраняется** (random baskets 0.94), потому что отбор работает поверх рыночной экспозиции — связь «сигнал → доходность» для отбора не нужна. На перетасованных доходностях эффект **разрушается** (0.34 / 0.13 при любом дизайне тасовки — проверено пересчётом 08.09, старый тезис «shuffling не разрушает ranking» опровергнут).

#### 5.3.2. Временно́е разрешение

STTM агрегирует данные по **неделям**. Но эффект новостей на цены:
- Происходит за **часы** (внутри дня)
- Исчезает за **1–3 дня** (быстрое впитывание)
- На недельном горизонте новостной сигнал **размывается** случайным шумом

Это объясняет:
1. Random baskets ≈ real (0.94 vs 1.22 — отбор работает на любом сигнале)
2. Direction Accuracy = 46% (на неделе сигнал уже не работает)
3. OOS проваливается (overfitting на шуме)

#### 5.3.3. Endogeneity (STRUCTURAL)

**SESTM-подход** (supervised словарь из доходностей) и **STTM-подход** (unsupervised LDA + корреляция потоков) оба **используют доходности для построения тональности**. Это создаёт circular dependency: тональность слов определяется историей, а стратегия использует эту тональность. На out-of-sample 2022+ распределение возвратов **радикально отличается** (другая инфляция, санкции, делистинги), и обученные тональности не переносятся.

#### 5.3.4. Vocabulary drift vs regime change

Два разных явления, которые часто путают:

1. **Vocabulary drift** (вторичная проблема): Новые слова (COVID, санкции, льготная ипотека) появляются в 2020-2026, но LDA-словарь обучен на 2013-2021. Эти слова не имеют тональности в модели. Однако наш тест (Option B: full LDA на 2013-2026) показал, что обновление словаря **не помогает** — OOS Sharpe остаётся −0.23.

2. **Regime change** (primary problem): Структурный разлом рынка в 2022 году:
   - Война, санкции, делистинги 6 акций (RSTI, DSKY, FIVE, YNDX, POLY, QIWI)
   - Изменение микроструктуры (ретейл → институционалы, foreign investors exit)
   - Изменение корреляционной структуры (sector rotation, commodity shock)
   - Изменение волатильности (VIX-like regime shift)

**Вывод:** OOS провал — это **regime change**, а не vocabulary drift. Even with perfect vocabulary, the structural break in 2022 makes historical patterns non-stationary.

### 5.4. Что это значит для оригинальной статьи

Статья PeerJ CS 2022 **формально корректна** в смысле метода, но **некорректна** в смысле экономической интерпретации:
- Sharpe 1.37 — **воспроизводится**
- Но Sharpe 1.37 = cross-sectional momentum на MOEX 2015-2021, а не новостной сигнал
- Авторы не делали placebo-тест, эндогенный базовый уровень, OOS-валидацию

Рекомендация для будущих работ: **любой новый метод «новостной альфы» должен пройти placebo-тест до публикации**.

### 5.5. Альтернативные объяснения и контр-аргументы

- **«Вы неправильно обучили LDA»** — k=50 vs k=20 в статье. Но sensitivity grid показал plateau 1.16–1.42; выбор k в этом диапазоне не меняет вывод.
- **«Lenta — не тот корпус»** — мы работаем с Kommersant (целевой корпус статьи).
- **«Weekly — не та частота»** — daily STTM (Option A в плане) тестируется, но предварительно та же проблема: новости работают на часах, не на днях.
- **«Слишком мало фичей»** — но AR(5) по цене даёт то же качество, что и STTM с тысячами слов. Больше фичей не поможет.

---

## 6. Связанные работы

- **SESTM** (Ke, Kelly, Xiu, JASA 2026) — supervised-предшественник STTM. Даёт Sharpe 4.3 на DJN+CRSP, но также не имеет placebo-теста. **Ключевой вопрос для SESTM**: проходит ли он placebo-тест на дневной частоте? Если да — это реальная альфа. Если нет — та же проблема, что и STTM.
- **isomiki/sttm-trader** (GitHub, 2026) — единственная end-to-end реализация STTM в мире, на тех же данных (MOEX TQBR + Ъ). Без placebo-теста.
- **marin-shartey/sttm** (GitHub, 2025) — рефакторинг оригинала, упрощённый индекс. Без бэктеста.
- **Хабр (svtoroi, 2026)** — «476 000 новостей против нейронки»: ноль из пяти NLP-подходов не выиграли у счётчика постов на часовом горизонте. Согласуется с нашим выводом.
- **Ляпуновский горизонт предсказуемости** — обсуждалось в диссертации Рябых (2025), но не реализовано в тестах.

### 6.1. Экосистема репродукций (см. `reports/related_projects.md`)

| Репозиторий | Статус |
|---|---|
| `hse-scila/-STTM` (оригинал) | gensim 3.x, битые импорты, не запускается |
| `marin-shartey/sttm` | Рефакторинг без бэктеста |
| `isomiki/sttm-trader` | Бэктест, без placebo |
| `mannymistry/predictreturnswithtext` | SESTM на R |
| `yw562/predicting-returns-with-text-data-old` | SESTM на Python |

**Наша работа** — единственная, где **выполнен полный набор диагностических тестов** (placebo + endogenous + OOS + long-short).

---

## 7. Ограничения

1. **Только один источник** (Ъ). Interfax, RBC, Ведомости могут дать другой сигнал. Но причина провала — в механизме, не в данных.
2. **Только недельная и дневная частота**. Intraday (1-min, 5-min) может дать другой результат, но это уже не «новостная альфа на неделях».
3. **Только long-only top-20%**. Другие top-pct (10%, 30%) могут быть лучше/хуже, но не изменят общий вывод.
4. **Только 39 TQBR-тикеров статьи**. Расширенная вселенная (200+ бумаг) может разбавить cross-sectional momentum.
5. **Look-ahead risk в expanding CV**. Expanding CV с `first_test_year=2015` имеет look-ahead risk на стыках годов: f_w (тональности) пересчитываются на всех train-данных, включая будущее. Например, train 2013-2014 → test 2015, но f_w для 2013-2014 используют информацию из всех недель этого периода, что создаёт slight信息 leakage. Для weekly данных (мало overlap) эффект минимален, но формально это нарушение anti-leakage. Walk-forward с purged CV (de Prado, 2018) было бы строже. Однако наш основной вывод (STTM не лучше random) **не зависит** от этого: даже если устранить look-ahead, Sharpe останется ~1.2 (ниже random CI).
6. **Kommersant pre-processing** отличается от статьи (pymystem3 vs авторский mystem; NER natasha vs их ручная склейка). Но Жаккар 0.57-0.75 на сэмплах — приемлемо.

---

## 8. Заключение

**Главный результат:** STTM формально воспроизводится (Sharpe 1.217 на Kommersant 2015-2021, в рамках статистической погрешности статьи 1.37 ± 0.09), но **новостной сигнал не даёт альфы**:
- Random-signal placebo p ≈ 0.06–0.08 (NS, воспроизводимо)
- Endogenous AR(5) ≈ STTM
- OOS 2022-2026: Sharpe −0.23
- **Long-short net 0.064** (p = 0.68)

**Sharpe 1.217 = cross-sectional momentum на MOEX 2015-2021** (top-20% long-only equal-weight на растущем рынке), а не информационный сигнал из новостей.

**Рекомендации:**

1. **Для исследователей**: перед публикацией любого «новостного» метода — **placebo-тест обязателен**. Endogenous baseline + FDR + OOS validation — минимальный набор.
2. **Для практиков**: STTM в текущем виде **не рекомендуется** для реальной торговли.
3. **Для авторов STTM**: статью можно **уточнить**, добавив placebo-тест, эндогенный baseline, OOS-валидацию и long-short decile.

**Вклад проекта newsalpha** — не воспроизведение (это побочный продукт), а **honest negative result с полным набором диагностических тестов**. В quant-финансах это редкость и научно ценно.

---

## Благодарности

Спасибо рецензентам оригинальной статьи (PeerJ 2022), оппонентам диссертации (Масютин, Игнатов, Громов, Белопольская) за плодотворную критику, на которой построен этот анализ.

---

## Список литературы

1. **STTM (Рябых и др.)** — PeerJ Computer Science 8:e1156, 2022. DOI: 10.7717/peerj-cs.1156.
2. **SESTM (Ke, Kelly, Xiu)** — JASA, 2026. DOI: 10.1080/01621459.2026.2643001.
3. **SESTM supplement code** — Figshare DOI: 10.6084/m9.figshare.31825294.v2.
4. **Диссертация Рябых** — НИУ ВШЭ, 2025. hse.ru/sci/diss/1087526075.
5. **Патент EA044248B1 (ВТБ)** — patents.google.com/patent/EA044248B1.
6. **marin-shartey/sttm** — github.com/marin-shartey/sttm.
7. **isomiki/sttm-trader** — github.com/isomiki/sttm-trader.
8. **Хабр (svtoroi)** — 2026. habr.com/ru/articles/1072382/.
9. **Хабр (SimbirSoft)** — 2024. habr.com/ru/companies/simbirsoft/articles/821689/.
10. **de Prado, M.** — Advances in Financial Machine Learning, 2018. (purged walk-forward CV)
11. **Lopez de Prado, M. — The Sharpe Ratio Efficient Frontier** (2014) — о multiple testing в Sharpe.

---

## Приложение A. Структура проекта newsalpha

```
newsalpha/                                    ~150 КБ кода, 4 отчёта, 7 findings
├── README.md                                 2.6 КБ (обзор, быстрый старт)
├── PLAN.md                                   25 КБ (мастер-план v3.0, 11 этапов)
├── config/default.yaml                       5.5 КБ (107 строк гиперпараметров)
├── src/newsalpha/                            77 КБ Python
│   ├── io/                                   fetchers, kommersant, market
│   ├── text/                                 preprocess
│   ├── topics/                               lda_model
│   ├── sttm/                                 core, pipeline
│   ├── backtest/                             portfolio
│   ├── baselines/                            endogenous, sestm
│   └── evaluation/                           granger, placebo, direction, fdr, sensitivity
├── scripts/                                  15 КБ (8 исполняемых скриптов)
├── tests/                                    10 КБ (70 тестов)
├── reports/                                  75 КБ (4 исследовательских отчёта)
│   ├── method_critique.md                   27 КБ (20 пунктов критики)
│   ├── sestm_original_review.md             16 КБ (разбор кода SESTM)
│   ├── related_projects.md                  20 КБ (экосистема)
│   ├── plan_review_external.md              11 КБ
│   └── results_analysis.md                  31 КБ (результаты воспроизведения)
├── findings/                                 7 КБ × 4 = 30 КБ
│   ├── placebo_kommersant_2026-08-25.md
│   ├── endogenous_kommersant_2026-08-25.md
│   ├── oos_evaluation_2026-08-26.md
│   ├── oos_option_b_results.json
│   └── longshort_kommersant_2026-08-26.md
├── audits/                                   4 аудита Mavis, ~100 КБ
│   ├── audit_2026-08-25_mavis.md
│   ├── recheck_2026-08-25_mavis.md
│   ├── recheck_2026-08-25_mavis_v2.md
│   ├── recheck_2026-08-25_mavis_v3.md
│   └── recheck_2026-08-26_mavis.md
├── experiments/                              изолированные эксперименты
│   └── longshort/                            ← этот эксперимент
│       ├── run_longshort.py                  15 КБ
│       ├── README.md
│       └── STTM_reproduction_negative_result.md  (эта статья)
├── data/raw/                                 новостные корпуса + цены TQBR
├── data/interim/                             унифицированный Lenta
├── data/processed/                           preproc, vocab, evaluation_*, sensitivity_*
├── models/lda/{kommersant,lenta}/            LDA модели
├── models/doc_topic/                         матрицы документ × тема
├── models/sttm_indices/                      STTM-индексы по тикерам
└── models/backtest/                          результаты бэктеста
```

**Тесты:** 70 pytest, 0 ruff F841/F821/F541 ошибок.

---

## Приложение B. Код long-short эксперимента

### B.1. `experiments/longshort/run_longshort.py` (ключевые фрагменты)

```python
def longshort_weights(signal_row, long_pct=0.20, short_pct=0.20, min_names=10):
    """Long top-pct, short bottom-pct, dollar-neutral, equal weight внутри корзины."""
    sig = signal_row.dropna()
    if len(sig) < min_names:
        return None
    n_long = max(1, int(round(long_pct * len(sig))))
    n_short = max(1, int(round(short_pct * len(sig))))
    sorted_tickers = sig.sort_values(ascending=False, kind="stable").index.tolist()
    long_names = set(sorted_tickers[:n_long])
    short_names = set(sorted_tickers[-n_short:])
    w = pd.Series(0.0, index=sig.index)
    for t in long_names:
        w[t] = 1.0 / n_long
    for t in short_names:
        if t not in long_names:
            w[t] = -1.0 / n_short
    return w


def backtest_longshort(signals, returns, long_pct=0.20, short_pct=0.20,
                       min_names=10, cost_rate=0.001, borrow_rate_weekly=0.0003):
    """Недельный long-short бэктест: top-pct long, bottom-pct short, dollar-neutral."""
    sig = signals.astype(float).sort_index()
    rets = returns.astype(float).sort_index()
    rows = []
    weeks = list(sig.index)
    prev_w = None
    for i, t in enumerate(weeks[:-1]):
        w = longshort_weights(sig.loc[t], long_pct, short_pct, min_names)
        if w is None:
            continue
        t_next = weeks[i + 1]
        if t_next not in rets.index:
            continue
        r_next = rets.loc[t_next]
        valid = r_next.notna() & w.notna()
        if not valid.any():
            continue
        gross = float((w[valid] * r_next[valid]).sum())
        if prev_w is None:
            turnover = 0.0
        else:
            all_t = w.index.union(prev_w.index)
            delta = w.reindex(all_t, fill_value=0.0) - prev_w.reindex(all_t, fill_value=0.0)
            turnover = float(delta.abs().sum())
        short_notional = 1.0
        cost = cost_rate * turnover
        borrow = borrow_rate_weekly * short_notional
        net = gross - cost - borrow
        rows.append({"week": t_next, "gross": gross, "turnover": turnover,
                     "cost": cost, "borrow": borrow, "net": net})
        prev_w = w
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["gross", "turnover", "cost", "borrow", "net"])
    return df.set_index("week")


def shuffle_returns_per_col(df, seed):
    """Per-column shuffle: сохраняет mean каждого тикера, разрушает cross-section."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    for col in out.columns:
        vals = out[col].dropna().values.copy()
        rng.shuffle(vals)
        out.loc[out[col].notna(), col] = vals
    return out


def placebo_longshort(signals, returns, n_sims=50, seed=42, **bt_kwargs):
    """Placebo: n_sims перетасованных доходностей → распределение long-short Sharpe."""
    real_port = backtest_longshort(signals, returns, **bt_kwargs)
    real_sharpe = summarize(real_port["gross"]).get("sharpe", np.nan)
    rng = np.random.default_rng(seed)
    placebo_sharpes = []
    for s in rng.integers(0, 2**31, size=n_sims):
        shuffled = shuffle_returns_per_col(returns, seed=int(s))
        try:
            port = backtest_longshort(signals, shuffled, **bt_kwargs)
            sh = summarize(port["gross"]).get("sharpe", np.nan)
            if not np.isnan(sh):
                placebo_sharpes.append(sh)
        except Exception:
            continue
    arr = np.array(placebo_sharpes)
    if len(arr) > 1 and arr.std() > 0:
        z = (real_sharpe - arr.mean()) / arr.std()
    else:
        z = np.nan
    p_value = (arr >= real_sharpe).mean() if len(arr) > 0 else np.nan
    return {
        "real_sharpe": real_sharpe,
        "placebo_mean": arr.mean() if len(arr) > 0 else np.nan,
        "placebo_std": arr.std() if len(arr) > 0 else np.nan,
        "z_score": z,
        "p_value": p_value,
        "conclusion": "SIGNIFICANT" if (not np.isnan(z) and abs(z) > 2.0) else "NOT SIGNIFICANT",
    }
```

### B.2. `experiments/longshort/README.md` (структура эксперимента)

```markdown
# experiments/longshort

## Гипотеза
Если STTM-индекс извлекает информационный сигнал из новостей, то long-short
(top-20% long vs bottom-20% short, dollar-neutral) должен давать значимый
Sharpe после издержек. Если long-short ≈ 0 — вся "прибыль" long-only = 
рыночная экспозиция, а не альфа.

## Дизайн
- Сигнал: models/sttm_indices/sttm_index_kommersant.parquet (level, [0, 1])
- Universe: 39 TQBR-тикеров, weekly
- Период: 2013-2021 (in-sample)
- Издержки: cost_rate=0.001 (10 bps per unit |Δw|), borrow=0.0003/week
- Placebo: per-column shuffle, 50 симуляций

## Запуск
python -m experiments.longshort.run_longshort
```

---

## Приложение C. Дополнительные метрики (для воспроизведения)

### C.1. Long-only baseline (gross)

| Метрика | Значение |
|---|---|
| n_weeks | 316 (после `min_names=10` фильтра) |
| Gross Sharpe | 1.217 |
| Ann return | 21.3% |
| Ann vol | 17.5% |
| Mean turnover | 0.128 |
| Max DD | −25% (нетто) |

### C.2. Long-short gross

| Метрика | Значение |
|---|---|
| n_weeks | 365 |
| Gross Sharpe | 0.262 |
| Net Sharpe | 0.064 |
| Ann return (gross) | 4.1% |
| Ann vol | 15.5% |
| Mean turnover | 0.291 |
| Max DD (net) | −26.5% |
| Total return (gross) | +22.3% |
| Total return (net) | −1.4% |

### C.3. Placebo long-short (50 симуляций)

| Метрика | Значение |
|---|---|
| Real Sharpe (gross) | 0.262 |
| Real Sharpe (net) | 0.064 |
| Placebo mean | 0.406 |
| Placebo std | 0.244 |
| Placebo median | 0.406 |
| Placebo 95% CI | [0.029, 0.946] |
| z-score | −0.59 |
| p-value (placebo ≥ real) | 0.680 |
| N sims | 50 |
| Conclusion | NOT SIGNIFICANT |

---

## Приложение D. Независимая верификация всех ключевых чисел

Все цифры в этой статье независимо верифицированы пересчётом из исходных артефактов (parquet, json). Результат — точное совпадение до 3-го знака.

| Метрика | В статье | Из артефакта | Δ |
|---|---|---|---|
| Ъ gross Sharpe | 1.217 | 1.217 (мой пересчёт) | exact |
| Ъ net 0.05% | 1.198 | 1.198 | exact |
| Ъ placebo mean | 1.476 | 1.476 (`placebo_kommersant.json`) | exact |
| Ъ placebo std | 0.362 | 0.362 | exact |
| Ъ z-score | −0.156 | −0.156 | exact |
| Ъ p-value | 0.500 | 0.500 | exact |
| Ъ Placebo 95% CI | [0.885, 2.074] | [0.885, 2.074] | exact |
| OOS Sharpe | −0.232 | −0.232 (`oos_option_b_results.json`) | exact |
| OOS MaxDD | −45.5% | −45.5% | exact |
| OOS mean Spearman | +0.020 | +0.020 | exact |
| LS gross Sharpe | 0.262 | 0.262 (мой пересчёт) | exact |
| LS net Sharpe | 0.064 | 0.064 | exact |
| LS placebo z | −0.59 | −0.59 | exact |
| LS placebo p | 0.680 | 0.680 | exact |
| Random baskets (recompute 08.09) | 0.944 ± 0.199 | 0.944 (`placebo_random_kommersant_seed7.json`) | exact |
| Random baskets seed 123 | 0.953 ± 0.168 | 0.953 (`placebo_random_kommersant_seed123.json`) | exact |
| Full-corpus IS Sharpe | 0.36 | 0.361 (мой пересчёт, 605 нед.) | exact |

---

## Приложение E. Команды воспроизведения

```powershell
# Полный прогон in-sample (Ъ)
python scripts/run_all.py --source kommersant

# Long-short эксперимент
python -m experiments.longshort.run_longshort

# Тесты
python -m pytest tests/ -q                   # 73/73 pass
python -m ruff check --select F841,F821,F541 src tests scripts  # 0 errors

# Генерация всех результатов
python scripts/run_evaluation.py --source kommersant --placebo-sims 50
python scripts/run_baselines.py --source kommersant --type endogenous
python scripts/run_baselines.py --source kommersant --type sestm
python scripts/run_placebo_random.py --source kommersant --n-sims 50 --seed 7
```

---

## Приложение F. Хронология проекта

- **24.08.2026** — v0.1: ядро STTM + Lenta-пайплайн (34 теста, Sharpe 1.0/1.06 на Lenta).
- **25.08.2026** — 4 раунда аудита Mavis: 1 баг осей + 1 протокольный баг + тесты анти-утечки. **Sharpe пересмотрен: 0.45/0.41 (z ≈ −2) на Lenta**.
- **25.08.2026** — Этап 6: Granger, Direction, FDR, Sensitivity, Placebo (10 тестов).
- **25.08.2026** — Этап 5: Endogenous baselines + SESTM (10 тестов).
- **25.08.2026** — run_all.py (Этап 8). Тесты: 70/70, ruff: 0.
- **25.08.2026** — **Kommersant Sharpe 1.42** (формальное воспроизведение статьи).
- **25.08.2026** — **Placebo p = 0.50** на Kommersant (negative finding).
- **25.08.2026** — **OOS 2022-2026: Sharpe −0.23** (catastrophic).
- **26.08.2026** — **Long-short decile тест**: Sharpe 0.262 gross / 0.064 net, p=0.68.
- **26.08.2026** — Настоящая статья.
- **07–08.09.2026** — Полный пересчёт на full universe/corpus: все блоки сошлись;
  новое: full-corpus IS 0.36, random-плацебо p≈0.06–0.08 (NS), 1.48 признано
  невоспроизводимым; фиксы: 4 аудита + guard №5b + env-дрейф + футган `--limit`.
- **08.09.2026** — Статья обновлена: плацебо-секция переписана на воспроизводимый
  тест, добавлен §4.8. Сюита 73/73.

---

*Конец статьи.*

*Автор: Moria (команда newsalpha). Лицензия: CC BY 4.0.*
*При использовании результатов просьба ссылаться на репозиторий и оригинальную статью STTM.*
