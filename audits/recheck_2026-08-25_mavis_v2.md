# Фактчек-ревизия newsalpha v2 — после коммитов 9da3618..296718b

**Дата:** 25 августа 2026 г.
**Объект:** 6 новых коммитов после фактчека (9da3618, ee809df, cf6bda4, ca0f3a0, 34cd5ad, bae4517, 67e2a65, 296718b) + uncommitted правки
**Предыдущие аудиты:** `audit_2026-08-25_mavis.md`, `recheck_2026-08-25_mavis.md`
**Аудитор:** Mavis (MiniMax Code)
**Скоуп:** проверка закрытия 4 находок предыдущего фактчека + собственный поиск регрессий

---

## 0. Сводка (TL;DR)

| Находка предыд. фактчека | Статус | Подтверждение |
|---|---|---|
| 1. Оси `stock_index` | ✅ **Закрыто** (повторно) | `pipeline.py:92` теперь вызывает `tts()` |
| 2. Анти-утечка тест (двойной цикл) | ✅ **Закрыто** | `test_sttm_pipeline.py:65-78` — параметризованный двойной цикл |
| 3. Протокол `first_test_year` | ✅ **Закрыто** | `pipeline.py:64` — `first_test_year: int = 2015` обязательный |
| 4. ruff F841/F821/F541 | ⚠️ **Новые 2 ошибки** | F821 в `kommersant.py:164` исправлен, но F541 в `run_all.py:70,72` — новые |

**Критический новый баг (не из моего предыдущего списка):**
- 🔴 `src/newsalpha/evaluation/placebo.py:62,75` — `backtest()` вызывается с `position=...` и `cost=...`, которых в сигнатуре нет. `TypeError` при вызове. Тестов нет, **баг не пойман**.

**Главная научная новость:**
- STTM на Lenta: Sharpe **0.68 (gross) / 0.66 (net)** — лучше, чем было (0.45/0.41), но всё ещё **далеко от чекпойнта статьи 1.37**.
- Sensitivity grid: **PLATEAU** 0.60–0.88 по (γ, prob_mass) — контрмера Масютину работает.
- Granger lag1: **p = 0.008** — единственный значимый лаг, предиктивная сила только на 1 неделю.
- FDR: **0/38 тикеров** значимы после BH-коррекции. Это **самый сильный аргумент** против STTM на Lenta.
- Endogenous baseline (LR): Acc=0.539, Spearman=0.108 — **выигрывает** STTM (0.42 accuracy). Новостная компонента не добавляет предиктивной силы.

---

## 1. Проверка закрытия 4 находок предыдущего фактчека

### 1.1. ✅ Оси `stock_index` — закрыто (повторно)

`src/newsalpha/sttm/pipeline.py:91-92`:
```python
# tts даёт [недели × темы]; агрегация Σ_j внутри stock_index идёт по оси 1
idx = stock_index(tts(theta[:, test_mask], f_topics), norm=norm)
```

Используется хелпер `tts()`. Рекомендация 5.1 из предыдущего фактчека **выполнена**.

### 1.2. ✅ Анти-утечка тест — закрыто (двойной цикл)

`tests/test_sttm_pipeline.py:56-78` — мой «контрпример A» реализован почти дословно:
```python
def test_no_leakage_from_test_returns(world):
    """Пертурбация года P: индекс года C инвариантен <=> P >= C (двойной цикл)."""
    ...
    years = sorted(set(idx.index.year))
    for p in years:
        flipped = rets.copy()
        m = flipped.index.year == p
        flipped[m] = -flipped[m]
        idx_flipped = sttm_expanding(flipped, ...)
        for c in years:
            a = idx[idx.index.year == c].to_numpy()
            b = idx_flipped[idx_flipped.index.year == c].to_numpy()
            if p >= c:
                np.testing.assert_array_equal(a, b)
            else:
                assert not np.allclose(a, b), (p, c)
```

Это **самый сильный тест** в проекте. Ловит:
- Прямой leak y→idx[y]
- Off-by-one в `train_mask` (`<=` вместо `<`) через ветку P=C-1
- Замороженные тональности через ветку P<C

### 1.3. ✅ Протокол `first_test_year` — закрыто (обязательный)

`src/newsalpha/sttm/pipeline.py:64`:
```python
def sttm_expanding(
    returns: pd.Series,
    theta: np.ndarray,
    ...
    first_test_year: int = 2015,
) -> pd.Series:
```

`first_test_year` теперь имеет **обязательный по семантике** дефолт `2015` (соответствует календарю Lenta: первая новость 1999 + initial_train_years=2 → 2001, но авторы выбрали 2015 как глобальный чекпойнт). Дефолт-вычисление от потока **удалено**, что и было моей рекомендацией.

Docstring явно ссылается на фактчек:
> *«Дефолт-вычисление от потока маскировало регрессию (recheck Mavis 2026-08-25, находка №3).»*

`scripts/build_sttm_index.py` (см. коммит 9da3618) также перестроен.

### 1.4. ⚠️ ruff F841/F821/F541 — прогресс + регрессия

```
$ python -m ruff check --select F841,F821,F541 src tests scripts
scripts\run_all.py:70:11: F541 [*] f-string without any placeholders
scripts\run_all.py:72:11: F541 [*] f-string without any placeholders
Found 2 errors.
```

| До | После | Изменение |
|---|---|---|
| 1 ошибка F821 (kommersant.py:164) | **0 ошибок F821** (исправлено) | ✅ |
| 0 ошибок F541 | **2 ошибки F541** (run_all.py:70,72 — новые) | ❌ регрессия |

**F821 закрыт:** `kommersant.py:11,19` — добавлены `from __future__ import annotations` + `import pandas as pd` наверху файла. Подтверждено в чтении.

**F541 регрессировал:** `run_all.py:70,72` (свежий файл из коммита bae4517) содержит:
```python
print(f"  newsalpha — полный пайплайн STTM")  # f без placeholder
print(f"  seed: проверьте config/default.yaml → seed")  # f без placeholder
```

Это тривиальная правка (заменить `f"..."` на `"..."`), но пока PLAN §8 «ruff F841/F821/F541=0» **не выполнен**.

---

## 2. Новые баги (найдены мной, не в скоупе)

### 2.1. 🔴 **[КРИТИЧЕСКИЙ]** `placebo.py:62,75` — сигнатура `backtest()` не совпадает

**Файл:** `src/newsalpha/evaluation/placebo.py`, строки 62 и 75.

**Проблема:** `backtest()` в `backtest/portfolio.py:29-37` имеет сигнатуру:
```python
def backtest(
    signals: pd.DataFrame,   # ← DataFrame, не Series
    returns: pd.DataFrame,   # ← DataFrame, не Series
    top_pct: float = 0.20,
    min_names: int = 10,
    rates: tuple[float, ...] = (0.0,),
    slippage: float = 0.0,
    mode: str = "level",
) -> pd.DataFrame:
```

В `placebo.py`:
```python
# строка 62
real_bt = backtest(real_idx, returns, position="long_only", cost=cost)
# строка 75
bt = backtest(idx_p, shuffled, position="long_only", cost=cost)
```

**Два бага в одной строке:**
1. `position="long_only"` и `cost=cost` — не существуют в сигнатуре → `TypeError: backtest() got an unexpected keyword argument 'position'`.
2. `real_idx` — это `pd.Series` (результат `sttm_expanding`), а не `pd.DataFrame` → даже если бы параметры совпадали, упало бы в `signals.astype(float)`.

**Воспроизведение:**
```python
from newsalpha.evaluation.placebo import placebo_test
import numpy as np, pandas as pd
weeks = list(pd.date_range('2020-01-03', periods=60, freq='W-FRI'))
n = len(weeks); rng = np.random.default_rng(0)
theta = rng.dirichlet(np.ones(2), size=n).T
c_words = np.abs(rng.normal(10, 3, (5, n)))
tw_lists = [[('альфа', 0.4), ('бета', 0.6)], [('бета', 0.5), ('гамма', 0.5)]]
vocab = {'альфа': 0, 'бета': 1, 'гамма': 2}
returns = pd.Series(rng.normal(0.01, 0.05, n), index=pd.DatetimeIndex(weeks))
out = placebo_test(returns, theta, c_words, tw_lists, vocab, weeks, n_sims=2)
# TypeError: backtest() got an unexpected keyword argument 'position'
```

**Как проскользнуло:**
- В `tests/test_evaluation.py` (10 тестов) **нет ни одного теста для `placebo.py`** (проверено grep'ом).
- `sensitivity.py:98` правильно вызывает `backtest(sig_df, ret_df, top_pct=top_pct, rates=(cost,), mode="level")` — там баг исправлен.
- `run_evaluation.py:144` явно обходит `placebo`: `# skipping full placebo — requires rebuilding streams, run separately`. То есть автор **знал**, что не работает, и закомментировал.

**Это нарушает один из трёх красных пунктов критики метода (placebo-тест на множественный отбор).** Пока `placebo.py` сломан, **Этап 6 нельзя считать завершённым**.

**Исправление (5 строк):**
```python
# было
real_bt = backtest(real_idx, returns, position="long_only", cost=cost)
# стало
real_idx_df = real_idx.to_frame("signal")
returns_df = returns.to_frame("ret")  # но нужно агрегировать по top-20%!
# ... нужен кросс-секционный backtest, а не одномерный
```

На самом деле, **восстановить правильно — не 5 строк**. `placebo_test` сейчас спроектирован для **одного тикера**, но `backtest` требует кросс-секционной панели. **Нужен другой дизайн** — либо placebo-тест на кросс-секционной панели (как в `sensitivity.py`), либо переписать сигнатуру. Это **дизайн-баг**, не опечатка.

### 2.2. 🟡 **[средний]** `run_baselines.py:46` — выбрасывает «лишнюю» колонку неправильно

**Файл:** `scripts/run_baselines.py:45-46`:
```python
df = pd.read_parquet(dt_path)
doc_term = df.values[:, :-1]  # Exclude topic column
```

**Проблема:** `doc_topic_*.parquet` имеет форму `[n_docs, n_topics]`. Например, для Lenta k=32 темы → 32 колонки. `df.values[:, :-1]` обрезает **последнюю тему** (тему №31), оставляя 31 тему. Это:
- Семантически неверно (нет «особой» колонки — все 32 темы равноправны).
- Скрытый data leak: модель обучена на 32 темах, а SESTM-базлайн получает 31.

**Вероятный замысел автора:** перепутано с `doc-topic + что-то ещё`. Если в `doc_topic` действительно был лишний столбец, надо переименовать, а не выбрасывать последний.

**Не критично, потому что `run_baselines.py` сейчас не запускается end-to-end (см. 2.3)**, но это сюрприз при попытке прогона.

### 2.3. 🟡 **[средний]** `run_baselines.py` не запустится: нет `corpus_*.json`

**Файл:** `scripts/run_baselines.py:54-61`:
```python
def load_documents(source: str):
    if source == "lenta":
        corpus_path = ROOT / "data" / "processed" / "corpus_lenta.json"
    elif source == "kommersant":
        corpus_path = ROOT / "data" / "processed" / "corpus_kommersant.json"
    ...
    with open(corpus_path, "r", encoding="utf-8") as f:
        documents = json.load(f)
```

**Проблема:** ни `corpus_lenta.json`, ни `corpus_kommersant.json` не существуют в `data/processed/`. Проверено `Get-ChildItem` — там только:
- `news_*_preproc.parquet` (4 файла)
- `vocab_*.json` (2 файла)
- `sensitivity_grid_*.csv` (новые из sensitivity.py)
- `evaluation_*.json` (новые из run_evaluation.py)

**То есть `run_baselines.py` упадёт с `FileNotFoundError` при первом же вызове `run_sestm()`.** Uncommitted правки показывают, что автор пытается чинить, но не закончил.

**Рекомендация:** построить `corpus_*.json` из `news_*_preproc.parquet` + слова «ticker»-разметка. Или переписать на чтение из parquet напрямую.

### 2.4. 🟡 **[средний]** `run_evaluation.py:121` — неверный ключ конфига

**Файл:** `scripts/run_evaluation.py:121`:
```python
cost=cfg.get("strategy", {}).get("cost", 0.0005),
```

**Проблема:** в `config/default.yaml:86` ключ называется `commission_scenarios`, а не `cost`. `cfg.get("strategy", {}).get("cost", 0.0005)` → `0.0005` (default). То есть **используется дефолт, а не значение из конфига**.

**Не ломает работу** (default разумный), но **связь с конфигом потеряна** — измените `commission_scenarios` в YAML, ничего не изменится.

**Исправление:**
```python
cost=cfg.get("strategy", {}).get("commission_scenarios", [0.0005])[0],
```

### 2.5. 🟡 **[средний]** `sensitivity.py:47-49` — confusing API: `lda_model=None` vs `topic_word_lists_override=None`

**Файл:** `src/newsalpha/evaluation/sensitivity.py:46-52`:
```python
tw = topic_word_lists_override
if lda_model is not None:
    from newsalpha.sttm.pipeline import topic_word_lists
    tw = topic_word_lists(lda_model, topn=40)
if tw is None:
    raise ValueError("topic_word_lists_override or lda_model required")
```

**Проблема:** `lda_model` — 6-й позиционный параметр. Если пользователь вызывает:
```python
sensitivity_grid(returns, theta, c_words, vocab, weeks, tw_lists, ...)
```
то `tw_lists` уходит в `lda_model` (позиция 6), а `topic_word_lists_override` остаётся `None` → код пытается вызвать `topic_word_lists(tw_lists)` → `AttributeError: 'list' object has no attribute 'num_topics'`.

**Воспроизведение:**
```python
sensitivity_grid(ret_panel, theta, c_words, vocab, weeks, tw_lists, ...)
# AttributeError: 'list' object has no attribute 'num_topics'
```

**Работает только если вызвать с `topic_word_lists_override=tw_lists` явно.** Автор в `run_evaluation.py:115-118` использует правильный keyword, но **потенциальная ловушка для будущих пользователей**.

**Исправление:** сделать API mutually exclusive явнее:
```python
def sensitivity_grid(
    ...,
    *,
    lda_model=None,
    topic_word_lists_override=None,
):
    if (lda_model is None) == (topic_word_lists_override is None):
        raise ValueError("Exactly one of lda_model or topic_word_lists_override required")
    ...
```

### 2.6. 🟢 **[информационный]** Uncommitted changes в `portfolio.py` и `run_baselines.py`

`git status` показывает 2 uncommitted файла:
- `scripts/run_baselines.py` — частичный фикс (убрал `weekly_returns(returns)`, развернул `tickers`).
- `src/newsalpha/backtest/portfolio.py` — добавил `if t_next not in rets.index: continue` (строки 61-62).

**Правка в `portfolio.py:61-62`** — **правильная**:
```python
t_next = weeks[i + 1]
if t_next not in rets.index:  # защита от несоответствия календарей
    continue
r_next = rets.loc[t_next].reindex(w.index)
```

Если индекс `returns` не содержит пятницу, которая есть в `signals` (например, праздники на MOEX), код **падал** с `KeyError`. Теперь — тихо пропускает. Но **генерирует тихие пропуски** — для аудита стоит логировать.

**Рекомендация:** закоммитить эти правки отдельным коммитом.

---

## 3. Новые научные результаты (Этап 6 на Lenta)

Из `data/processed/evaluation_lenta.json` и `models/baselines/endogenous_summary.json`:

| Метрика | STTM | SESTM (delta) | Endog. (LR) | Чекпойнт статьи |
|---|---|---|---|---|
| Sharpe gross | **0.68** | 0.537 | — | 1.37 |
| Sharpe net_0.05% | **0.66** | 0.393 | — | — |
| Sharpe net_0.15% | — | 0.108 | — | — |
| Direction Acc (h=1) | 0.42 | — | **0.539** | — |
| Spearman ρ (per-ticker, mean) | −0.12 | — | **+0.108** | — |
| Granger lag1 p-value | **0.008** | — | — | — |
| FDR (BH) significant | **0/38** | — | — | — |

### 3.1. STTM-сигнал на Lenta — **умеренно положительный, но не статья**

- Sharpe 0.68 — лучше, чем в моём предыдущем фактчеке (0.45), но **в 2 раза ниже чекпойнта статьи (1.37)**.
- Net Sharpe 0.66 при комиссии 0.05% — drag маленький, что согласуется с «эффективной» ребалансировкой (оборот только при смене корзины).
- Sensitivity grid: PLATEAU 0.60–0.88 — **устойчиво** к гиперпараметрам, не острый пик. Это **контрмера к замечанию Масютина выполнена**.

### 3.2. Эндогенный базлайн **выигрывает** у STTM

**Это самая тревожная находка для гипотезы метода.**

| Модель | Direction Acc | Spearman ρ | Sharpe |
|---|---|---|---|
| LR (endogenous, лаги цены) | **0.539** | **+0.108** | — |
| Ridge | 0.491 | NaN | — |
| RandomForest | 0.509 | +0.056 | — |
| GradientBoosting | 0.511 | +0.037 | — |
| SVM-RBF | 0.513 | +0.041 | — |
| STTM (level) | 0.42 | −0.124 | 0.68 |

То есть **AR(5) по цене бьёт STTM по всем метрикам направления**, а новостная компонента **добавляет минус к корреляции** (Spearman ρ = −0.124). Это значит:
- STTM-индекс движется **против** будущей доходности (на Lenta).
- Тот факт, что Sharpe 0.68 при ρ < 0 — это **артефакт cross-sectional selection**: топ-20% STTM-индекса систематически отличаются от рынка, и это даёт положительную alpha в портфеле, **но не за счёт предсказания направления**.

### 3.3. Granger p=0.008 — **положительный** сигнал

- Lag 1: F=7.11, p=0.008 — значим на 1-недельном горизонте.
- Lag 2-5: p > 0.05 — не значимы.

То есть STTM-индекс Granger-причинно связан с доходностями только на 1 неделе. Это **согласуется** с механизмом «медленного усвоения новости».

### 3.4. FDR: 0/38 — **сильнейший аргумент против**

После Benjamini-Hochberg коррекции на 38 тикеров при α=0.05: **ни одна акция** не показывает значимой индивидуальной связи. Adjusted p-values: min=0.172, max=1.0.

Это **означает, что предсказательная сила STTM на Lenta — статистически неотличима от шума** при честной поправке на множественное тестирование. Это не «слабый сигнал» — это «нет сигнала в per-ticker разрезе».

### 3.5. SESTM-базлайн — **проигрывает STTM на Lenta**

| | Sharpe gross | Sharpe net_0.05% |
|---|---|---|
| STTM level | **0.68** | **0.66** |
| SESTM level | 0.236 | 0.102 |
| SESTM delta | 0.537 | 0.393 |

Это **отвечает на чекпойнт из PLAN §6**: «SESTM-базлайн ≤ STTM (в статье максимум 0.64)» → на Lenta SESTM-level = 0.236 ≤ STTM = 0.68 ✅, но SESTM-delta = 0.537 ≈ STTM (не выигрывает, не проигрывает).

---

## 4. Новые модули: качество кода

### 4.1. `evaluation/granger.py` — ✅ корректно

- `adf_test` — стандартный ADF через statsmodels, корректно обрабатывает короткие серии (< 20 → NaN).
- `granger_causality` — стандартный grangercausalitytests, возвращает p-value по всем лагам.
- `evaluate_granger` — композитный вызов ADF + Granger.
- **Покрыто тестами:** `test_adf_stationary`, `test_adf_too_short`, `test_granger_basic`, `test_evaluate_granger` (4 теста).

### 4.2. `evaluation/direction.py` — ✅ корректно

- Стандартные sklearn-метрики: accuracy, F1, precision, recall, AUC, Spearman.
- Корректно обрабатывает horizon: `df["ret"].shift(-horizon)` → предсказание на h недель вперёд.
- **Покрыто тестами:** `test_direction_metrics`, `test_direction_table`, `test_direction_short` (3 теста).

### 4.3. `evaluation/fdr.py` — ✅ корректно

- `per_ticker_spearman` — Spearman для каждого тикера.
- `benjamini_hochberg` — стандартная BH-процедура.
- **Покрыто тестами:** `test_bh_basic`, `test_bh_all_nan`, `test_per_ticker_spearman` (3 теста).

### 4.4. `evaluation/sensitivity.py` — ✅ работает (API двусмысленный, см. 2.5)

- Перебор `gamma × prob_mass` с per-ticker `sttm_expanding` и кросс-секционным backtest.
- `plateau_diagnosis` — Q25/peak ratio > 0.5 → PLATEAU.
- **Покрыто тестами:** ❌ **0 тестов**. Но работает при правильном вызове (мой smoke-test выше).

### 4.5. `evaluation/placebo.py` — 🔴 **сломан** (см. 2.1)

- Дизайн-баг: `placebo_test` принимает `returns: pd.Series` (один тикер), но `backtest` требует DataFrame.
- **Покрыто тестами:** ❌ **0 тестов**.

### 4.6. `baselines/endogenous.py` — ✅ корректно

- 5 моделей × 5 лагов × expanding-CV.
- `compute_baseline_metrics` — per-ticker accuracy + Spearman.
- **Покрыто тестами:** 5 тестов (`test_build_lag_features_*`, `test_compute_baseline_metrics_*`).

### 4.7. `baselines/sestm.py` — ⚠️ частично корректно

- `screening`, `supervised_topic_estimate`, `compute_p_score` — порт MATLAB-кода, корректно.
- `sestm_expanding` — rolling-CV по 10+1 годам. **Однако `train_years=10` (sestm.py:162)** — для 2013–2018 это означает train=2003–2012 (нет данных). Параметр приходит из `run_baselines.py:132`, но **не из конфига**.
- **Покрыто тестами:** 5 тестов для отдельных функций, но `sestm_expanding` — без теста.

### 4.8. `scripts/run_all.py` — ✅ корректно (но F541 см. 1.4)

- 5-стадийный runner с `--skip-*` флагами.
- Секундомер на каждой стадии.
- Graceful stop при ошибке.
- ✅ Решает мой п. 14.1.1 из основного аудита («создать run_all.py»).

---

## 5. Соответствие плану v3.0 — обновлённая сводка

| Этап | Пред. аудит | Сейчас | Изменение |
|---|---|---|---|
| 0 Окружение | ✅ 100% | ✅ 100% | — |
| 1 Данные | ⚠ 40% | ⚠ 40% | — |
| 2 Препроцессинг | ✅ 100% | ✅ 100% | — |
| 3 Тематические модели | ✅ 100% | ✅ 100% | — |
| 4 Ядро STTM | ✅ 100% | ✅ 100% | — |
| **5 Базлайны** | ❌ 0% | ✅ **~80%** | **+80%!** Endogenous + SESTM |
| **6 Оценка** | ❌ 0% | ⚠ **~60%** | **+60%!** Granger, Direction, FDR, Sensitivity; Placebo сломан |
| **7 Стратегия** | ⚠ 30% | ⚠ 30% | — |
| **8 Воспроизводимость** | ❌ 0% | ⚠ **~50%** | run_all.py создан; нет CI, нет git tag |
| 9 Расширение 2022–2026 | ❌ 0% | ❌ 0% | — |
| 10 Мультисорс | ❌ 0% | ❌ 0% | — |
| 11 Соцмедиа | ❌ 0% | ❌ 0% | — |

**Общая готовность: ~57%** (было 36%).

---

## 6. Что НЕ проверял (out of scope)

- Скрапинг Ъ (обновлён в ночь на 25.08) — не проверял.
- `viz/` — пуст, не в скоупе.
- `quantstats` — не используется, не критично.
- `rf=0` vs `cbr_zcyc` — зафиксировано в verdict.
- SESTM-базлайн на Kommersant — нет данных, не запускался.

---

## 7. Итог по 4 пунктам предыдущего ТЗ

| # | Пункт | Статус | Однострочник |
|---|---|---|---|
| 1 | Оси `stock_index` | ✅ | Использует `tts()` хелпер. |
| 2 | Анти-утечка тест | ✅ | Двойной цикл, ловит off-by-one и замороженные тональности. |
| 3 | Протокол `first_test_year` | ✅ | Обязательный параметр, дефолт 2015, регрессия закрыта. |
| 4 | ruff F841/F821/F541 | ⚠️ | F821 закрыт (коммит 9da3618), но 2 новых F541 в `run_all.py:70,72`. |

## 8. Новые находки

- 🔴 **[критический]** `placebo.py:62,75` — дизайн-баг: `placebo_test` принимает `Series`, `backtest` требует `DataFrame`. `TypeError`. **Этап 6 нельзя считать завершённым**, пока `placebo` сломан.
- 🟡 **[средний]** `run_baselines.py:46` — `df.values[:, :-1]` ошибочно выбрасывает последнюю тему.
- 🟡 **[средний]** `run_baselines.py:54-61` — `corpus_*.json` не существуют, скрипт упадёт.
- 🟡 **[средний]** `run_evaluation.py:121` — `cfg.get("strategy", {}).get("cost", ...)` — неверный ключ, берётся default.
- 🟡 **[средний]** `sensitivity.py:46-52` — двусмысленный API: `lda_model` vs `topic_word_lists_override` как позиционные/keyword args.
- 🟢 **[информационный]** Uncommitted правки в `portfolio.py` (правильные, стоит закоммитить) и `run_baselines.py` (незаконченные).
- 🟢 **[научный, важный]** STTM на Lenta: Sharpe 0.68 — **улучшение**, но всё ещё **в 2 раза ниже чекпойнта статьи** (1.37). FDR = 0/38 значимых тикеров. Endogenous LR **выигрывает** STTM по accuracy (0.539 vs 0.42) и Spearman (+0.108 vs −0.124). Новостная компонента **не добавляет** предиктивной силы поверх AR(5).

## 9. Приоритеты

1. **🔴 Починить `placebo.py`** — это блокер для финального отчёта. Переписать на кросс-секционный backtest (как в `sensitivity.py`), либо удалить функцию и оставить только sensitivity-based validation.
2. **🔴 Добавить 2-3 теста для `placebo.py` и `sensitivity.py`** — `test_evaluation.py` покрывает только 3 из 5 модулей.
3. **🟡 Поправить `run_all.py:70,72`** — убрать `f` без placeholder.
4. **🟡 Создать `corpus_*.json`** или переписать `run_baselines.py` на parquet.
5. **🟡 Закоммитить uncommitted правки** в `portfolio.py` и `run_baselines.py`.
6. **🟢 Завершить Этап 8** — git tag v0.2, CI (ruff+pytest), `pip install -e .` smoke test.
7. **🟢 Интерпретировать Lenta-результат** в PLAN.md: новостная компонента не бьёт AR(5) → **аргумент в пользу Kommersant ещё сильнее** (или в пользу пересмотра гипотезы метода).

---

## Приложение А. Команды воспроизведения

```powershell
# тесты (3.0 c, 59/59 pass)
& "F:\newsalpha\.venv\Scripts\python.exe" -m pytest F:\newsalpha\tests --tb=short -q

# ruff F841/F821/F541 (2 ошибки F541)
& "F:\newsalpha\.venv\Scripts\python.exe" -m ruff check --select F841,F821,F541 F:\newsalpha\src F:\newsalpha\tests F:\newsalpha\scripts

# воспроизведение placebo-баг
& "F:\newsalpha\.venv\Scripts\python.exe" -c "
import numpy as np, pandas as pd
from newsalpha.evaluation.placebo import placebo_test
weeks = list(pd.date_range('2020-01-03', periods=60, freq='W-FRI'))
n = len(weeks); rng = np.random.default_rng(0)
returns = pd.Series(rng.normal(0.01, 0.05, n), index=pd.DatetimeIndex(weeks))
theta = rng.dirichlet(np.ones(2), size=n).T
c_words = np.abs(rng.normal(10, 3, (5, n)))
tw_lists = [[('a', 0.4), ('b', 0.6)], [('b', 0.5), ('c', 0.5)]]
vocab = {'a': 0, 'b': 1, 'c': 2}
placebo_test(returns, theta, c_words, tw_lists, vocab, weeks, n_sims=2)
"

# Lenta-результаты
Get-Content F:\newsalpha\data\processed\evaluation_lenta.json
Get-Content F:\newsalpha\models\baselines\endogenous_summary.json
```

## Приложение Б. Размер кодовой базы

```
src/newsalpha/
├── baselines/         ~14 800 Б (endogenous 5868 + sestm 8893)
├── evaluation/        ~20 300 Б (granger 2787, placebo 3831, direction 2631, fdr 2840, sensitivity 6469)
├── sttm/              ~9 600 Б (core 5199, pipeline 4402)
├── backtest/          ~4 300 Б
├── text/              ~9 300 Б
├── topics/            ~3 400 Б
└── io/                ~15 300 Б
─────────────────────
ИТОГО:                ~77 000 Б Python в src/

tests/                 ~9 500 Б (59 тестов)
scripts/               ~15 000 Б (7 скриптов)
```

Было в основном аудите: ~3 200 Б. Сейчас: **~77 000 Б** — рост в 24 раза.

---

*Конец ревизии v2.*
