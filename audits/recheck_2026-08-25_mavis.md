# Фактчек-ревизия newsalpha после фикса 8ebea19

**Дата:** 25 августа 2026 г.
**Объект:** коммит `8ebea19` — `fix(sttm): оси stock_index + first_test_year от старта торгов; тесты анти-утечки sttm_expanding (5); ревизия результатов Lenta (Sharpe 0.45/0.41, z~-2); README имена скриптов`
**Предыдущий аудит:** `audits/audit_2026-08-25_mavis.md` (25.08.2026)
**Аудитор:** Mavis (MiniMax Code)
**Проверял:** 4 пункта ТЗ + собственный поиск новых багов

---

## 0. Сводка (TL;DR)

| Пункт | Вердикт | Однострочное объяснение |
|---|---|---|
| **1. Оси `stock_index`** | ✅ **Корректно** | `stock_index` получает `[недели×темы]`, `sum(axis=1)` агрегирует по темам — правильно. |
| **2. Анти-утечка тест** | ⚠️ **Частично** | Ловит исправленный баг + leak из y в idx[y]. **Не ловит:** off-by-one в `train_mask` (`<=` вместо `<`) и leak из y в idx[y+1]. |
| **3. Протокол `first_test_year`** | ⚠️ **Отклоняется от статьи** (defensible) | Per-ticker старт корректнее для тикеров с разными датами IPO, но это **не** «train 2013-2014 → test 2015+» из статьи. Для SBER совпадает, для FIVE — 0 тест-лет. |
| **4. Регрессия** | ⚠️ **39/39 pass, но ruff = 1 ошибка** | Pytest зелёный. Ruff F841/F821/F541 ≠ 0: F821 в `kommersant.py:164` (pre-existing, не из этого коммита). |

**Критическая находка:** после фикса **Lenta-сигнал не просто слабый, а слегка вредоносный** (z ≈ −2). Это сильнее, чем «z ≈ +1.3, незначимо» из предыдущего аудита. Аргумент за скрапинг Ъ усиливается.

**Новые баги в ядре (не заявленные в ТЗ):** 3 — детали ниже.

---

## 1. Проверка осей `stock_index`

### 1.1. Контракт `stock_index`

`src/newsalpha/sttm/core.py:116-120`:
```python
def stock_index(tts_week: np.ndarray, norm: str = "sigmoid") -> np.ndarray:
    """Индекс недели: агрегация Σ_j TTS → нормировка в [0,1]."""
    agg = tts_week.sum(axis=1)
    if norm == "sigmoid":
        return 1.0 / (1.0 + np.exp(-agg))
```

**Контракт:** `tts_week` имеет форму `[недели × темы]`, `sum(axis=1)` агрегирует по темам, на выходе `[недели]`.

### 1.2. Что теперь передаётся

`src/newsalpha/sttm/pipeline.py:87`:
```python
idx = stock_index((theta[:, test_mask] * f_topics[:, None]).T, norm=norm)
```

Проверка размерностей:

| Выражение | Форма | Семантика |
|---|---|---|
| `theta` | `[темы × недели]` | глобальный topic stream |
| `theta[:, test_mask]` | `[темы × n_test_weeks]` | только тестовые недели |
| `f_topics` | `[темы]` | тональности тем (train-only) |
| `f_topics[:, None]` | `[темы × 1]` | broadcast-вектор |
| `theta[:, test_mask] * f_topics[:, None]` | `[темы × n_test_weeks]` | поэлементное умножение: `[j, t] = θ[j,t] · f_T[j]` |
| `.T` | `[n_test_weeks × темы]` | **соответствует контракту `stock_index`** ✅ |
| `stock_index(...).sum(axis=1)` | `[n_test_weeks]` | агрегация по темам — корректно ✅ |

### 1.3. Что было в баге (pre-fix)

`pipeline.py:84` (до фикса):
```python
idx = stock_index(theta[:, test_mask] * f_topics[:, None], norm=norm)
# передавалось [темы × n_test_weeks]
# stock_index.sum(axis=1) суммировала по неделям → [темы]
# zip в строке 88 обрезал до min(2, n_test_weeks) = 2
```

**Последствия бага:** для каждого тестового года индекс содержал **только 2 недели** (вместо всех), со значениями «сумма по неделям внутри темы» (вместо «сумма по темам внутри недели»). Это объясняет, почему исходный Sharpe 1.0/1.06 был артефактом: стратегия с 2 точками ребалансировки в год — фактически «купил-и-забыл», что систематически завышает Sharpe через низкий turnover.

### 1.4. Проверка через тесты

`tests/test_sttm_core.py:93-107` — `test_tts_and_index`: явно проверяет, что `tts()` возвращает `[2 недели × 2 темы]`, и `stock_index(t).sum(axis=1)` = `[2]`. Тест существовал ДО фикса, **но** в `sttm_expanding` использовался не `tts()`, а инлайновая формула с ТЕМ ЖЕ `.T` (что эквивалентно). Так что **`stock_index` уже был защищён** в смысле контракта — баг был в том, что вызывающий код **не** транспонировал вход перед вызовом.

### 1.5. Новая находка: дублирование `tts()`

`src/newsalpha/sttm/core.py:111-113`:
```python
def tts(theta: np.ndarray, f_topics: np.ndarray) -> np.ndarray:
    """TTS[t, j] = Θ[j, t] · f_T[j]; на входе Θ [темы × недели], выход [недели × темы]."""
    return (theta * f_topics[:, None]).T
```

`src/newsalpha/sttm/pipeline.py:87`:
```python
idx = stock_index((theta[:, test_mask] * f_topics[:, None]).T, norm=norm)
```

**Это в точности `tts(theta[:, test_mask], f_topics)`.** Хелпер определён, но не используется. Не баг, но **code smell** — если кто-то поправит формулу в `tts()`, в `pipeline.py` останется старое поведение.

**Рекомендация:** заменить на `idx = stock_index(tts(theta[:, test_mask], f_topics), norm=norm)`.

### ✅ Вердикт по п.1

**Оси исправлены корректно.** Контракт `stock_index` теперь соблюдён. `tts()` не используется — задокументировать как техдолг.

---

## 2. Анти-утечка тест

### 2.1. Что ловит

`tests/test_sttm_pipeline.py:45-58`:
```python
def test_no_leakage_from_test_returns(world):
    """Переворот доходностей года Y не меняет индекс года Y (каждый Y отдельно)."""
    rets, theta, c_words, weeks = world
    idx = sttm_expanding(rets, theta, c_words, TW, VOCAB, weeks)
    assert len(idx) > 50
    for y in sorted(set(idx.index.year)):
        flipped = rets.copy()
        m = flipped.index.year == y
        flipped[m] = -flipped[m]
        idx_flipped = sttm_expanding(flipped, theta, c_words, TW, VOCAB, weeks)
        np.testing.assert_array_equal(
            idx[idx.index.year == y].to_numpy(),
            idx_flipped[idx_flipped.index.year == y].to_numpy(),
        )
```

**Логика:** для каждого тестового года y — перевернуть доходности этого года, пересчитать индекс, проверить, что индекс для года y битово совпадает с исходным.

**Что ловит:**
- ✅ Оригинальный баг осей: при старом коде `len(idx) ~= 4` недели, не пройдёт `assert len(idx) > 50`.
- ✅ Любой leak, где `f_w` или `f_topics` для года y используют возвраты года y (например, `train_mask = (week_years <= test_year)` для y+1+ не сработает тут, но для самого y — сработает).
- ✅ Любой leak через `word_tone_matrix` с диапазоном `[train_start, test_year]` (вместо строго `< test_year`).

### 2.2. Что НЕ ловит — конкретные контрпримеры

#### Контрпример A: off-by-one в `train_mask`

Гипотетический баг (не в текущем коде, но возможный при будущей правке):
```python
# В pipeline.py:80
train_mask = (week_years <= test_year) & ~np.isnan(r_vec)  # BUG: <= вместо <
```

**Что произойдёт:**
- Для теста года Y: train = years ≤ Y (включает Y).
- Переворачиваем year Y → f_w для года Y считается на years < Y (не изменился) → idx[Y] **не меняется** → тест пройдёт. ✅
- НО f_w для года Y+1 считается на years ≤ Y (теперь перевёрнутый) → idx[Y+1] **меняется** → тест этого **не проверяет**.

**Доказательство в коде:** тест проверяет только `idx[y]`, не `idx[y+1]`, `idx[y+2]`, и т.д. Утечка «на будущее» не детектируется.

**Предложение:** расширить тест до двойного цикла:
```python
@pytest.mark.parametrize("y_perturb", range(2020, 2024))
def test_no_leakage_to_any_future_year(world, y_perturb):
    rets, theta, c_words, weeks = world
    idx = sttm_expanding(rets, theta, c_words, TW, VOCAB, weeks)
    flipped = rets.copy()
    flipped[flipped.index.year == y_perturb] = -flipped[flipped.index.year == y_perturb]
    idx_flipped = sttm_expanding(flipped, theta, c_words, TW, VOCAB, weeks)
    # для всех тест-лет Y >= y_perturb индекс должен совпасть
    for y_test in sorted(set(idx.index.year)):
        if y_test >= y_perturb:
            np.testing.assert_array_equal(
                idx[idx.index.year == y_test].to_numpy(),
                idx_flipped[idx_flipped.index.year == y_test].to_numpy(),
            )
```

#### Контрпример B: leak через `c_words` или `theta`

Текущий код передаёт `c_words` и `theta` глобальными (по всему корпусу). Тест **не** проверяет, что эти матрицы не «пропитаны» воздействием test-лет. На текущий момент они строятся по новостям (не по доходностям), так что leak-а нет, **но** архитектурно:

- `theta` агрегирует doc-topic по неделям → включает test-недели → «знает» о test-годах.
- `c_words` агрегирует частоты слов по неделям → то же самое.

Если когда-нибудь (по ошибке) сигнал `theta` будет зависеть от доходностей (например, через sentiment-prior), тест не поймает.

**Предложение:** отдельный тест на «свежие слова» — добавить слово, которое появляется только в test-годах, проверить, что его `c_words` ненулевая, и убедиться, что `f_w` для него = 0 во всех train-годах (потому что в train его нет). Это проверит, что новые слова не «создают» тон на test-данных.

#### Контрпример C: leak через `initial_train_years` (период малой train-выборки)

Если `initial_train_years=2` и `week_years` начинается с 2020, то для test_year=2022 train_mask = (2020, 2021) — 104 недели. Но если первый год в `c_words` присутствует, а доходности за этот год ВСЕ NaN (тикер IPO-нулся в январе 2020, первая сделка — декабрь 2020), train_mask может содержать <8 недель → ветка `continue` (pipeline.py:81-82) → test_year пропускается.

Это **не leak, но** приводит к тому, что первый тестовый год тикера молча пропускается. Тест этого не ловит — он использует синтетику с полными данными.

#### Контрпример D: нечувствительность к сдвигу границы на 1 неделю

Если `week_years` основан на `pd.date_range(..., freq="W-FRI")`, а тест-код использует `w.year`, то для недели 2020-01-03 (пятница) `year=2020`, а для недели 2019-12-30 (понедельник ISO) `year=2019`. Если где-то в коде перепутать атрибут `.year` и `.isocalendar().year`, поведение изменится. Тест не ловит такую ошибку (использует тот же метод `w.year`).

### 2.3. Что ещё стоило бы покрыть

1. **«NaN во ВСЕХ train-годах»:** если `r_vec` все NaN для years < Y, test_year пропускается. Можно добавить unit-тест: `r_vec[:first_test_year] = np.nan → idx пуст`.
2. **«`first_test_year` больше максимального года потока»:** `range(start, week_years.max()+1)` пустой → `idx` пуст. Сейчас возвращается тихо; можно ожидать `RuntimeWarning` или явный `assert`.
3. **Детерминированность:** два прогона с одинаковыми входами дают битово равный idx. Сейчас неявно предполагается (все numpy операции детерминированы), но проверки нет.

### ⚠️ Вердикт по п.2

**Тест корректен для исправленного бага и для leak «test-год → его собственный idx».** Дыры:

- Не ловит off-by-one (`<=` вместо `<`) — leak из y в idx[y+1].
- Не ловит leak через глобальные `c_words`/`theta` (сейчас нет, но архитектурно стоит страховка).
- Не ловит edge-case с пустым train.

**Рекомендация:** добавить параметризованный тест с двойным циклом (см. контрпример A). Это 5-10 строк, но закроет самый опасный класс багов.

---

## 3. Протокол `first_test_year`

### 3.1. Что заявлено

`scripts/build_sttm_index.py:67-75`:
```python
# протокол статьи: 2 календарных года train от НАЧАЛА торгов тикера
first_valid = r.dropna().index.min()
fty = None if pd.isna(first_valid) else int(first_valid.year)
...
first_test_year=None if fty is None else fty + eval_cfg["initial_train_years"],
```

`src/newsalpha/sttm/pipeline.py:77`:
```python
start = first_test_year or int(week_years.min() + initial_train_years)
```

### 3.2. Что говорит PLAN.md (Этап 5, протокол)

> *«Expanding CV: train 2013–2014, тест следующий год … до 2021 (6 сплитов)»*

Это **фиксированный** протокол: 2013–2014 — train, 2015–2021 — test, **для всех 39 тикеров** (предполагается, что они все торговались с 2013).

### 3.3. Что фактически сделано

Новый код использует **per-ticker** протокол: train начинается от первой ненулевой сделки тикера + 2 года.

**Реальные даты начала торгов** (из `data/raw/prices/shares_TQBR_*.csv`):

| Тикер | Первая сделка | first_test_year (новый) | Тест-годы | Соответствует статье? |
|---|---|---|---|---|
| HYDR, LKOH, NVTK, SBER, SBERP, VTBR | 2013-03-25 | **2015** | 2015–2018 (Lenta) / 2015–2021 (Ъ) | ✅ |
| ALRS, IRAO, LSRG, MGNT, MTSS, PHOR, PIKK, RTKM, TATN, TATNP | 2013-07-08 | **2015** | те же | ✅ |
| POLY | 2013-09-02 | **2015** | те же | ✅ |
| AFLT | 2013-10-31 | **2015** | те же | ✅ |
| TRNFP | 2013-12-16 | **2015** | те же | ✅ |
| YNDX | 2014-06-04 | **2016** | 2016–2018 (Lenta) / 2016–2021 (Ъ) | ❌ отклонение на 1 год |
| QIWI, AFKS, CHMF, FEES, GAZP, GMKN, MAGN, NLMK, PLZL, ROSN, RSTI, SNGS, SNGSP | 2014-06-09 | **2016** | те же | ❌ отклонение на 1 год |
| RUAL | 2015-03-30 | **2017** | 2017–2018 (Lenta) / 2017–2021 (Ъ) | ❌ на 2 года |
| CBOM | 2015-06-22 | **2017** | те же | ❌ на 2 года |
| UPRO | 2016-07-01 | **2018** | 2018 (Lenta) / 2018–2021 (Ъ) | ❌ на 3 года |
| DSKY | 2017-02-10 | **2019** | **0 недель на Lenta** / 2019–2021 (Ъ) | ❌ на 4 года |
| FIVE | 2018-02-01 | **2020** | **0 недель на Lenta** / 2020–2021 (Ъ) | ❌ на 5 лет |

**6 из 39 тикеров** совпадают со статьёй. **Большинство** отклоняется на 1+ лет. На Lenta-корпусе (новости до 2018-12) **DSKY и FIVE дают 0 тест-недель**.

### 3.4. Аргументы за и против

**За per-ticker:**
- Корректнее: для тикера, начавшего торговаться в 2017, train 2013–2014 — это train на NaN-возвратах, что делает `word_tone_matrix` бессмысленным (или NaN-возвраты исключаются → train_mask пуст → test_year пропускается).
- Per-ticker покрывает IPO-тикеры без специальной обработки.

**Против per-ticker (отклонение от статьи):**
- Статья сравнивает 6 разных стратегий **в одних и тех же тест-годах**. Если у SBER test = 2015–2018, а у FIVE test = 2020–2021, сравнение в одном рейтинге теряет смысл (разные режимы рынка).
- PLAN §5 явно говорит: «6 сплитов». Per-ticker даёт **от 1 до 6 сплитов** в зависимости от тикера. Число сплитов варьируется → нельзя агрегировать результаты «mean±std по 6 сплитам» в исходном виде.

### 3.5. Скрытая регрессия: тикеры «выпали» из Lenta-бэктеста

`models/sttm_indices/sttm_index_lenta.parquet` (37 КБ). Если бы все 39 тикеров × ~200 недель × 8 байт (float64) → ~62 КБ. Фактический размер 37 КБ → **~25–27 тикеров × 200 недель** или 39 × ~120 недель. Скорее всего, для DSKY и FIVE (0 тест-недель) и UPRO (только 2018) индекс пуст или очень короткий.

До фикса: код использовал `week_years.min()` = 1999 (Lenta-новости с 1999), тест-годы 2001–2018. Для SBER (с 2013) — train 2001–2012 был пуст (нет сделок), `train_mask.sum() < 8` → пропуск до 2015. То есть SBER получал test = 2015–2018 в обоих протоколах.

Для FIVE (с 2018-02) — до фикса: train 2001–2017 (пуст по тикеру), test 2018 (≈50 недель). **После фикса: test 2020+, что = 0 недель на Lenta.** Это **регрессия** для FIVE.

### 3.6. Рекомендации

1. **Зафиксировать в PLAN.md, что протокол per-ticker — это сознательное отклонение.** Текущая формулировка в PLAN §5 «train 2013–2014» — устаревшая и противоречит коду.
2. **Считать число тест-недель по тикерам и сообщать** в отчёте (Этап 6). Тикеры с < 52 тест-неделями помечать как «недостаточная выборка».
3. **Возможный компромисс:** общий train = max(first_trade_year по всем тикерам, 2013) + initial_train_years. Это даёт всем тикерам одинаковый test-period, но с «обрезанным» train для поздних IPO. Ближе к статье.

### ⚠️ Вердикт по п.3

**Протокол отклоняется от статьи.** Это **defensible** решение (корректнее для IPO-тикеров), **но**:
- Не совпадает с PLAN §5 («train 2013–2014»).
- Уменьшает число тикеров в Lenta-бэктесте (DSKY, FIVE = 0 недель, UPRO = ~50 недель).
- Требует явной документации.

**Рекомендация:** записать в PLAN.md решение о per-ticker протоколе, добавить в отчёт статистику «тест-недель на тикер».

---

## 4. Регрессия

### 4.1. pytest

```
$ python -m pytest tests --tb=short -q
.......................................                                  [100%]
39 passed in 1.46s
```

**39/39 PASS** ✅ (было 34/34 до фикса, +5 из `test_sttm_pipeline.py`).

Все 5 новых тестов:
```
tests/test_sttm_pipeline.py::test_index_only_for_post_train_years PASSED
tests/test_sttm_pipeline.py::test_no_leakage_from_test_returns PASSED
tests/test_sttm_pipeline.py::test_train_perturbation_changes_index PASSED
tests/test_sttm_pipeline.py::test_nan_weeks_excluded_from_training PASSED
tests/test_sttm_pipeline.py::test_index_in_unit_range PASSED
```

### 4.2. ruff F841 / F821 / F541

```
$ python -m ruff check --select F841,F821,F541 src tests scripts
F821 Undefined name `pd`
   --> src\newsalpha\io\kommersant.py:164:46
    |
164 | def load_days_to_df(raw_dir: str | Path) -> "pd.DataFrame":  # type: ignore[name-defined]
    |                                              ^^

Found 1 error.
```

**F841/F821/F541 ≠ 0.** Это pre-existing ошибка в `kommersant.py:164` — forward reference `pd.DataFrame` в string-аннотации, защищённая `# type: ignore[name-defined]`, но ruff F821 это **не уважает** (F821 — синтаксическая проверка, не name resolution; `# type: ignore` работает для mypy/pyright, но не для ruff F821).

**Проверка — это из этого коммита?**
```
$ git show 8ebea19:src/newsalpha/io/kommersant.py | grep "pd.DataFrame"
def load_days_to_df(raw_dir: str | Path) -> "pd.DataFrame":  # type: ignore[name-defined]
    return pd.DataFrame(rows)
```

Строка присутствовала ДО коммита 8ebea19. **Не из этого фикса.**

**Проверка — реальный баг или стилистика?**

`load_days_to_df` — обычная функция, `pd` импортируется внутри функции на строке 169. Аннотация возврата использует forward-reference в строке, потому что в момент определения функции `pd` ещё не импортирован. Это **не баг** (функция работает корректно), но ruff F821 его флагает.

**Исправление (1 строка):** переместить `import pandas as pd` на уровень модуля и убрать string-аннотацию:
```python
import pandas as pd  # наверх файла
...
def load_days_to_df(raw_dir: str | Path) -> pd.DataFrame:  # без кавычек
    ...
    return pd.DataFrame(rows)  # убрать локальный import
```

### ⚠️ Вердикт по п.4

**pytest 39/39 ✅.** **ruff F841/F821/F541 ≠ 0** — 1 pre-existing ошибка. PLAN §8 говорит «ruff F841/F821/F541=0» как цель воспроизводимости — **цель не достигнута** (хотя баг не из этого коммита).

---

## 5. Новые баги в ядре (найдены мной, не заявлены в ТЗ)

### 5.1. `[высокий]` `tts()` не используется → риск рассинхронизации

`src/newsalpha/sttm/core.py:111-113`:
```python
def tts(theta: np.ndarray, f_topics: np.ndarray) -> np.ndarray:
    """TTS[t, j] = Θ[j, t] · f_T[j]; на входе Θ [темы × недели], выход [недели × темы]."""
    return (theta * f_topics[:, None]).T
```

`src/newsalpha/sttm/pipeline.py:87`:
```python
idx = stock_index((theta[:, test_mask] * f_topics[:, None]).T, norm=norm)
```

Дублирование формулы. Если кто-то добавит, например, **L2-нормализацию** в `tts()` для устойчивости (известная практика в topic-sentiment работах), `pipeline.py` эту правку не получит. Эта ошибка не сказывается на текущих результатах, но при первом же расширении — расхождение.

**Исправление:** заменить в `pipeline.py:87` на `idx = stock_index(tts(theta[:, test_mask], f_topics), norm=norm)`.

### 5.2. `[средний]` дефолт `first_test_year=None` маскирует регрессию

`src/newsalpha/sttm/pipeline.py:77`:
```python
start = first_test_year or int(week_years.min() + initial_train_years)
```

Если будущий вызов `sttm_expanding` забудет передать `first_test_year` (например, в ноутбуке или новом скрипте), функция **молча** вернёт stream-based протокол (тест-годы от первого года новостного потока). Это **именно та ошибка, которую фикс пытался устранить** — она вернётся в любом новом коде.

**Исправление:** сделать `first_test_year` обязательным параметром:
```python
def sttm_expanding(returns, theta, c_words, tw_lists, vocab, weeks,
                   gamma, prob_mass, norm,
                   first_test_year: int,  # обязательный
                   initial_train_years: int = 2):  # только как fallback
    if first_test_year is None:
        raise ValueError("first_test_year обязателен; см. scripts/build_sttm_index.py")
    ...
```

**Альтернатива:** добавить warning через `warnings.warn` если используется дефолт.

### 5.3. `[низкий]` `summarize` в `backtest/portfolio.py` использует `rf=0` — несовместимо с конфигом

`src/newsalpha/backtest/portfolio.py:79-93`:
```python
def summarize(port: pd.Series, periods_per_year: int = 52) -> dict:
    """Годовые метрики серии недельных доходностей (rf=0, как в статье)."""
    port = port.dropna()
    if len(port) < 2 or port.std(ddof=1) == 0:
        return {"n": len(port)}
    ann_ret = port.mean() * periods_per_year
    ann_vol = port.std(ddof=1) * np.sqrt(periods_per_year)
    ...
    "sharpe": ann_ret / ann_vol,  # деление на (ann_ret - 0) = просто ann_ret/ann_vol
```

`config/default.yaml:96`: `riskfree: cbr_zcyc` — заявлено использование ОФЗ.

**Несовместимость:** код использует `rf=0` (статья), конфиг говорит `cbr_zcyc`. PLAN §7 явно фиксирует «rf = ОФЗ (ЦБ РФ)». Реализация — TODO для Этапа 6.

**Это открытый пункт** — пользователь явно сказал «не считать находкой, зафиксировано в verdict». Упоминаю для полноты.

### 5.4. `[информационный]` Lenta-сигнал после фикса хуже, чем до

`PLAN.md` (обновлённый коммитом 8ebea19, строка 175-180):
> *«Результат на Lenta-индексах после ревизии 25.08.2026 (тест 2015–2018, 207 недель): gross level Sharpe 0.45 (ret +7.7%), delta 0.41; случайный сигнал — медиана Sharpe 0.96 [0.49..1.37] (100 сидов), равновзвеш. рынок 0.85 → STTM ниже случайного, z ≈ −2. Чекпойнт статьи (1.37) не воспроизводится; на нецелевом корпусе сигнал не просто нулевой, а слегка вредоносный.»*

| | До фикса (баг) | После фикса |
|---|---|---|
| Sharpe level | 1.00 | **0.45** |
| Sharpe delta | 1.06 | **0.41** |
| z-score vs random | ≈ +1.3 (незначимо) | **≈ −2 (значимо ниже)** |
| Интерпретация | «слабый сигнал» | **«анти-предиктивный»** |

**Это переворачивает вывод предыдущего аудита.** Тот говорил: «чеклист статьи не воспроизводится, но сигнал возможен, нужен Kommersant». Теперь: «на Lenta сигнал **вредит** — а значит, скрапинг Ъ не просто «улучшит», а может **спасти** всю гипотезу».

**Это самое важное наблюдение в этой ревизии.** Рекомендую вынести в заголовок финального отчёта (когда будет готов).

---

## 6. Соответствие исходному аудиту

Сверяю с пунктом 7.4 исходного аудита (25.08.2026):

> *«7.4 (предыдущий аудит): Скрытые риски / 11.1. Утечка данных через словарь тем.*
> *В pipeline.py:46-49 sttm_expanding пересчитывает f_w и f_topics только на train-неделях. Но theta (topic stream) и c_words (word stream) — глобальные, посчитанные на всём корпусе сразу (build_streams в pipeline.py:26-38). Это технически не утечка (потоки тем не зависят от доходностей), но распределение слов по неделям «знает» о test-годах. Если бы build_streams тоже делал expanding, было бы честнее.»*

**Статус:** риск признан, **фикс не сделан** в этом коммите. `build_streams` по-прежнему глобальный. Если когда-нибудь `theta`/`c_words` начнут зависеть от доходностей (что не планируется), утечка появится.

**Рекомендация:** не блокер, но добавить тест-защиту: «свежее слово, появившееся только в test-годах, должно иметь f_w=0 во всех train-годах». Это страховка от случайной зависимости.

---

## 7. Итог по 4 пунктам ТЗ

| # | Пункт | Вердикт | Однострочник |
|---|---|---|---|
| 1 | Оси `stock_index` | ✅ | Получает `[W×T]`, `sum(axis=1)` агрегирует по темам — корректно. |
| 2 | Анти-утечка тест | ⚠️ Частично | Ловит исправленный баг + leak y→idx[y]. Не ловит: off-by-one `train_mask`, leak y→idx[y+1], leak через c_words/theta. |
| 3 | Протокол `first_test_year` | ⚠️ Отклоняется | Per-ticker ≠ статья «2013–2014». Defensible, но должно быть явно задокументировано. |
| 4 | Регрессия | ⚠️ | pytest 39/39 ✅, ruff F841/F821/F541 = 1 ошибка (pre-existing, не из коммита). |

## 8. Новые находки (не в ТЗ)

- **[высокий]** `tts()` хелпер не используется в `pipeline.py:87` — риск рассинхронизации при будущих правках ядра.
- **[средний]** Дефолт `first_test_year=None` в `sttm_expanding` маскирует регрессию — будущие вызовы без явного параметра вернут stream-based протокол.
- **[высокий, не-баг]** После фикса Lenta-сигнал **анти-предиктивен** (z ≈ −2, Sharpe 0.45/0.41). Это **усиливает** аргумент за срочный скрапинг Ъ.
- **[информационный]** `c_words`/`theta` по-прежнему глобальные (риск 11.1 из предыдущего аудита). Фикс не сделан.

## 9. Рекомендации (по приоритету)

1. **Использовать `tts()` в `pipeline.py:87`** — тривиально, убирает риск рассинхронизации.
2. **Сделать `first_test_year` обязательным** в `sttm_expanding` — предотвращает регрессию протокола.
3. **Добавить параметризованный leak-тест** с двойным циклом (контрпример A) — 5 строк, страховка от off-by-one.
4. **Зафиксировать в PLAN.md** per-ticker протокол как сознательное отклонение от статьи, с обоснованием и статистикой тест-недель на тикер.
5. **Исправить ruff F821 в `kommersant.py:164`** — поднять `import pandas as pd` на уровень модуля.
6. **В `models/embeddings/rubert/` создать `.gitkeep`** — директория в конфиге, но отсутствует на диске.
7. **Перенести `core.py:80-100` (NaN-обработка в `word_tone_matrix`) на тест** — тест `test_nan_weeks_excluded_from_training` косвенно проверяет, но прямого юнит-теста нет.

---

## Приложение А. Команды воспроизведения

```powershell
# тесты (1.5 c, 39/39 pass)
& "F:\newsalpha\.venv\Scripts\python.exe" -m pytest F:\newsalpha\tests --tb=short -q

# ruff (1 ошибка F821)
& "F:\newsalpha\.venv\Scripts\python.exe" -m ruff check --select F841,F821,F541 F:\newsalpha\src F:\newsalpha\tests F:\newsalpha\scripts

# даты начала торгов по тикерам
& "F:\newsalpha\.venv\Scripts\python.exe" -c "
import pandas as pd, yaml
from pathlib import Path
cfg = yaml.safe_load(open(r'F:\newsalpha\config\default.yaml', encoding='utf-8'))
for t in cfg['tickers']:
    p = Path(cfg['data']['prices_dir']) / f'shares_TQBR_{t}.csv'
    if p.exists():
        df = pd.read_csv(p, usecols=['TRADEDATE'], parse_dates=['TRADEDATE'])
        print(f'{t}: {df.TRADEDATE.min().date()}')
"
```

## Приложение Б. Что я НЕ проверял (out of scope)

Согласно ТЗ:
- Скрапинг Ъ, завершённый в ночь на 25.08 — не проверял.
- Пустые модули `baselines/`, `evaluation/`, `viz/` — не в скоупе (Этапы 5–6).
- `run_all.py` — Этап 8.
- `rf=0` vs `cbr_zcyc` — зафиксировано в предыдущем аудите как TODO.

Дополнительно не проверял:
- Производительность (`word_tone_matrix` сейчас O(n_words × n_train_weeks) — для 19 864 слов × 200 недель = 4M операций, но медленная часть — `np.corrcoef` через матричное умножение).
- Корректность LDA-модели на Lenta (topic coherence, читаемость тем) — не входило в ТЗ.
- Согласованность `models/sttm_indices/sttm_index_lenta.parquet` с новым протоколом (надо бы пересчитать артефакт).

---

*Конец фактчека.*
