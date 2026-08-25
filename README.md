# newsalpha

Открытая реализация метода **STTM** (Stock Tonal Topic Modeling, PeerJ CS 2022,
DOI 10.7717/peerj-cs.1156) и его первоисточника **SESTM** (Ke/Kelly/Xiu, JASA 2026):
тематическое моделирование новостного потока → тональность тем → недельный сигнал →
long-only портфель топ-20% акций MOEX.

Отличие от других реализаций: издержки и placebo-тесты с первого прогона,
отчёт по всем моделям без отбора лучших, протокол 10 сидов.

## Быстрый старт

```bash
# 1. Окружение
python -m venv .venv
.venv\Scripts\activate            # Windows (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"

# 2. Данные: положить корпуса в data/raw/news/ (см. PLAN.md, Этап 1)
#    Цены MOEX TQBR уже в data/raw/prices/

# 3. Пайплайн по стадиям (каждая читает config/default.yaml)
python scripts/fetch_news.py      # скачивание готовых корпусов
python scripts/preprocess.py
python scripts/train_topics.py
python scripts/build_index.py     # ядро STTM
python scripts/backtest.py

# Или всё сразу:
python scripts/run_all.py
```

## Структура

```
config/default.yaml   ← все гиперпараметры
data/raw|interim|processed
src/newsalpha/        ← пакет: io text topics sttm baselines evaluation strategy viz
scripts/              ← argparse-CLI по стадиям + run_all.py
notebooks/ tests/ reports/ models/
```

Подробности — в [PLAN.md](PLAN.md) (v3.0): принципы архитектуры, этапы 0–8,
протокол экспериментов, чекпойнты воспроизведения.

## Ключевые источники

1. STTM (PeerJ CS 2022): https://peerj.com/articles/cs-1156/
2. SESTM (JASA 2026, Ke/Kelly/Xiu): DOI 10.1080/01621459.2026.2643001 · код: DOI 10.6084/m9.figshare.31825294.v2
3. Диссертация Рябых (ВШЭ 2025): https://www.hse.ru/sci/diss/1087526075
4. Патент ВТБ EA044248B1: https://patents.google.com/patent/EA044248B1/ru

## Лицензия

Код — CC BY 4.0. При использовании указывайте авторство оригинального метода:
Riabykh, Surzhko, Konovalikhin, Koltsov (PeerJ CS 2022).
