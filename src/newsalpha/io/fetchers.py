"""Загрузка готовых новостных корпусов и конвертация в единый формат.

Схема interim-данных: date, source, title, text, url, rubric (parquet-парты).
"""
from datetime import datetime
from pathlib import Path
import re

import pandas as pd
import requests
from tqdm import tqdm

SCHEMA_COLS = ["date", "source", "title", "text", "url", "rubric"]

# Дата из URL новости Lenta: .../news/2018/12/14/cancer/
LENTA_URL_DATE_RE = re.compile(r"/news/(\d{4})/(\d{2})/(\d{2})/")

# NB: корпус ODS proj_news_viz (interfax/gazeta/iz/meduza/ria/tass) потерян —
# репозиторий ods-ai-ml4sg удалён с GitHub, в Wayback Machine только редиректы.
# Interfax заменяется скрапингом Kommersant (Этап 1b, основной источник статьи STTM).
DOWNLOAD_URLS = {
    "lenta": (
        "https://github.com/yutkin/Lenta.Ru-News-Dataset/releases/download/v1.0/lenta-ru-news.csv.gz"
    ),
    "ria": "https://github.com/RossiyaSegodnya/ria_news_dataset/raw/master/ria.json.gz",
}


def download_archive(name: str, dest_dir: str | Path) -> Path:
    """Стриминговое скачивание архива источника с прогрессом."""
    url = DOWNLOAD_URLS[name]
    dest = Path(dest_dir) / url.rsplit("/", 1)[-1]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[skip] {dest.name} уже скачан")
        return dest
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=name
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    return dest


def _get(rec, *names):
    for n in names:
        v = getattr(rec, n, None)
        if v:
            return v
    return None


def _norm_date(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.date().isoformat()
    if hasattr(v, "isoformat"):
        return v.isoformat()[:10]
    return str(v)[:10]


def _iter_rows(name: str, path: str | Path):
    """Единый поток словарей из архива источника."""
    from corus import load_lenta, load_ods_interfax

    if name == "interfax":
        for r in load_ods_interfax(str(path)):
            yield {
                "date": _norm_date(_get(r, "timestamp")),
                "title": _get(r, "title"),
                "text": _get(r, "text"),
                "url": _get(r, "url"),
                "rubric": ";".join(_get(r, "topics") or []),
            }
    elif name == "lenta":
        # В lenta-ru-news.csv.gz нет колонки даты — извлекаем из URL:
        # https://lenta.ru/news/2018/12/14/cancer/ -> 2018-12-14
        for r in load_lenta(str(path)):
            url = _get(r, "url") or ""
            m = LENTA_URL_DATE_RE.search(url)
            yield {
                "date": "-".join(m.groups()) if m else None,
                "title": _get(r, "title"),
                "text": _get(r, "text"),
                "url": url,
                "rubric": _get(r, "topic") or "",
            }
    else:
        raise ValueError(f"Конвертер для '{name}' пока не реализован")


def convert_to_interim(
    name: str,
    archive: str | Path,
    out_dir: str | Path,
    date_range: tuple[str, str],
    rubric_filter: list[str] | None = None,
    limit: int | None = None,
    batch_size: int = 200_000,
) -> int:
    """Архив → фильтр по датам/рубрикам → parquet-парты в interim.

    Возвращает число сохранённых документов.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lo, hi = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    rubrics = {r.lower() for r in rubric_filter} if rubric_filter else None

    buf: list[dict] = []
    parts = kept = seen = 0

    def flush():
        nonlocal parts, buf
        if not buf:
            return
        df = pd.DataFrame(buf, columns=SCHEMA_COLS)
        path = out / f"news_{name}_part{parts:03d}.parquet"
        df.to_parquet(path, index=False)
        print(f"  saved {path.name}: {len(df)} rows")
        parts += 1
        buf = []

    for row in _iter_rows(name, archive):
        seen += 1
        if limit and seen > limit:
            break
        if row["date"] is None:
            continue
        ts = pd.Timestamp(row["date"])
        if ts < lo or ts > hi:
            continue
        if rubrics and row["rubric"].lower() not in rubrics:
            continue
        row["source"] = name
        buf.append(row)
        kept += 1
        if len(buf) >= batch_size:
            flush()
    flush()
    print(f"[{name}] просмотрено {seen}, сохранено {kept}")
    return kept
