"""Скрапинг архива Коммерсантъ, рубрика «Экономика» (Этап 1b).

Страница дня https://www.kommersant.ru/archive/rubric/3/day/{YYYY-MM-DD}
содержит JSON-LD ItemList со всеми статьями дня (title + url, пагинации нет).
У каждой статьи /doc/{id}: точное время в meta article:published_time,
полный текст — абзацы div.article_text_wrapper.

Сырые данные: один JSONL на день, data/raw/news/kommersant/kommersant_YYYY-MM-DD.jsonl.
Файл пишется атомарно после полного успеха по дню — resume бесплатный.
"""
import json
import random
import re
import time
from datetime import date, timedelta
from pathlib import Path

import requests

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
DAY_URL = "https://www.kommersant.ru/archive/rubric/3/day/{date}"
LD_RE = re.compile(r'(?s)<script type="application/ld\+json">(.*?)</script>')
DOC_ID_RE = re.compile(r"/doc/(\d+)")
TIME_META_RE = re.compile(
    r'<meta[^>]*property="article:published_time"[^>]*content="([^"]+)"'
)
BODY_RE = re.compile(
    r'(?s)<div class="article_text_wrapper[^"]*"[^>]*>(.*?)</div>\s*(?:<div|<aside|<section)'
)
P_RE = re.compile(r"<p[^>]*>(.*?)</p>", re.S)
TAG_RE = re.compile(r"<[^>]+>")


def _get(url: str, retries: int = 3, timeout: int = 30) -> requests.Response | None:
    """GET с бэкоффом; None только при устойчивом 404."""
    for attempt in range(retries):
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** (attempt + 1))
            continue
        if r.status_code == 404:
            return None
        if r.status_code == 200:
            return r
        if attempt == retries - 1:
            r.raise_for_status()
        time.sleep(2 ** (attempt + 1))
    return None


def _iter_ld_objects(html: str):
    """Все JSON-объекты из всех ld+json блоков (в т.ч. массивы и конкатенации)."""
    dec = json.JSONDecoder()
    for block in LD_RE.findall(html):
        s = block.strip()
        pos = 0
        while pos < len(s):
            try:
                obj, end = dec.raw_decode(s, pos)
            except json.JSONDecodeError:
                break
            if isinstance(obj, list):
                yield from obj
            else:
                yield obj
            pos += end


def parse_day_listing(html: str) -> list[dict]:
    """Все ссылки /doc/ страницы-архива дня (ItemList ∪ href).

    ItemList содержит лишь часть статей; href включает и сайдбар
    «читайте также» с чужими датами — их отбрасывает фильтр по
    published_time в scrape_day.
    """
    seen: dict[str, dict] = {}
    for obj in _iter_ld_objects(html):
        if isinstance(obj, dict) and obj.get("@type") == "ItemList":
            for el in obj.get("itemListElement", []):
                url = el.get("url") or ""
                dm = DOC_ID_RE.search(url)
                if url and dm:
                    seen[dm.group(1)] = {
                        "title": (el.get("name") or "").strip(),
                        "url": url,
                        "doc_id": dm.group(1),
                    }
    for url in re.findall(r'href="(/doc/\d+)"', html):
        dm = DOC_ID_RE.search(url)
        if dm and dm.group(1) not in seen:
            seen[dm.group(1)] = {
                "title": "",
                "url": f"https://www.kommersant.ru{url}",
                "doc_id": dm.group(1),
            }
    return list(seen.values())


def _clean_para(t: str) -> str:
    t = TAG_RE.sub("", t)
    t = t.replace("&nbsp;", " ").replace("&laquo;", "«").replace("&raquo;", "»")
    t = t.replace("&mdash;", "—").replace("&ndash;", "–").replace("&amp;", "&")
    return re.sub(r"\s+", " ", t).strip()


def parse_article(html: str, url: str) -> dict | None:
    """HTML статьи → {datetime, title, text}; None без published_time."""
    tm = TIME_META_RE.search(html)
    if not tm:
        return None
    bm = BODY_RE.search(html)
    paras = []
    if bm:
        paras = [_clean_para(p) for p in P_RE.findall(bm.group(1))]
    text = "\n".join(p for p in paras if len(p) > 20)
    title_m = re.search(r"<title>(.*?)</title>", html, re.S)
    title = _clean_para(title_m.group(1)) if title_m else ""
    title = re.sub(r"\s*[–-]\s*Коммерсантъ\s*$", "", title)
    return {"datetime": tm.group(1), "title": title, "text": text, "url": url}


def scrape_day(day: date, out_dir: Path, delay: float = 0.6) -> int:
    """Один день → JSONL; пропускает уже готовые дни. Возврат: число статей."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"kommersant_{day.isoformat()}.jsonl"
    if dest.exists():
        return -1

    r = _get(DAY_URL.format(date=day.isoformat()))
    items = [] if r is None else parse_day_listing(r.text)

    rows = []
    kept = 0
    for it in items:
        time.sleep(delay + random.uniform(0, delay * 0.4))
        ar = _get(it["url"])
        rec = None if ar is None else parse_article(ar.text, it["url"])
        # чужие даты из сайдбара «читайте также» — мимо
        if rec and not rec["datetime"].startswith(day.isoformat()):
            continue
        kept += 1
        rows.append({
            "doc_id": it["doc_id"],
            "title": it["title"] or (rec or {}).get("title", ""),
            "url": it["url"],
            **({"error": "no_page"} if ar is None else {}),
            **(rec or {"error": "no_published_time"}),
        })

    tmp = dest.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.rename(dest)
    return kept


def load_days_to_df(raw_dir: str | Path) -> "pd.DataFrame":  # type: ignore[name-defined]
    """Дневные JSONL → DataFrame единой схемы interim (date,source,title,text,url,rubric).

    Строки с error (нет страницы/времени) пропускаются; дедупликация по doc_id.
    """
    import pandas as pd

    rows = []
    seen: set[str] = set()
    for p in sorted(Path(raw_dir).glob("kommersant_*.jsonl")):
        day = p.stem.removeprefix("kommersant_")
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r.get("error") or r["doc_id"] in seen:
                continue
            seen.add(r["doc_id"])
            rows.append({
                "date": r.get("datetime", day)[:10],
                "datetime": r.get("datetime"),
                "source": "kommersant",
                "title": r.get("title", ""),
                "text": r.get("text", ""),
                "url": r["url"],
                "rubric": "экономика",
            })
    return pd.DataFrame(rows)


def scrape_range(
    start: str,
    end: str,
    out_dir: str | Path,
    delay: float = 0.6,
    max_days: int | None = None,
) -> dict:
    """Диапазон дней включительно; сбои дня не останавливают прогон."""
    out_dir = Path(out_dir)
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    done = failed = skipped = total_articles = 0
    n = 0
    day = d0
    while day <= d1:
        if max_days is not None and n >= max_days:
            break
        n += 1
        try:
            k = scrape_day(day, out_dir, delay=delay)
            if k < 0:
                skipped += 1
            else:
                done += 1
                total_articles += k
                print(f"[{day}] статей: {k}", flush=True)
        except Exception as e:
            failed += 1
            print(f"[{day}] FAIL: {type(e).__name__}: {e}", flush=True)
        day += timedelta(days=1)
    stats = {"days_done": done, "days_skipped": skipped, "days_failed": failed,
             "articles": total_articles}
    print(f"[итог] {stats}", flush=True)
    return stats
