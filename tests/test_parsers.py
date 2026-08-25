"""Тесты парсеров источников (fixture = реальные HTML-страницы Ъ, 17.06.2019).

Защищают целостность данных: тихая регрессия парсера = неверные даты/тексты
= шум в метках и мусорный Sharpe на этапах 6-7.
"""
from pathlib import Path

import pytest

from newsalpha.io.fetchers import LENTA_URL_DATE_RE
from newsalpha.io.kommersant import parse_article, parse_day_listing

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def day_html() -> str:
    return (FIXTURES / "ka_day_2019-06-17.html").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def doc_html() -> str:
    return (FIXTURES / "ka_doc_4003487.html").read_text(encoding="utf-8")


def test_day_listing_union_covers_itemlist_and_hrefs(day_html):
    items = parse_day_listing(day_html)
    ids = {i["doc_id"] for i in items}
    # из ItemList (позиция 1 дня) и только из href-ов (сайдбар)
    assert "4004220" in ids
    assert "4002797" in ids
    assert len(ids) >= 25


def test_day_listing_titles_from_itemlist(day_html):
    items = parse_day_listing(day_html)
    by_id = {i["doc_id"]: i for i in items}
    assert "Кудрин" in by_id["4004139"]["title"]
    # каждый элемент валиден: id и абсолютный url
    assert all(i["doc_id"] and i["url"].startswith("https://") for i in items)


def test_day_listing_href_only_fallback():
    html = '<a href="/doc/123">x</a><a href="/doc/123">y</a>'
    items = parse_day_listing(html)
    assert items == [
        {"title": "", "url": "https://www.kommersant.ru/doc/123", "doc_id": "123"}
    ]


def test_day_listing_no_page_without_docs():
    assert parse_day_listing("<html><body>пусто</body></html>") == []


def test_article_extracts_datetime_title_text(doc_html):
    rec = parse_article(doc_html, "https://www.kommersant.ru/doc/4003487")
    assert rec is not None
    assert rec["datetime"] == "2019-06-17T00:20:00+03:00"
    assert rec["title"] == "Россия немного недорабатывает в трудовой сфере"
    assert len(rec["text"]) > 2000
    assert "<" not in rec["text"]
    assert "МОТ" in rec["text"] or "труда" in rec["text"]


def test_article_rejects_missing_published_time():
    assert parse_article("<html><head></head><body>x</body></html>", "u") is None


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://lenta.ru/news/2018/12/14/cancer/", "2018-12-14"),
        ("https://lenta.ru/news/1999/10/04/boot/", "1999-10-04"),
        ("https://lenta.ru/photo/2018/12/14/", None),
        ("", None),
    ],
)
def test_lenta_url_date(url, expected):
    m = LENTA_URL_DATE_RE.search(url)
    assert ("-".join(m.groups()) if m else None) == expected
