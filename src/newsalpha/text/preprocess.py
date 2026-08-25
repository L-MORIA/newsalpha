"""Препроцессинг новостей до формата колонки `preproc` авторов (Этап 2).

Пайплайн (патент EA044248B1 + сэмплы авторов):
razdel-токены → выкидываем токены без букв и чистые цифры → pymystem3-леммы →
NLTK RU stopwords → дефис внутри леммы заменяется на "_" → склейка NER-спанов
(natasha PER/ORG/LOC) через "_" → строка лемм через пробел.

Лемматизация — файловый батч-режим бинарника mystem напрямую: обёртка
pymystem3 на Windows перезапускает процесс на каждую строку (~27 мс/токен),
файловый режим даёт ~0.03 мс/токен. С флагом -c каждая входная строка даёт
ровно одну выходную JSON-строку → соответствие 1:1.

Отклонение от сэмплов (документировано): авторы склеивают и часть ADJ+NOUN
биграмм («национальный_валюта», «сельский_хозяйство») вне NER; правило не
восстанавливается по 5 строкам — держим только воспроизводимые склейки
(NER + дефисы). Для BoW/LDA влияние вторично; A/B при необходимости.
"""
import json
import os
import re
import subprocess
import tempfile

from razdel import tokenize as _razdel_tokenize

_HAS_LETTER = re.compile(r"[а-яёa-z]", re.IGNORECASE)
_HYPHEN = re.compile(r"-+")
MYSTEM_CHUNK = 200_000  # токенов за один вызов бинарника


def load_stopwords() -> set[str]:
    from nltk.corpus import stopwords

    return set(stopwords.words("russian"))


def keep_token(tok: str) -> bool:
    """Токен должен содержать букву (выкидывает числа, пунктуацию, $, %)."""
    return bool(_HAS_LETTER.search(tok))


def _mystem_bin() -> str:
    bin_path = os.environ.get("MYSTEM_BIN")
    if bin_path and os.path.isfile(bin_path):
        return bin_path
    from pymystem3.constants import MYSTEM_BIN

    if not os.path.isfile(MYSTEM_BIN):
        from pymystem3 import autoinstall

        autoinstall()
    return MYSTEM_BIN


class Preprocessor:
    def __init__(self, ner_glue: bool = True):
        self.stop = load_stopwords()
        self.ner_glue = ner_glue
        self._bin = _mystem_bin()
        self._tmpdir = tempfile.mkdtemp(prefix="mystem_")
        self._segmenter = None
        self._ner = None
        if ner_glue:
            from natasha import NewsEmbedding, NewsNERTagger, Segmenter

            self._segmenter = Segmenter()
            self._ner = NewsNERTagger(NewsEmbedding())

    def _lemma_groups(self, tokens: list[str]) -> list[str]:
        """Лемма для каждого токена через файловые батчи mystem.

        С -c весь вход копируется в вывод: строка i входа → строка i выхода
        (JSON-массив кусков). Лемма — первый непустой analysis[].lex,
        иначе склеенный текст строки в нижнем регистре.
        """
        out: list[str] = []
        fin = os.path.join(self._tmpdir, "in.txt")
        fout = os.path.join(self._tmpdir, "out.json")
        for i in range(0, len(tokens), MYSTEM_CHUNK):
            chunk = tokens[i : i + MYSTEM_CHUNK]
            with open(fin, "w", encoding="utf-8") as f:
                f.write("\n".join(chunk))
            subprocess.run(
                [self._bin, "--format", "json", "-c", fin, fout],
                check=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            with open(fout, encoding="utf-8") as f:
                for line_no, line in enumerate(f):
                    lem = None
                    text_parts = []
                    try:
                        items = json.loads(line)
                    except json.JSONDecodeError:
                        items = []
                    for it in items or []:
                        text_parts.append(it.get("text", ""))
                        for a in it.get("analysis") or []:
                            lex = a.get("lex")
                            if lex and lem is None:
                                lem = lex
                    if lem is None:
                        lem = "".join(text_parts) or chunk[line_no]
                    out.append(lem.strip().lower())
        return out

    def _ner_spans(self, text: str) -> list[tuple[int, int]]:
        """Char-спаны сущностей PER/ORG/LOC."""
        if not self._ner:
            return []
        from natasha import Doc

        doc = Doc(text)
        doc.segment(self._segmenter)
        doc.tag_ner(self._ner)
        return [(sp.start, sp.stop) for sp in doc.spans]

    def preprocess_texts(self, texts: list[str], doc_chunk: int = 1000) -> list[str]:
        """Список текстов → список строк preproc.

        Токены всего чанка лемматизируются одним файловым батчем mystem
        (офсет-маппинг обратно на документы); NER-спаны считаются до сборки.
        """
        results: list[str] = []
        for i in range(0, len(texts), doc_chunk):
            chunk = texts[i : i + doc_chunk]
            spans_all = (
                [self._ner_spans(t) for t in chunk]
                if self.ner_glue
                else [[] for _ in chunk]
            )
            tokenized = [list(_razdel_tokenize(t)) for t in chunk]

            flat: list[str] = []
            flags: list[bool] = []
            offsets = [0]
            for toks, spans in zip(tokenized, spans_all):
                for tk in toks:
                    if not keep_token(tk.text):
                        continue
                    flat.append(tk.text)
                    flags.append(
                        any(s <= tk.start and tk.stop <= e for s, e in spans)
                    )
                offsets.append(len(flat))

            lemmas = self._lemma_groups(flat)

            for k in range(len(tokenized)):
                lo, hi = offsets[k], offsets[k + 1]
                parts: list[str] = []
                glued: list[str] = []
                for j in range(lo, hi):
                    lem = _HYPHEN.sub("_", lemmas[j])
                    if not keep_token(lem) or lem in self.stop:
                        continue
                    if flags[j]:
                        glued.append(lem)
                    else:
                        if glued:
                            parts.append("_".join(glued))
                            glued = []
                        parts.append(lem)
                if glued:
                    parts.append("_".join(glued))
                results.append(" ".join(parts))
        return results


def idf_vocab(
    preproc_docs: list[str],
    low_q: float = 0.05,
    high_q: float = 0.95,
) -> tuple[list[str], dict]:
    """Словарь после отсечения квантилей document-frequency.

    DF-квантили: выбрасываются самые редкие low_q и самые частые high_q доли
    словаря (по рангу DF). Возвращает (словарь, статистика).
    """
    from collections import Counter

    df: Counter = Counter()
    for d in preproc_docs:
        df.update(set(d.split()))
    n_words = len(df)
    freqs = sorted(df.values())
    lo_thresh = freqs[int(low_q * n_words)]
    hi_thresh = freqs[min(int(high_q * n_words), n_words - 1)]
    vocab = sorted(w for w, c in df.items() if lo_thresh <= c <= hi_thresh)
    stats = {
        "words_total": n_words,
        "df_lo": lo_thresh,
        "df_hi": hi_thresh,
        "words_kept": len(vocab),
    }
    return vocab, stats


_WORKER_PP: Preprocessor | None = None


def _worker_init(ner_glue: bool) -> None:
    global _WORKER_PP
    _WORKER_PP = Preprocessor(ner_glue=ner_glue)


def _worker_run(chunk: list[str]) -> list[str]:
    assert _WORKER_PP is not None
    return _WORKER_PP.preprocess_texts(chunk)


def preprocess_parallel(
    texts: list[str],
    ner_glue: bool = True,
    n_jobs: int = 1,
    doc_chunk: int = 1000,
) -> list[str]:
    """preprocess_texts с опциональным пулом процессов (узкое место — natasha)."""
    if n_jobs <= 1:
        pp = Preprocessor(ner_glue=ner_glue)
        return pp.preprocess_texts(texts, doc_chunk=doc_chunk)
    from concurrent.futures import ProcessPoolExecutor

    chunks = [texts[i : i + doc_chunk] for i in range(0, len(texts), doc_chunk)]
    out: list[list[str]] = []
    with ProcessPoolExecutor(
        max_workers=n_jobs, initializer=_worker_init, initargs=(ner_glue,)
    ) as ex:
        out = list(ex.map(_worker_run, chunks))
    return [r for chunk_res in out for r in chunk_res]
