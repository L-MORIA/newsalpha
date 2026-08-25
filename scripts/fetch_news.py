"""Скачивание готовых корпусов новостей и конвертация в interim (Этап 1a).

Примеры:
    python scripts/fetch_news.py --sources lenta --limit 5000        # smoke
    python scripts/fetch_news.py --sources lenta,ria                 # полный прогон
"""
import argparse
import sys
import traceback
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from newsalpha.io.fetchers import DOWNLOAD_URLS, convert_to_interim, download_archive


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--sources", default="lenta,ria",
                    help="через запятую: lenta,ria")
    ap.add_argument("--skip-download", action="store_true",
                    help="конвертировать уже скачанные архивы")
    ap.add_argument("--limit", type=int, default=None,
                    help="макс. записей на источник (smoke-тест)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    news_dir = Path(cfg["data"]["news_dir"])
    interim_dir = Path(cfg["data"]["interim_dir"])
    sources_cfg = {s["name"]: s for s in cfg["data"]["news_sources"]}

    failures = []
    for name in [s.strip() for s in args.sources.split(",") if s.strip()]:
        if name not in sources_cfg:
            print(f"[warn] источник '{name}' отсутствует в конфиге — пропуск")
            continue
        if name not in DOWNLOAD_URLS:
            print(f"[warn] у источника '{name}' нет готового архива "
                  f"(скрапинг реализуется отдельно) — пропуск")
            failures.append(name)
            continue
        src = sources_cfg[name]
        print(f"=== {name} ({src.get('range')}) ===", flush=True)
        try:
            archive = None
            if not args.skip_download:
                archive = download_archive(name, news_dir)
            else:
                candidates = list(news_dir.glob(Path(DOWNLOAD_URLS[name]).name))
                archive = candidates[0] if candidates else None
                if archive is None:
                    raise FileNotFoundError(
                        f"архив {name} не найден в {news_dir}")
            convert_to_interim(
                name,
                archive,
                interim_dir,
                date_range=src["range"],
                rubric_filter=src.get("rubric_filter"),
                limit=args.limit,
            )
        except Exception:
            failures.append(name)
            print(f"[error] источник '{name}' упал:", flush=True)
            traceback.print_exc()

    if failures:
        print(f"[done] с ошибками: {', '.join(failures)}")
        sys.exit(1)
    print("[done] ok")


if __name__ == "__main__":
    main()
