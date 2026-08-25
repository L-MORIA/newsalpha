"""Скрапинг архива Коммерсантъ «Экономика» 2013–2021 (Этап 1b).

Примеры:
    python scripts/fetch_kommersant.py --start 2019-06-01 --end 2019-06-03   # smoke
    python scripts/fetch_kommersant.py                                        # полный прогон
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2013-01-01")
    ap.add_argument("--end", default="2021-12-31")
    ap.add_argument("--out", default="data/raw/news/kommersant")
    ap.add_argument("--delay", type=float, default=0.6,
                    help="пауза между запросами статей, сек")
    ap.add_argument("--max-days", type=int, default=None)
    args = ap.parse_args()

    from newsalpha.io.kommersant import scrape_range

    scrape_range(
        args.start, args.end,
        out_dir=ROOT / args.out,
        delay=args.delay,
        max_days=args.max_days,
    )


if __name__ == "__main__":
    main()
