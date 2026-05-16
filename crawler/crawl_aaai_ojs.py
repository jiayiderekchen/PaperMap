"""
CLI entry point for crawling AAAI papers from ojs.aaai.org.

Usage (from project root):
    python crawler/crawl_aaai_ojs.py --year 2026 --out-dir aaai2026/data
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

try:
    from crawler.aaai_ojs_client import crawl_aaai_ojs, YEAR_ISSUE_RANGES
except ModuleNotFoundError:
    from aaai_ojs_client import crawl_aaai_ojs, YEAR_ISSUE_RANGES


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def run() -> None:
    parser = argparse.ArgumentParser(description="Crawl AAAI papers from ojs.aaai.org")
    parser.add_argument("--year", type=int, default=2026, help=f"Conference year. Known: {sorted(YEAR_ISSUE_RANGES.keys())}")
    parser.add_argument("--out-dir", default="aaai2026/data", help="Output directory for papers.parquet / papers.jsonl")
    parser.add_argument("--sleep-s", type=float, default=0.5, help="Seconds to sleep between requests")
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--start-issue", type=int, default=None, help="Resume from this issue ID (merges with existing data)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    new_papers = list(
        crawl_aaai_ojs(
            year=args.year,
            sleep_s=args.sleep_s,
            max_retries=args.max_retries,
            verbose=True,
            start_issue=args.start_issue,
        )
    )

    jsonl_path = out_dir / "papers.jsonl"
    parquet_path = out_dir / "papers.parquet"
    meta_path = out_dir / "crawl_meta.json"

    # Merge with existing data when resuming
    if args.start_issue is not None and jsonl_path.exists():
        existing: List[Dict[str, Any]] = []
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.append(json.loads(line))
        existing_ids = {p["paper_id"] for p in existing}
        added = [p for p in new_papers if p["paper_id"] not in existing_ids]
        papers = existing + added
        print(f"\nMerged {len(existing)} existing + {len(added)} new = {len(papers)} total papers")
    else:
        papers = new_papers

    _write_jsonl(jsonl_path, papers)
    pd.DataFrame(papers).to_parquet(parquet_path, index=False)
    with meta_path.open("w") as f:
        json.dump({"year": args.year, "count": len(papers), "source": "aaai_ojs"}, f, indent=2)

    print(f"Wrote {len(papers)} papers → {parquet_path}")


if __name__ == "__main__":
    run()
