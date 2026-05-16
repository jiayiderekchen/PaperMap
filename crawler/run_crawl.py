import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

try:
    from crawler.openreview_client import OpenReviewClient, dump_json
except ModuleNotFoundError:
    from openreview_client import OpenReviewClient, dump_json


DEFAULT_INVITATION = "ICLR.cc/2026/Conference/-/Submission"
DEFAULT_VENUEID = "ICLR.cc/2026/Conference"

# Conferences that use a .org TLD on OpenReview instead of .cc
_ORG_DOMAIN_CONFERENCES = {"AAAI"}


def _conference_domain(conference: str) -> str:
    """Return the OpenReview domain suffix for a given conference."""
    if conference in _ORG_DOMAIN_CONFERENCES:
        return f"{conference}.org"
    return f"{conference}.cc"


def _get_default_invitation(year: int, conference: str = "ICLR") -> str:
    """Get default invitation based on year and conference."""
    return f"{_conference_domain(conference)}/{year}/Conference/-/Submission"


def _get_default_venueid(year: int, conference: str = "ICLR") -> str:
    """Get default venueid based on year and conference."""
    return f"{_conference_domain(conference)}/{year}/Conference"


def _build_query(args: argparse.Namespace) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if args.invitation:
        params["invitation"] = args.invitation
    if args.venueid:
        params["content.venueid"] = args.venueid
    if args.venue:
        params["content.venue"] = args.venue
    if not params:
        # Default to fetching accepted papers via submission invitation + venueid filter
        conference = args.conference if hasattr(args, 'conference') else "ICLR"
        params["invitation"] = _get_default_invitation(args.year, conference)
        params["content.venueid"] = _get_default_venueid(args.year, conference)
    return params


def _venue_for_track(track: str, year: int) -> str:
    track = track.lower()
    if track == "oral":
        return f"ICLR {year} Oral"
    if track == "poster":
        return f"ICLR {year} Poster"
    raise ValueError(f"Unsupported track: {track}")


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def run() -> None:
    parser = argparse.ArgumentParser(description="Crawl accepted papers from OpenReview.")
    parser.add_argument("--base-url", default="https://api2.openreview.net")
    parser.add_argument("--invitation", default=None)
    parser.add_argument("--venueid", default=None)
    parser.add_argument("--venue", default=None)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--conference", default="ICLR", choices=["ICLR", "NeurIPS", "ICML", "AAAI"], help="Conference name")
    parser.add_argument("--accepted-only", action="store_true")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--sleep-s", type=float, default=1.0)
    parser.add_argument("--page-sleep", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--username", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--list-invitations", action="store_true")
    parser.add_argument("--acceptance-track", choices=["oral", "poster", "both"], default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = _build_query(args)
    username = args.username or os.getenv("OPENREVIEW_USERNAME")
    password = args.password or os.getenv("OPENREVIEW_PASSWORD")

    client = OpenReviewClient(
        base_url=args.base_url,
        max_retries=args.max_retries,
        sleep_s=args.sleep_s,
        page_sleep_s=args.page_sleep,
        username=username,
        password=password,
    )
    if args.list_invitations:
        prefix = f"{_conference_domain(args.conference)}/{args.year}"
        invitations = client.list_invitations(prefix=prefix)
        print(f"Invitations under {prefix}:")
        for inv in invitations:
            print(inv)
        return

    papers: List[Dict[str, Any]] = []
    if args.acceptance_track:
        tracks = ["oral", "poster"] if args.acceptance_track == "both" else [args.acceptance_track]
        for track in tracks:
            venue_value = _venue_for_track(track, args.year)
            track_params = {"content.venue": venue_value}
            papers.extend(
                client.crawl_papers(
                    track_params,
                    year=args.year,
                    accepted_only=False,
                    limit=args.limit,
                )
            )
        seen = {}
        for paper in papers:
            seen[paper["paper_id"]] = paper
        papers = list(seen.values())
    else:
        papers = client.crawl_papers(
            params,
            year=args.year,
            accepted_only=args.accepted_only,
            limit=args.limit,
        )
    if not papers and not (args.invitation or args.venueid or args.venue):
        discovered = client.find_accepted_invitation(args.year)
        if discovered and discovered != params.get("invitation"):
            print(f"No papers found. Retrying with discovered invitation: {discovered}")
            params = {"invitation": discovered}
            papers = client.crawl_papers(
                params,
                year=args.year,
                accepted_only=args.accepted_only,
                limit=args.limit,
            )

    jsonl_path = out_dir / "papers.jsonl"
    parquet_path = out_dir / "papers.parquet"
    meta_path = out_dir / "crawl_meta.json"

    _write_jsonl(jsonl_path, papers)
    pd.DataFrame(papers).to_parquet(parquet_path, index=False)
    dump_json(
        str(meta_path),
        {
            "query": params,
            "count": len(papers),
            "year": args.year,
            "base_url": args.base_url,
        },
    )

    print(f"Wrote {len(papers)} papers to {jsonl_path} and {parquet_path}")


if __name__ == "__main__":
    run()
