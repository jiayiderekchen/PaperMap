"""
Scraper for AAAI papers from the AAAI OJS proceedings site:
  https://ojs.aaai.org/index.php/AAAI/

Each AAAI conference year maps to a consecutive range of OJS issue IDs.
AAAI 2026 (Vol. 40) occupies issues 683–729.
"""

import re
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://ojs.aaai.org/index.php/AAAI"

# Issue ID ranges for each AAAI year (inclusive).
# Add new entries here as new proceedings are published.
YEAR_ISSUE_RANGES: Dict[int, Tuple[int, int]] = {
    2026: (683, 729),
    2025: (601, 643),  # AAAI-25, Vol. 39 — approximate; adjust if needed
}

DEFAULT_SLEEP_S = 0.5
DEFAULT_MAX_RETRIES = 4


def _get(
    session: requests.Session,
    url: str,
    sleep_s: float = DEFAULT_SLEEP_S,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> Optional[BeautifulSoup]:
    """Fetch a URL with retries and return a parsed BeautifulSoup, or None on failure."""
    for attempt in range(max_retries):
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "lxml")
        except requests.RequestException as exc:
            wait = sleep_s * (2 ** attempt)
            print(f"  [retry {attempt+1}/{max_retries}] {url}: {exc} — waiting {wait:.1f}s")
            time.sleep(wait)
    print(f"  [skip] failed after {max_retries} retries: {url}")
    return None


def _get_issue_article_ids(
    soup: BeautifulSoup,
) -> List[int]:
    """Extract all article IDs linked from an issue page."""
    ids: List[int] = []
    seen = set()
    for a in soup.find_all("a", href=True):
        m = re.search(r"/article/view/(\d+)$", a["href"])
        if m:
            aid = int(m.group(1))
            if aid not in seen:
                seen.add(aid)
                ids.append(aid)
    return ids


def _get_issue_track_name(soup: BeautifulSoup) -> str:
    """Extract the track/section name from an issue page (first h2 heading)."""
    h2 = soup.find("h2")
    if h2:
        return h2.get_text(strip=True)
    # Fallback to page title
    h1 = soup.find("h1", class_="page_title")
    if h1:
        return h1.get_text(strip=True)
    return ""


def _parse_article(soup: BeautifulSoup, article_id: int, year: int, track: str) -> Optional[Dict[str, Any]]:
    """Parse a single article page into a normalized paper record."""
    # Title
    title_tag = soup.find("h1", class_="page_title")
    title = title_tag.get_text(strip=True) if title_tag else ""
    if not title:
        return None

    # Abstract — the section element's text begins with the label "Abstract"
    abstract = ""
    abstract_tag = soup.find("section", class_="abstract")
    if abstract_tag:
        raw = abstract_tag.get_text(separator=" ", strip=True)
        abstract = re.sub(r"^Abstract\s*", "", raw, flags=re.IGNORECASE).strip()

    # Authors
    authors: List[str] = []
    for name_tag in soup.select(".authors .name"):
        name = name_tag.get_text(strip=True)
        if name:
            authors.append(name)

    # DOI
    doi_url: Optional[str] = None
    doi_tag = soup.find("section", class_="doi")
    if doi_tag:
        doi_text = doi_tag.get_text(strip=True)
        m = re.search(r"https?://doi\.org/\S+", doi_text)
        if m:
            doi_url = m.group(0)

    paper_url = f"{BASE_URL}/article/view/{article_id}"

    return {
        "paper_id": str(article_id),
        "title": title,
        "abstract": abstract,
        "keywords": [],
        "authors": authors,
        "venue": f"AAAI {year}",
        "track": track,
        "year": year,
        "paper_url": paper_url,
        "pdf_url": doi_url,
        "source": "aaai_ojs",
    }


def crawl_aaai_ojs(
    year: int,
    sleep_s: float = DEFAULT_SLEEP_S,
    max_retries: int = DEFAULT_MAX_RETRIES,
    verbose: bool = True,
    start_issue: Optional[int] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Yield normalized paper records for every AAAI paper in the given year.

    Iterates over all OJS issues for that year, collects article IDs from
    each issue's table of contents, then fetches individual article pages.
    """
    if year not in YEAR_ISSUE_RANGES:
        raise ValueError(
            f"No issue range configured for AAAI {year}. "
            f"Add an entry to YEAR_ISSUE_RANGES in aaai_ojs_client.py. "
            f"Known years: {sorted(YEAR_ISSUE_RANGES.keys())}"
        )

    start_id, end_id = YEAR_ISSUE_RANGES[year]
    if start_issue is not None:
        start_id = max(start_id, start_issue)
    session = requests.Session()
    session.headers["User-Agent"] = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )

    total_articles = 0

    for issue_id in range(start_id, end_id + 1):
        issue_url = f"{BASE_URL}/issue/view/{issue_id}"
        if verbose:
            print(f"Issue {issue_id}: {issue_url}")

        soup = _get(session, issue_url, sleep_s=sleep_s, max_retries=max_retries)
        if soup is None:
            continue

        article_ids = _get_issue_article_ids(soup)
        track = _get_issue_track_name(soup)
        if not article_ids:
            if verbose:
                print(f"  No articles found — skipping")
            continue

        if verbose:
            print(f"  Track: {track!r}  |  {len(article_ids)} articles")

        for article_id in article_ids:
            article_url = f"{BASE_URL}/article/view/{article_id}"
            time.sleep(sleep_s)
            art_soup = _get(
                session, article_url, sleep_s=sleep_s, max_retries=max_retries
            )
            if art_soup is None:
                continue

            paper = _parse_article(art_soup, article_id, year, track)
            if paper:
                total_articles += 1
                yield paper

        time.sleep(sleep_s)

    if verbose:
        print(f"\nTotal papers scraped: {total_articles}")
