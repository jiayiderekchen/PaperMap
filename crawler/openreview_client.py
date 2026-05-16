import json
import time
from typing import Any, Dict, Iterable, List, Optional

import openreview


DEFAULT_BASE_URL = "https://api.openreview.net"
OPENREVIEW_WEB = "https://openreview.net"


def _unwrap_value(value: Any) -> Any:
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def _get_content_field(content: Dict[str, Any], key: str) -> Any:
    if key in content:
        return _unwrap_value(content[key])
    # fallback to case-insensitive match
    lower_key = key.lower()
    for k, v in content.items():
        if k.lower() == lower_key:
            return _unwrap_value(v)
    return None


def _normalize_keywords(value: Any) -> List[str]:
    if value is None:
        return []
    value = _unwrap_value(value)
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        parts = [v.strip() for v in value.split(",")]
        return [p for p in parts if p]
    return [str(value).strip()] if str(value).strip() else []


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    value = _unwrap_value(value)
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _normalize_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if url.startswith("/"):
        return f"{OPENREVIEW_WEB}{url}"
    return url


def _extract_arxiv_url(content: Dict[str, Any]) -> Optional[str]:
    for key in content.keys():
        if "arxiv" in key.lower():
            value = _unwrap_value(content[key])
            if isinstance(value, str) and value.strip():
                if value.startswith("http"):
                    return value
                return f"https://arxiv.org/abs/{value}"
    return None


def _extract_pdf_url(note: Dict[str, Any]) -> Optional[str]:
    content = note.get("content", {}) or {}
    for key in ["pdf", "pdf_url", "pdfUrl"]:
        value = _unwrap_value(content.get(key))
        if isinstance(value, str) and value.strip():
            return _normalize_url(value)
    pdf = note.get("pdf")
    if isinstance(pdf, str) and pdf.strip():
        return _normalize_url(pdf)
    return None


def _extract_best_paper_url(note: Dict[str, Any]) -> Optional[str]:
    # Always prefer the canonical forum URL so the link actually works.
    # The hash-based /pdf/{hash}.pdf paths that OpenReview embeds in notes are
    # internal and frequently return 404 on the public site.
    note_id = note.get("id") or note.get("noteId")
    if note_id:
        return f"{OPENREVIEW_WEB}/forum?id={note_id}"
    # Fallback: arxiv > pdf
    content = note.get("content", {}) or {}
    arxiv_url = _extract_arxiv_url(content)
    if arxiv_url:
        return arxiv_url
    return _extract_pdf_url(note)


def _normalize_note(note: Dict[str, Any], year: int) -> Dict[str, Any]:
    content = note.get("content", {}) or {}
    title = _get_content_field(content, "title") or ""
    abstract = _get_content_field(content, "abstract") or ""
    keywords = _normalize_keywords(
        _get_content_field(content, "keywords") or _get_content_field(content, "keyword")
    )
    authors = _as_list(
        _get_content_field(content, "authors") or _get_content_field(content, "authorids")
    )
    venue = _get_content_field(content, "venue") or _get_content_field(content, "venueid")
    paper_id = note.get("id") or note.get("noteId") or ""
    pdf_url = _extract_pdf_url(note)
    paper_url = _extract_best_paper_url(note)

    return {
        "paper_id": str(paper_id),
        "title": str(title).strip(),
        "abstract": str(abstract).strip(),
        "keywords": keywords,
        "authors": authors,
        "venue": str(venue).strip() if venue else None,
        "year": year,
        "paper_url": paper_url,
        "pdf_url": pdf_url,
        "source": "openreview",
    }


class OpenReviewClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = 30,
        max_retries: int = 5,
        sleep_s: float = 1.0,
        page_sleep_s: float = 0.0,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.sleep_s = sleep_s
        self.page_sleep_s = page_sleep_s
        self.client = self._init_client(username=username, password=password)

    def _init_client(self, username: Optional[str], password: Optional[str]):
        if "api2.openreview.net" in self.base_url:
            if username and password:
                return openreview.api.OpenReviewClient(
                    baseurl=self.base_url, username=username, password=password
                )
            return openreview.api.OpenReviewClient(baseurl=self.base_url)
        if username and password:
            return openreview.Client(baseurl=self.base_url, username=username, password=password)
        return openreview.Client(baseurl=self.base_url)

    def fetch_notes(self, params: Dict[str, Any], limit: int = 1000) -> Iterable[Dict[str, Any]]:
        offset = 0
        while True:
            last_error: Optional[Exception] = None
            for attempt in range(self.max_retries):
                try:
                    notes = self.client.get_notes(limit=limit, offset=offset, **params)
                    if self.page_sleep_s > 0:
                        time.sleep(self.page_sleep_s)
                    if not notes:
                        return
                    for note in notes:
                        yield note.to_json() if hasattr(note, "to_json") else note
                    offset += len(notes)
                    break
                except Exception as exc:
                    last_error = exc
                    wait_s = self.sleep_s * (2 ** attempt)
                    time.sleep(wait_s)
            else:
                raise RuntimeError(f"OpenReview request failed: {last_error}")

    def find_accepted_invitation(self, year: int) -> Optional[str]:
        prefix = f"ICLR.cc/{year}"
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                invitations = self.client.get_invitations(prefix=prefix)
                accepted = [
                    inv.id for inv in invitations if "accepted" in inv.id.lower()
                ]
                if not accepted:
                    return None
                accepted.sort()
                return accepted[0]
            except Exception as exc:
                last_error = exc
                wait_s = self.sleep_s * (2 ** attempt)
                time.sleep(wait_s)
        raise RuntimeError(f"OpenReview invitation lookup failed: {last_error}")

    def list_invitations(self, prefix: str) -> List[str]:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                invitations = self.client.get_invitations(prefix=prefix)
                return sorted([inv.id for inv in invitations])
            except Exception as exc:
                last_error = exc
                wait_s = self.sleep_s * (2 ** attempt)
                time.sleep(wait_s)
        raise RuntimeError(f"OpenReview invitation lookup failed: {last_error}")

    def crawl_papers(
        self,
        params: Dict[str, Any],
        year: int,
        accepted_only: bool = False,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        # Extract content filters that need client-side filtering
        venueid_filter = params.pop("content.venueid", None)
        venue_filter = params.pop("content.venue", None)
        
        papers: List[Dict[str, Any]] = []
        for note in self.fetch_notes(params, limit=limit):
            # Apply client-side filtering for content fields
            if venueid_filter:
                note_content = note.get("content", {}) or {}
                note_venueid = note_content.get("venueid", {})
                if isinstance(note_venueid, dict):
                    note_venueid = note_venueid.get("value", "")
                if str(note_venueid) != venueid_filter:
                    continue
            
            if venue_filter:
                note_content = note.get("content", {}) or {}
                note_venue = note_content.get("venue", {})
                if isinstance(note_venue, dict):
                    note_venue = note_venue.get("value", "")
                if str(note_venue) != venue_filter:
                    continue
            
            normalized = _normalize_note(note, year=year)
            if accepted_only:
                venue = normalized.get("venue") or ""
                if "accepted" not in str(venue).lower():
                    continue
            papers.append(normalized)
        return papers


def dump_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
