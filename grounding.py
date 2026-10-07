# grounding.py
"""Deterministic check that the revised answer only cites retrieved URLs.

The model is asked to cite sources from the search results, but nothing
forces it to. After every revision we:
  * keep a reference only if every URL in it was returned by a search,
  * drop references with no URL or with a URL we never retrieved,
  * renumber the kept references and fix the inline [n] markers to match,
  * remove [n] markers that point at dropped/missing references,
  * remove raw URLs in the answer text that were never retrieved.
Everything that was removed is reported in `dropped` so it can be shown.
"""
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set

from langchain_core.messages import BaseMessage, ToolMessage

from execute_tools import parse_tool_message

URL_RE = re.compile(r"https?://[^\s<>\"'\)\]\}]+")
CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
REF_NUM_RE = re.compile(r"^\s*(?:\[(\d+)\]|(\d+)[.)])\s*")
REFS_HEADING_RE = re.compile(r"\n\s*(?:#+\s*|\*\*)?References:?(?:\*\*)?\s*:?\s*\n", re.IGNORECASE)


def normalize_url(url: str) -> str:
    """Make URL comparison tolerant of trailing punctuation, case and slashes."""
    url = url.strip().rstrip(".,;:!?")
    url = url.split("#", 1)[0]
    m = re.match(r"^(https?)://([^/]+)(.*)$", url, re.IGNORECASE)
    if not m:
        return url
    scheme, host, rest = m.groups()
    host = host.lower()
    if host.startswith("www."):
        host = host[4:]
    return f"{scheme.lower()}://{host}{rest}".rstrip("/")


def find_urls(text: str) -> List[str]:
    return [u.rstrip(".,;:!?") for u in URL_RE.findall(text or "")]


def retrieved_sources(messages: Iterable[BaseMessage]) -> Dict[str, Dict[str, str]]:
    """All search results seen in this run, keyed by normalized URL (in order)."""
    sources: Dict[str, Dict[str, str]] = {}
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        for outcome in parse_tool_message(msg).values():
            if not isinstance(outcome, dict):
                continue
            for item in outcome.get("results") or []:
                url = item.get("url") if isinstance(item, dict) else None
                if url:
                    sources.setdefault(normalize_url(url), item)
    return sources


@dataclass
class GroundingResult:
    answer: str
    references: List[str]
    dropped: List[str] = field(default_factory=list)


def ground_answer(answer: Optional[str], references: Optional[List[str]], allowed_urls: Set[str]) -> GroundingResult:
    """Filter references/citations down to URLs in `allowed_urls` (normalized)."""
    answer = answer or ""
    dropped: List[str] = []

    # The UI/CLI print references separately; a "References" block inside the
    # answer text would bypass the check, so cut it off.
    m = REFS_HEADING_RE.search(answer)
    if m:
        answer = answer[: m.start()].rstrip()

    # Map old reference number -> new reference number
    renumber: Dict[int, int] = {}
    kept: List[str] = []
    for i, ref in enumerate(references or []):
        ref = str(ref).strip()
        num_match = REF_NUM_RE.match(ref)
        old_num = int(num_match.group(1) or num_match.group(2)) if num_match else i + 1
        body = ref[num_match.end():] if num_match else ref
        urls = find_urls(body)
        if urls and all(normalize_url(u) in allowed_urls for u in urls):
            kept.append(f"[{len(kept) + 1}] {body}")
            renumber.setdefault(old_num, len(kept))
        else:
            dropped.append(ref)

    def _fix_cite(match: re.Match) -> str:
        nums = [int(n) for n in re.split(r"\s*,\s*", match.group(1))]
        new = []
        for n in nums:
            if n in renumber and renumber[n] not in new:
                new.append(renumber[n])
        return "".join(f"[{n}]" for n in new)

    answer = CITE_RE.sub(_fix_cite, answer)

    for url in find_urls(answer):
        if normalize_url(url) not in allowed_urls:
            dropped.append(url)
            answer = answer.replace(url, "")

    # Tidy spaces left behind by removed markers/URLs
    answer = re.sub(r"[ \t]+([.,;:!?])", r"\1", answer)
    answer = re.sub(r"[ \t]{2,}", " ", answer).strip()
    return GroundingResult(answer=answer, references=kept, dropped=dropped)
