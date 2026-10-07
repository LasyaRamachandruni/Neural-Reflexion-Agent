# scoring.py
"""Heuristic quality score used to decide when to stop revising.

This is a hand-written checklist, not a learned reward. It is computed on the
answer *after* the grounding check, so only citations to retrieved URLs count:

  length          up to 30  (peaks at ~250 words)
  references      up to 20  (5 per grounded reference)
  inline [n]      up to 20  (4 per marker that points at a grounded reference)
  searches        up to 30  (10 per query that actually returned results, max 3)

The loop stops when a revision fails to beat the best score so far in the
same run, or when the iteration cap is hit.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import BaseMessage, ToolMessage

from execute_tools import parse_tool_message

CITE_NUM_RE = re.compile(r"\[(\d+)\]")


def evaluate_answer(answer: Optional[str], grounded_refs: Optional[List[str]], successful_queries: Optional[List[str]]) -> float:
    if not answer:
        return 0.0
    score = 0.0

    words = len(answer.split())
    score += max(0.0, 30 - abs(words - 250) * 0.1)

    refs = grounded_refs or []
    score += min(20, 5 * len(refs))

    valid_cites = [n for n in CITE_NUM_RE.findall(answer) if 1 <= int(n) <= len(refs)]
    score += min(20, 4 * len(valid_cites))

    score += 10 * min(3, len(successful_queries or []))
    return round(score, 2)


def successful_queries(tool_output: Dict[str, Any]) -> List[str]:
    """Queries in one execute_tools output that returned at least one result."""
    ok = []
    for query, outcome in tool_output.items():
        if isinstance(outcome, dict) and outcome.get("results") and not outcome.get("error"):
            ok.append(query)
    return ok


def latest_search_outcome(messages: List[BaseMessage]) -> Dict[str, Any]:
    """The decoded output of the most recent execute_tools step."""
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            return parse_tool_message(msg)
    return {}


def extract_last_tool_answer(messages: List[BaseMessage],
                             names: Tuple[str, ...] = ("ReviseAnswer", "AnswerQuestion")) -> Tuple[Optional[str], List[str]]:
    """Find latest tool-produced answer + references (if any)."""
    for msg in reversed(messages):
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            continue
        for tc in reversed(tool_calls):
            if tc.get("name") in names:
                args = tc.get("args") or {}
                return args.get("answer"), list(args.get("references") or [])
    return None, []
