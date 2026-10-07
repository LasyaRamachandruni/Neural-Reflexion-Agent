# execute_tools.py
"""Runs the search queries the model asks for and returns them as ToolMessages.

langchain_tavily.TavilySearch.invoke({"query": ...}) returns a dict like
    {"query": ..., "results": [{"title", "url", "content", "score", ...}], ...}
On API errors it returns {"error": <exception>}, and when nothing is found it
returns an error string (handle_tool_error=True). normalize_search_response
turns all of these into ([{title, url, content}], error_or_None).
"""
import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

SEARCH_TOOL_NAMES = ("AnswerQuestion", "ReviseAnswer")
RESULTS_PER_QUERY = 3
MAX_CONTENT_CHARS = 600


def get_default_search_tool(max_results: int = 5):
    """Create the real Tavily tool. Needs TAVILY_API_KEY in the environment."""
    from langchain_tavily import TavilySearch

    return TavilySearch(max_results=max_results)


def _clean_query(q: Any) -> str:
    if not isinstance(q, str):
        q = str(q)
    # remove trailing commas and surrounding quotes/whitespace
    return q.strip().strip(",").strip('"').strip("'").strip()


def normalize_search_response(raw: Any) -> Tuple[List[Dict[str, str]], Optional[str]]:
    """Return (results, error). results is a list of {title, url, content}."""
    if isinstance(raw, str):
        # The tool returns a plain string for "no results" errors; it may also
        # be JSON if the tool was invoked with a ToolCall.
        try:
            raw = json.loads(raw)
        except ValueError:
            return [], raw.strip() or "empty response"

    if isinstance(raw, dict):
        if raw.get("error"):
            return [], str(raw["error"])
        items = raw.get("results")
        if items is None:
            return [], "response has no 'results' field"
    elif isinstance(raw, list):
        items = raw
    elif raw is None:
        return [], "empty response"
    else:
        return [], f"unexpected response type: {type(raw).__name__}"

    results: List[Dict[str, str]] = []
    seen = set()
    for item in items or []:
        if not isinstance(item, dict):
            continue
        url = (item.get("url") or "").strip()
        if not url.startswith(("http://", "https://")) or url in seen:
            continue
        seen.add(url)
        content = (item.get("content") or item.get("snippet") or "").strip()
        results.append(
            {
                "title": (item.get("title") or "").strip(),
                "url": url,
                "content": content[:MAX_CONTENT_CHARS],
            }
        )

    if not results:
        return [], "no results"
    return results, None


def run_search(search_tool, query: str, limit: int = RESULTS_PER_QUERY) -> Dict[str, Any]:
    """Run one query. Returns {"results": [...]} or {"error": "..."}."""
    try:
        raw = search_tool.invoke({"query": query})
    except Exception as e:  # network errors, missing key, etc.
        return {"error": repr(e)}
    results, error = normalize_search_response(raw)
    if error:
        return {"error": error}
    return {"results": results[:limit]}


def make_execute_tools(search_tool=None) -> Callable[[List[BaseMessage]], List[BaseMessage]]:
    """Build the execute_tools graph node around a search tool.

    The tool only needs an .invoke({"query": str}) method, so tests can pass a fake.
    If none is given, a real TavilySearch is created on first use.
    """
    tool_holder = {"tool": search_tool}

    def _tool():
        if tool_holder["tool"] is None:
            tool_holder["tool"] = get_default_search_tool()
        return tool_holder["tool"]

    def execute_tools(state: List[BaseMessage]) -> List[BaseMessage]:
        """Execute search queries produced by AnswerQuestion / ReviseAnswer tool calls."""
        if not state:
            return []
        last = state[-1]
        if not isinstance(last, AIMessage) or not getattr(last, "tool_calls", None):
            return []

        tool_messages: List[ToolMessage] = []
        for tool_call in last.tool_calls:
            name = tool_call.get("name")
            if name not in SEARCH_TOOL_NAMES:
                continue
            args = tool_call.get("args") or {}

            query_results: Dict[str, Any] = {}
            for query in args.get("search_queries") or []:
                query = _clean_query(query)
                if query and query not in query_results:
                    query_results[query] = run_search(_tool(), query)

            # Always answer the tool call, even with no queries, so the
            # conversation stays valid for the model.
            tool_messages.append(
                ToolMessage(
                    content=json.dumps(query_results),
                    tool_call_id=tool_call.get("id"),
                    name=name,
                )
            )
        return tool_messages

    return execute_tools


# Default node using the real Tavily tool (created lazily on first search).
execute_tools = make_execute_tools()


def parse_tool_message(msg: ToolMessage) -> Dict[str, Any]:
    """Decode the JSON written by execute_tools. Returns {} if it can't."""
    try:
        data = json.loads(msg.content)
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}
