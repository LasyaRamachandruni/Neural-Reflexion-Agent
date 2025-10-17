# execute_tools.py
import json
from typing import List, Dict, Any
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_tavily import TavilySearch  # NEW package (install: pip install -U langchain-tavily)

# Create the Tavily search tool
tavily_tool = TavilySearch(max_results=5)

# Track unique URLs seen across iterations
SEEN_URLS: set[str] = set()

def _clean_query(q: str) -> str:
    if not isinstance(q, str):
        q = str(q)
    # remove trailing commas and surrounding quotes/whitespace
    return q.strip().strip(",").strip('"').strip("'")

def execute_tools(state: List[BaseMessage]) -> List[BaseMessage]:
    """Execute search queries produced by AnswerQuestion / ReviseAnswer tool calls."""
    if not state:
        return []

    last = state[-1]
    if not isinstance(last, AIMessage):
        return []

    if not getattr(last, "tool_calls", None):
        return []

    tool_messages: List[ToolMessage] = []

    for tool_call in last.tool_calls:
        name = tool_call.get("name")
        if name not in ("AnswerQuestion", "ReviseAnswer"):
            continue

        call_id = tool_call.get("id")
        args = tool_call.get("args") or {}
        search_queries = args.get("search_queries", []) or []

        query_results: Dict[str, Any] = {}
        for query in search_queries:
            query = _clean_query(query)
            if not query:
                continue
            try:
                raw = tavily_tool.invoke({"query": query})
                # Compact projection: keep essentials and unique URLs
                compact = []
                for item in (raw or []):
                    url = (item.get("url") or "").strip()
                    if not url or url in SEEN_URLS:
                        continue
                    SEEN_URLS.add(url)
                    compact.append(
                        {
                            "title": item.get("title"),
                            "url": url,
                            "snippet": item.get("content") or item.get("snippet"),
                        }
                    )
                query_results[query] = compact[:3]  # top-3
            except Exception as e:
                query_results[query] = {"error": repr(e)}

        tool_messages.append(
            ToolMessage(
                content=json.dumps(query_results),
                tool_call_id=call_id,
            )
        )

    return tool_messages

