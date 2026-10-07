"""Test doubles for the Tavily tool and the Gemini chat model (no network, no keys)."""
import itertools
from typing import Any, Dict, List

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda


def tavily_response(query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Same shape langchain_tavily.TavilySearch.invoke returns."""
    return {
        "query": query,
        "follow_up_questions": None,
        "answer": None,
        "images": [],
        "results": [
            {"url": r["url"], "title": r.get("title", ""), "content": r.get("content", ""),
             "score": 0.9, "raw_content": None}
            for r in results
        ],
        "response_time": 0.5,
    }


class FakeTavily:
    """Returns canned responses keyed by query; unknown queries get no results."""

    def __init__(self, responses: Dict[str, Any] = None, fail: bool = False):
        self.responses = responses or {}
        self.fail = fail
        self.calls: List[str] = []

    def invoke(self, payload):
        query = payload["query"]
        self.calls.append(query)
        if self.fail:
            raise ConnectionError("tavily unreachable")
        if query in self.responses:
            return self.responses[query]
        # what TavilySearch returns (as a string) when nothing is found
        return f"No search results found for '{query}'."


class FakeLLM:
    """Scripted chat model. bind_tools(tool_choice=X) replays the args queued for X.

    The queues restart whenever a new draft (AnswerQuestion) is requested, so one
    compiled graph can be invoked several times with the same script.
    """

    def __init__(self, scripts: Dict[str, List[Dict[str, Any]]]):
        self._original = {name: list(items) for name, items in scripts.items()}
        self.scripts = {name: list(items) for name, items in scripts.items()}
        self.seen_inputs: Dict[str, list] = {name: [] for name in scripts}
        self._ids = itertools.count(1)

    def bind_tools(self, tools, tool_choice=None, **kwargs):
        name = tool_choice

        def _respond(prompt_value):
            self.seen_inputs.setdefault(name, []).append(prompt_value.to_messages())
            if name == "AnswerQuestion":
                self.scripts = {n: list(items) for n, items in self._original.items()}
            queue = self.scripts[name]
            args = queue.pop(0) if len(queue) > 1 else queue[0]
            return AIMessage(
                content="",
                tool_calls=[{"name": name, "args": args, "id": f"call_{next(self._ids)}"}],
            )

        return RunnableLambda(_respond)
