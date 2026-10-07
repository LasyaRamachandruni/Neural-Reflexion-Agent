import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from execute_tools import make_execute_tools, normalize_search_response, run_search
from tests.fakes import FakeTavily, tavily_response


def test_dict_response_is_normalized():
    raw = tavily_response("ai for small business", [
        {"url": "https://example.com/a", "title": "A", "content": "alpha"},
        {"url": "https://example.com/b", "title": "B", "content": "beta"},
    ])
    results, error = normalize_search_response(raw)
    assert error is None
    assert results == [
        {"title": "A", "url": "https://example.com/a", "content": "alpha"},
        {"title": "B", "url": "https://example.com/b", "content": "beta"},
    ]


def test_json_string_response_is_normalized():
    raw = json.dumps(tavily_response("q", [{"url": "https://example.com/a", "title": "A"}]))
    results, error = normalize_search_response(raw)
    assert error is None and results[0]["url"] == "https://example.com/a"


def test_error_shapes_are_reported_as_errors():
    assert normalize_search_response({"error": ValueError("bad key")}) == ([], "bad key")
    assert normalize_search_response("No search results found for 'x'.")[1].startswith("No search results")
    assert normalize_search_response({"results": []}) == ([], "no results")
    assert normalize_search_response(None)[1] == "empty response"


def test_run_search_catches_exceptions():
    out = run_search(FakeTavily(fail=True), "anything")
    assert "error" in out and "tavily unreachable" in out["error"]


def test_execute_tools_returns_real_urls():
    tool = FakeTavily({"q1": tavily_response("q1", [{"url": "https://example.com/a", "title": "A", "content": "x"}])})
    node = make_execute_tools(tool)
    state = [
        HumanMessage(content="question"),
        AIMessage(content="", tool_calls=[{"name": "AnswerQuestion", "id": "c1",
                                           "args": {"answer": "draft", "search_queries": ['"q1",', "q2"]}}]),
    ]
    [msg] = node(state)
    assert isinstance(msg, ToolMessage) and msg.tool_call_id == "c1"
    data = json.loads(msg.content)
    assert tool.calls == ["q1", "q2"]  # quotes/commas cleaned
    assert data["q1"]["results"][0]["url"] == "https://example.com/a"
    assert "error" in data["q2"]
