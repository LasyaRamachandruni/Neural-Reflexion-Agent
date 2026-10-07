import json

from langchain_core.messages import ToolMessage

from reflexion_agent import build_graph, final_result, print_final
from tests.fakes import FakeLLM, FakeTavily, tavily_response

DRAFT = {
    "answer": "Small businesses can use AI for marketing and support.",
    "reflection": {"missing": "evidence", "superfluous": "none"},
    "search_queries": ["ai chatbots small business"],
}
REVISION_1 = {
    "answer": "AI chatbots handle routine questions [1]. Email tools improve open rates [2].",
    "reflection": {"missing": "costs", "superfluous": "none"},
    "search_queries": ["ai email marketing small business"],
    "references": [
        "[1] Chatbots guide - https://example.com/chatbots",
        "[2] Made-up study - https://not-retrieved.example.net/study",
    ],
}
REVISION_2 = {
    "answer": "AI chatbots handle routine questions [1]. AI email tools improve open rates [2].",
    "reflection": {"missing": "", "superfluous": ""},
    "search_queries": [],
    "references": [
        "[1] Chatbots guide - https://example.com/chatbots",
        "[2] Email AI - https://example.org/email-ai",
    ],
}


def make_search():
    return FakeTavily({
        "ai chatbots small business": tavily_response("ai chatbots small business", [
            {"url": "https://example.com/chatbots", "title": "Chatbots guide", "content": "Chatbots answer FAQs."},
        ]),
        "ai email marketing small business": tavily_response("ai email marketing small business", [
            {"url": "https://example.org/email-ai", "title": "Email AI", "content": "AI subject lines."},
        ]),
    })


def make_llm():
    return FakeLLM({"AnswerQuestion": [DRAFT], "ReviseAnswer": [REVISION_1, REVISION_2]})


def test_graph_runs_end_to_end_with_grounded_answer(capsys):
    llm, search = make_llm(), make_search()
    messages = build_graph(llm=llm, search_tool=search, max_iterations=2).invoke("How can small businesses use AI?")

    assert search.calls == ["ai chatbots small business", "ai email marketing small business"]
    # The reviser actually received the search evidence
    first_revise_input = llm.seen_inputs["ReviseAnswer"][0]
    tool_msgs = [m for m in first_revise_input if isinstance(m, ToolMessage)]
    assert "https://example.com/chatbots" in tool_msgs[0].content

    result = final_result(messages)
    assert result["revisions"] == 2
    assert result["answer"] == REVISION_2["answer"]
    assert result["references"] == REVISION_2["references"]
    assert result["failed_queries"] == {}
    assert {s["url"] for s in result["sources"]} == {"https://example.com/chatbots", "https://example.org/email-ai"}

    print_final(messages)
    out = capsys.readouterr().out
    assert "=== Final Answer ===" in out and "https://example.org/email-ai" in out


def test_ungrounded_reference_is_dropped_inside_the_loop():
    messages = build_graph(llm=make_llm(), search_tool=make_search(), max_iterations=1).invoke("q")
    result = final_result(messages)
    assert result["references"] == ["[1] Chatbots guide - https://example.com/chatbots"]
    assert "[2]" not in result["answer"]
    assert result["dropped_references"] == ["[2] Made-up study - https://not-retrieved.example.net/study"]


def test_failed_searches_give_no_credit_and_no_references():
    messages = build_graph(llm=make_llm(), search_tool=FakeTavily(fail=True), max_iterations=1).invoke("q")
    result = final_result(messages)
    assert result["references"] == []
    assert set(result["failed_queries"]) == {"ai chatbots small business"}
    payload = json.loads(next(m for m in messages if isinstance(m, ToolMessage)).content)
    assert "error" in payload["ai chatbots small business"]
    # Only length credit is left: no grounded refs, no valid citations, no searches
    assert result["score"] <= 30


def test_compiled_graph_keeps_no_state_between_runs():
    app = build_graph(llm=make_llm(), search_tool=make_search(), max_iterations=2)
    first = final_result(app.invoke("q"))
    second = final_result(app.invoke("q"))
    # A global best score would end the second run after one revision
    assert first["revisions"] == second["revisions"] == 2


def test_cli_exits_cleanly_without_keys(monkeypatch):
    import pytest

    from reflexion_agent import main

    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "TAVILY_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    with pytest.raises(SystemExit) as exc:
        main(["question"])
    assert "TAVILY_API_KEY" in str(exc.value)
