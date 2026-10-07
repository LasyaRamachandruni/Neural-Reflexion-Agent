import json

from langchain_core.messages import ToolMessage

from grounding import ground_answer, normalize_url, retrieved_sources

ALLOWED = {normalize_url("https://example.com/a"), normalize_url("https://example.org/b")}


def test_ungrounded_reference_is_removed_and_citations_renumbered():
    answer = "Chatbots cut costs [1]. A made-up stat [2]. Email tools help [3]. Combined [1, 2, 3]."
    refs = [
        "[1] Real A - https://example.com/a",
        "[2] Invented - https://invented.example.net/report",
        "[3] Real B - https://www.example.org/b/",
    ]
    out = ground_answer(answer, refs, ALLOWED)
    assert out.references == ["[1] Real A - https://example.com/a", "[2] Real B - https://www.example.org/b/"]
    assert out.dropped == ["[2] Invented - https://invented.example.net/report"]
    assert out.answer == "Chatbots cut costs [1]. A made-up stat. Email tools help [2]. Combined [1][2]."


def test_reference_without_url_is_dropped():
    out = ground_answer("Claim [1].", ["[1] McKinsey: State of AI 2023"], ALLOWED)
    assert out.references == []
    assert out.answer == "Claim."
    assert out.dropped == ["[1] McKinsey: State of AI 2023"]


def test_inline_url_and_references_block_in_answer_are_removed():
    answer = "See https://fake.example.io/x for more.\n\nReferences:\n[1] https://fake.example.io/y"
    out = ground_answer(answer, [], ALLOWED)
    assert "fake.example.io" not in out.answer
    assert "References" not in out.answer
    assert "https://fake.example.io/x" in out.dropped


def test_retrieved_sources_reads_only_successful_results():
    msg = ToolMessage(tool_call_id="c1", content=json.dumps({
        "q1": {"results": [{"title": "A", "url": "https://example.com/a", "content": ""}]},
        "q2": {"error": "no results"},
    }))
    assert list(retrieved_sources([msg])) == [normalize_url("https://example.com/a")]
