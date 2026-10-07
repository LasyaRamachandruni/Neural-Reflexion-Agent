import pytest

from scoring import evaluate_answer, successful_queries

ANSWER = " ".join(["word"] * 240) + " [1] [2]"
REFS = ["[1] A - https://example.com/a", "[2] B - https://example.com/b"]


def test_failed_searches_get_no_credit():
    outcome = {"q1": {"error": "no results"}, "q2": {"error": "timeout"}}
    assert successful_queries(outcome) == []
    assert evaluate_answer(ANSWER, REFS, successful_queries(outcome)) == evaluate_answer(ANSWER, REFS, [])


def test_successful_searches_add_credit():
    outcome = {"q1": {"results": [{"url": "https://example.com/a"}]}, "q2": {"error": "timeout"}}
    assert successful_queries(outcome) == ["q1"]
    assert evaluate_answer(ANSWER, REFS, ["q1"]) == pytest.approx(evaluate_answer(ANSWER, REFS, []) + 10)


def test_citations_without_matching_reference_do_not_count():
    with_refs = evaluate_answer(ANSWER, REFS, [])
    without_refs = evaluate_answer(ANSWER, [], [])
    # loses 2 refs * 5 and 2 citations * 4
    assert with_refs - without_refs == pytest.approx(18)


def test_empty_answer_scores_zero():
    assert evaluate_answer("", REFS, ["q"]) == 0.0
