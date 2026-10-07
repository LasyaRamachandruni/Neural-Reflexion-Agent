# reflexion_agent.py
"""Reflexion loop: draft -> search -> revise (-> search -> revise ...).

Run from the command line:
    python reflexion_agent.py "Your question here" --max-iterations 2
"""
import argparse
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.graph import END, MessageGraph

from chains import build_chains
from execute_tools import make_execute_tools, parse_tool_message
from grounding import ground_answer, retrieved_sources
from scoring import evaluate_answer, latest_search_outcome, successful_queries

MAX_ITERATIONS = 2  # max search+revise passes (at least one always runs)
DEFAULT_PROMPT = "Write about how small business can leverage AI to grow"
META_KEY = "reflexion"  # where per-revision grounding/score info is stored


def _ground_and_score(msg: AIMessage, state: List[BaseMessage]) -> AIMessage:
    """Apply the grounding check to a ReviseAnswer call and attach its score."""
    allowed = set(retrieved_sources(state))
    ok_queries = successful_queries(latest_search_outcome(state))

    new_calls = []
    meta: Dict[str, Any] = {}
    for tc in msg.tool_calls:
        if tc.get("name") != "ReviseAnswer" or meta:
            new_calls.append(tc)
            continue
        args = dict(tc.get("args") or {})
        result = ground_answer(args.get("answer"), args.get("references"), allowed)
        args["answer"] = result.answer
        args["references"] = result.references
        new_calls.append({**tc, "args": args})
        meta = {
            "score": evaluate_answer(result.answer, result.references, ok_queries),
            "dropped_references": result.dropped,
            "successful_queries": ok_queries,
        }

    if not meta:
        return msg
    additional_kwargs = {**msg.additional_kwargs, META_KEY: meta}
    additional_kwargs.pop("function_call", None)  # tool_calls is the source of truth
    return msg.model_copy(update={"tool_calls": new_calls, "additional_kwargs": additional_kwargs})


def revision_messages(messages: List[BaseMessage]) -> List[AIMessage]:
    """AI messages produced by the revisor node in this run, oldest first."""
    return [m for m in messages if isinstance(m, AIMessage) and META_KEY in m.additional_kwargs]


def _has_search_queries(msg: AIMessage) -> bool:
    return any((tc.get("args") or {}).get("search_queries") for tc in msg.tool_calls)


def build_graph(llm=None, search_tool=None, max_iterations: int = MAX_ITERATIONS):
    """Compile the Reflexion graph. Pass fakes for llm/search_tool in tests."""
    first_responder_chain, revisor_chain = build_chains(llm)

    def revisor(state: List[BaseMessage]) -> AIMessage:
        return _ground_and_score(revisor_chain.invoke(state), state)

    def event_loop(state: List[BaseMessage]) -> str:
        # Everything is derived from the message list, so nothing leaks
        # between runs even when the compiled app is reused.
        if sum(isinstance(m, ToolMessage) for m in state) >= max_iterations:
            return END
        revisions = revision_messages(state)
        if revisions:
            scores = [m.additional_kwargs[META_KEY]["score"] for m in revisions]
            if len(scores) > 1 and scores[-1] <= max(scores[:-1]):
                return END  # no improvement over the best revision so far
            if not _has_search_queries(revisions[-1]):
                return END
        return "execute_tools"

    graph = MessageGraph()
    graph.add_node("draft", first_responder_chain)
    graph.add_node("execute_tools", make_execute_tools(search_tool))
    graph.add_node("revisor", revisor)
    graph.add_edge("draft", "execute_tools")
    graph.add_edge("execute_tools", "revisor")
    graph.add_conditional_edges("revisor", event_loop, {"execute_tools": "execute_tools", END: END})
    graph.set_entry_point("draft")
    return graph.compile()


def final_result(messages: List[BaseMessage]) -> Dict[str, Any]:
    """Best-scoring grounded revision (latest wins ties), plus run details."""
    revisions = revision_messages(messages)
    best: Optional[AIMessage] = None
    for m in revisions:
        if best is None or m.additional_kwargs[META_KEY]["score"] >= best.additional_kwargs[META_KEY]["score"]:
            best = m

    answer, refs, meta = None, [], {}
    if best is not None:
        tc = next(tc for tc in best.tool_calls if tc.get("name") == "ReviseAnswer")
        answer = tc["args"].get("answer")
        refs = tc["args"].get("references") or []
        meta = best.additional_kwargs[META_KEY]

    queries: List[str] = []
    failed: Dict[str, str] = {}
    for m in messages:
        if isinstance(m, ToolMessage):
            for q, outcome in parse_tool_message(m).items():
                queries.append(q)
                if isinstance(outcome, dict) and outcome.get("error"):
                    failed[q] = outcome["error"]

    return {
        "answer": answer,
        "references": refs,
        "score": meta.get("score"),
        "dropped_references": meta.get("dropped_references", []),
        "queries": queries,
        "failed_queries": failed,
        "sources": list(retrieved_sources(messages).values()),
        "revisions": len(revisions),
    }


def run(prompt: str, llm=None, search_tool=None, max_iterations: int = MAX_ITERATIONS) -> List[BaseMessage]:
    app = build_graph(llm=llm, search_tool=search_tool, max_iterations=max_iterations)
    return app.invoke(prompt)


def print_final(messages: List[BaseMessage]) -> None:
    result = final_result(messages)
    if not result["answer"]:
        print("No revised answer was produced.")
        return
    print("\n=== Final Answer ===\n")
    print(result["answer"])
    if result["references"]:
        print("\nReferences:")
        for r in result["references"]:
            print(r)
    print(f"\nScore: {result['score']}  (revisions: {result['revisions']})")
    print(f"Searches: {len(result['queries']) - len(result['failed_queries'])}/{len(result['queries'])} returned results")
    for q, err in result["failed_queries"].items():
        print(f"  failed: {q!r}: {err}")
    if result["dropped_references"]:
        print("\nRemoved (not in search results):")
        for r in result["dropped_references"]:
            print(f"  - {r}")


def main(argv: Optional[List[str]] = None) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the Reflexion agent once from the command line.")
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS,
                        help="maximum number of search passes (default: %(default)s)")
    parser.add_argument("--show-graph", action="store_true", help="print the graph as Mermaid and exit")
    args = parser.parse_args(argv)

    app = build_graph(max_iterations=args.max_iterations)
    if args.show_graph:
        print(app.get_graph().draw_mermaid())
        return
    print_final(app.invoke(args.prompt))


if __name__ == "__main__":
    main()
