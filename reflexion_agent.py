# reflexion_agent.py
from typing import List, Tuple, Optional
import json
import re
from dotenv import load_dotenv

# Load .env BEFORE importing chains/tools that need env vars
load_dotenv()

from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, HumanMessage
from langgraph.graph import END, MessageGraph

from chains import revisor_chain, first_responder_chain
from execute_tools import execute_tools, SEEN_URLS

# --------------------
# Graph setup
# --------------------
graph = MessageGraph()
MAX_ITERATIONS = 2  # hard cap to avoid endless loops

graph.add_node("draft", first_responder_chain)
graph.add_node("execute_tools", execute_tools)
graph.add_node("revisor", revisor_chain)

graph.add_edge("draft", "execute_tools")
graph.add_edge("execute_tools", "revisor")

# --------------------
# Helpers: scoring & extraction
# --------------------
def evaluate_answer(ans: str, refs: Optional[List[str]], queries_used: Optional[List[str]]) -> float:
    """Simple heuristic reward to measure revision quality."""
    score = 0.0
    if not ans:
        return score

    # 1) Concision around ~250 words
    words = len(ans.split())
    score += max(0, 30 - abs(words - 250) * 0.1)  # peak at 250, down as it deviates

    # 2) References present & count
    if refs:
        score += min(20, 5 * len(refs))  # up to +20

    # 3) Inline numeric citations like [1], [2]
    cites = len(re.findall(r"\[\d+\]", ans))
    score += min(20, cites * 4)  # up to +20

    # 4) Query coverage (how many queries were used)
    if queries_used:
        score += min(30, 10 * min(3, len(queries_used)))  # up to +30

    return round(score, 2)

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
                ans = args.get("answer")
                refs = args.get("references", []) or []
                return ans, refs
    return None, []

def extract_latest_queries_from_tools(messages: List[BaseMessage]) -> List[str]:
    """Read last ToolMessage JSON and return query keys."""
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            try:
                data = json.loads(msg.content)
                return list(data.keys())
            except Exception:
                return []
    return []

# --------------------
# Iteration control with reward-based early stop
# --------------------
_best_score = -1.0  # module-level holder

def event_loop(state: List[BaseMessage]) -> str:
    global _best_score

    # Stop if we've already run enough tool passes
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    if count_tool_visits > MAX_ITERATIONS:
        return END

    # Compute reward for the latest answer
    ans, refs = extract_last_tool_answer(state)
    queries_used = extract_latest_queries_from_tools(state)

    if ans:
        cur = evaluate_answer(ans, refs, queries_used)
        # Early stop if no improvement
        if cur <= _best_score:
            return END
        _best_score = cur

    return "execute_tools"

graph.add_conditional_edges("revisor", event_loop)
graph.set_entry_point("draft")

app = graph.compile()

# --------------------
# Visualize graph (Mermaid)
# --------------------
print(app.get_graph().draw_mermaid())

# --------------------
# Run an example
# --------------------
response = app.invoke(
    "Write about how small business can leverage AI to grow"
)

def print_final(messages):
    ans, refs = extract_last_tool_answer(messages)
    if ans:
        print("\n=== Final Answer ===\n")
        print(ans)
        # If the answer already contains "References", don't print refs again
        if "References" not in ans and refs:
            print("\nReferences:")
            for r in refs:
                print(r)
    # Optionally show unique sources gathered by the tool
    # (handy for debugging, but you can comment this out for a cleaner demo)
    # if SEEN_URLS:
    #     print("\nAll unique sources seen (deduplicated):")
    #     for u in SEEN_URLS:
    #         print(u)
def evaluate_answer(ans, refs, queries_used):
    score = 0.0
    # ...existing scoring...
    # Penalize if all refs are non-authoritative (very rough heuristic)
    if refs and not any(("mckinsey.com" in r or "mailchimp.com" in r) for r in refs):
        score -= 10
    return round(score, 2)

