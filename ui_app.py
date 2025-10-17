# ui_app.py
# Streamlit UI for the Reflexion Agent
# ------------------------------------
# Features:
# - Prompt input & Run
# - Sidebar settings: max iterations, model override (optional)
# - Live status during run
# - Final Answer + References
# - Deduplicated Sources panel
# - Run history and side-by-side comparison
# - Export answer as Markdown or full run as JSON

import os
import io
import json
import re
from typing import List, Tuple, Optional

import streamlit as st
from dotenv import load_dotenv

# Load .env before importing anything that needs keys
load_dotenv()

# LangGraph / LangChain
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, MessageGraph

# Your project modules
from chains import revisor_chain, first_responder_chain
from execute_tools import execute_tools, SEEN_URLS

# ---------------------------
# Helpers (mirroring your agent)
# ---------------------------
def evaluate_answer(ans: str, refs: Optional[List[str]], queries_used: Optional[List[str]]) -> float:
    score = 0.0
    if not ans:
        return score
    words = len(ans.split())
    score += max(0, 30 - abs(words - 250) * 0.1)  # peak near 250 words
    if refs:
        score += min(20, 5 * len(refs))
    cites = len(re.findall(r"\[\d+\]", ans))
    score += min(20, cites * 4)
    if queries_used:
        score += min(30, 10 * min(3, len(queries_used)))
    return round(score, 2)

def extract_last_tool_answer(messages: List[BaseMessage],
                             names: Tuple[str, ...] = ("ReviseAnswer", "AnswerQuestion")) -> Tuple[Optional[str], List[str]]:
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
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            try:
                data = json.loads(msg.content)
                return list(data.keys())
            except Exception:
                return []
    return []

def build_graph(max_iterations: int) -> MessageGraph:
    graph = MessageGraph()
    graph.add_node("draft", first_responder_chain)
    graph.add_node("execute_tools", execute_tools)
    graph.add_node("revisor", revisor_chain)
    graph.add_edge("draft", "execute_tools")
    graph.add_edge("execute_tools", "revisor")

    best = {"val": -1.0}

    def event_loop(state: List[BaseMessage]) -> str:
        # stop after N tool messages
        count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
        if count_tool_visits > max_iterations:
            return END
        ans, refs = extract_last_tool_answer(state)
        queries_used = extract_latest_queries_from_tools(state)
        if ans:
            cur = evaluate_answer(ans, refs, queries_used)
            if cur <= best["val"]:
                return END
            best["val"] = cur
        return "execute_tools"

    graph.add_conditional_edges("revisor", event_loop)
    graph.set_entry_point("draft")
    return graph

def export_markdown(prompt: str, answer: str, refs: List[str]) -> bytes:
    lines = [f"# Final Answer\n\n**Prompt:** {prompt}\n\n{answer}\n"]
    if refs and "References" not in (answer or ""):
        lines.append("\n## References\n")
        for r in refs:
            lines.append(f"- {r}")
        lines.append("\n")
    return "\n".join(lines).encode("utf-8")

def export_run_json(prompt: str, messages: List[BaseMessage]) -> bytes:
    # messages are not directly serializable; capture key fields
    serial = []
    for m in messages:
        entry = {"type": m.__class__.__name__}
        entry["content"] = getattr(m, "content", None)
        entry["tool_calls"] = getattr(m, "tool_calls", None)
        serial.append(entry)
    blob = {
        "prompt": prompt,
        "messages": serial,
        "sources": sorted(SEEN_URLS),
    }
    return json.dumps(blob, indent=2).encode("utf-8")

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Neural Reflexion Agent", page_icon="🧠", layout="wide")
st.title("🧠 Neural Reflexion Agent")
st.caption("LangGraph + Gemini + Tavily · Reflexive, evidence-backed answers")

# Session state for run history
if "runs" not in st.session_state:
    st.session_state["runs"] = []  # list of dicts {prompt, answer, refs, queries, messages, sources}

with st.sidebar:
    st.header("Settings")
    max_iters = st.slider("Max refinement iterations", 0, 6, 2)
    st.markdown("**Environment keys loaded**")
    st.write("GOOGLE_API_KEY:", bool(os.getenv("GOOGLE_API_KEY")))
    st.write("TAVILY_API_KEY:", bool(os.getenv("TAVILY_API_KEY")))
    st.divider()
    if st.button("🧼 Clear deduped sources"):
        SEEN_URLS.clear()
        st.toast("Cleared sources", icon="🧽")

prompt = st.text_area("Enter your prompt", value="Write about how small business can leverage AI to grow", height=120)
run = st.button("▶️ Run Reflexion", type="primary")

if run and prompt.strip():
    graph = build_graph(max_iters)
    app = graph.compile()
    with st.status("Running Reflexion loop…", expanded=True) as status:
        st.write("• Building initial draft")
        messages = app.invoke(prompt)
        st.write("• Searching + Revising")
        ans, refs = extract_last_tool_answer(messages)
        queries = extract_latest_queries_from_tools(messages)
        status.update(label="Reflexion complete", state="complete")

    # store run
    st.session_state["runs"].append({
        "prompt": prompt,
        "answer": ans,
        "refs": refs,
        "queries": queries,
        "messages": messages,
        "sources": sorted(SEEN_URLS),
    })

# ---------------------------
# Display latest run
# ---------------------------
if st.session_state["runs"]:
    latest = st.session_state["runs"][-1]
    left, right = st.columns([2, 1], gap="large")

    with left:
        st.subheader("Final Answer")
        if latest["answer"]:
            st.markdown(latest["answer"])
        else:
            st.info("No answer produced.")
        if latest["refs"] and "References" not in (latest["answer"] or ""):
            st.markdown("### References")
            for r in latest["refs"]:
                st.markdown(f"- {r}")

        # export buttons
        col_md, col_json = st.columns(2)
        with col_md:
            md_bytes = export_markdown(latest["prompt"], latest["answer"] or "", latest["refs"] or [])
            st.download_button("⬇️ Download Markdown", data=md_bytes, file_name="reflexion_answer.md", mime="text/markdown")
        with col_json:
            json_bytes = export_run_json(latest["prompt"], latest["messages"])
            st.download_button("⬇️ Download Full Run (JSON)", data=json_bytes, file_name="reflexion_run.json", mime="application/json")

    with right:
        st.subheader("Run Summary")
        st.write(f"Max iterations: **{max_iters}**")
        st.write(f"Queries used: **{len(latest['queries'])}**")
        if latest["queries"]:
            with st.expander("Search queries"):
                for q in latest["queries"]:
                    st.code(q)

        if latest["sources"]:
            st.subheader("Sources (dedup)")
            for u in latest["sources"]:
                st.write(f"- {u}")

    st.divider()
    st.subheader("Compare Runs")
    if len(st.session_state["runs"]) >= 2:
        idxs = list(range(len(st.session_state["runs"])))
        c1, c2 = st.columns(2)
        a = c1.selectbox("Left run", idxs, index=len(idxs)-2, key="cmp_left")
        b = c2.selectbox("Right run", idxs, index=len(idxs)-1, key="cmp_right")
        if a != b:
            ra = st.session_state["runs"][a]
            rb = st.session_state["runs"][b]
            lcol, rcol = st.columns(2)
            with lcol:
                st.markdown(f"**Run {a} — Prompt**")
                st.code(ra["prompt"])
                st.markdown("**Answer**")
                st.markdown(ra["answer"] or "_no answer_")
            with rcol:
                st.markdown(f"**Run {b} — Prompt**")
                st.code(rb["prompt"])
                st.markdown("**Answer**")
                st.markdown(rb["answer"] or "_no answer_")
        else:
            st.info("Pick two different runs to compare.")
else:
    st.info("Enter a prompt and click **Run Reflexion** to get started.")
