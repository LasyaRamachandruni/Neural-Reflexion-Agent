# ui_app.py
# Streamlit UI for the Reflexion Agent
# ------------------------------------
# - Prompt input & Run
# - Sidebar: max search/revise passes, which API keys are loaded
# - Final answer + grounded references, score, failed searches,
#   and any references that were removed because they weren't in the results
# - Sources returned by Tavily in this run
# - Run history and side-by-side comparison
# - Export answer as Markdown or full run as JSON

import json
import os
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage

# Load .env before building anything that needs keys
load_dotenv()

from reflexion_agent import MAX_ITERATIONS, build_graph, final_result  # noqa: E402


def export_markdown(prompt: str, answer: str, refs: List[str]) -> bytes:
    lines = [f"# Final Answer\n\n**Prompt:** {prompt}\n\n{answer}\n"]
    if refs:
        lines.append("\n## References\n")
        for r in refs:
            lines.append(f"- {r}")
        lines.append("\n")
    return "\n".join(lines).encode("utf-8")


def export_run_json(prompt: str, messages: List[BaseMessage], result: Dict[str, Any]) -> bytes:
    # messages are not directly serializable; capture key fields
    serial = []
    for m in messages:
        serial.append({
            "type": m.__class__.__name__,
            "content": getattr(m, "content", None),
            "tool_calls": getattr(m, "tool_calls", None),
        })
    blob = {"prompt": prompt, "result": result, "messages": serial}
    return json.dumps(blob, indent=2, default=str).encode("utf-8")


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Neural Reflexion Agent", page_icon="🧠", layout="wide")
st.title("🧠 Neural Reflexion Agent")
st.caption("LangGraph + Gemini + Tavily · draft → search → revise, with citations checked against search results")

if "runs" not in st.session_state:
    st.session_state["runs"] = []  # list of dicts {prompt, result, messages, max_iters}

with st.sidebar:
    st.header("Settings")
    max_iters = st.slider("Max search + revise passes", 1, 4, MAX_ITERATIONS)
    st.markdown("**Environment keys loaded**")
    st.write("GEMINI_API_KEY / GOOGLE_API_KEY:", bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")))
    st.write("TAVILY_API_KEY:", bool(os.getenv("TAVILY_API_KEY")))

prompt = st.text_area("Enter your prompt", value="Write about how small business can leverage AI to grow", height=120)
run = st.button("▶️ Run Reflexion", type="primary")

if run and prompt.strip():
    try:
        app = build_graph(max_iterations=max_iters)
        with st.status("Running Reflexion loop…", expanded=True) as status:
            st.write("• Drafting, searching and revising")
            messages = app.invoke(prompt)
            status.update(label="Reflexion complete", state="complete")
    except Exception as e:  # missing keys, API errors
        st.error(f"Run failed: {e}")
    else:
        st.session_state["runs"].append({
            "prompt": prompt,
            "result": final_result(messages),
            "messages": messages,
            "max_iters": max_iters,
        })

# ---------------------------
# Display latest run
# ---------------------------
if st.session_state["runs"]:
    latest = st.session_state["runs"][-1]
    res = latest["result"]
    left, right = st.columns([2, 1], gap="large")

    with left:
        st.subheader("Final Answer")
        if res["answer"]:
            st.markdown(res["answer"])
        else:
            st.info("No answer produced.")
        if res["references"]:
            st.markdown("### References")
            for r in res["references"]:
                st.markdown(f"- {r}")
        if res["dropped_references"]:
            with st.expander(f"Removed {len(res['dropped_references'])} citation(s) not found in search results"):
                for r in res["dropped_references"]:
                    st.markdown(f"- {r}")

        col_md, col_json = st.columns(2)
        with col_md:
            md_bytes = export_markdown(latest["prompt"], res["answer"] or "", res["references"])
            st.download_button("⬇️ Download Markdown", data=md_bytes, file_name="reflexion_answer.md", mime="text/markdown")
        with col_json:
            json_bytes = export_run_json(latest["prompt"], latest["messages"], res)
            st.download_button("⬇️ Download Full Run (JSON)", data=json_bytes, file_name="reflexion_run.json", mime="application/json")

    with right:
        st.subheader("Run Summary")
        st.write(f"Max passes: **{latest['max_iters']}** · Revisions: **{res['revisions']}**")
        st.write(f"Heuristic score: **{res['score']}**")
        ok = len(res["queries"]) - len(res["failed_queries"])
        st.write(f"Searches with results: **{ok}/{len(res['queries'])}**")
        if res["queries"]:
            with st.expander("Search queries"):
                for q in res["queries"]:
                    err = res["failed_queries"].get(q)
                    st.code(f"{q}\n  -> failed: {err}" if err else q)

        if res["sources"]:
            st.subheader("Sources retrieved")
            for s in res["sources"]:
                st.markdown(f"- [{s.get('title') or s['url']}]({s['url']})")

    st.divider()
    st.subheader("Compare Runs")
    if len(st.session_state["runs"]) >= 2:
        idxs = list(range(len(st.session_state["runs"])))
        c1, c2 = st.columns(2)
        a = c1.selectbox("Left run", idxs, index=len(idxs) - 2, key="cmp_left")
        b = c2.selectbox("Right run", idxs, index=len(idxs) - 1, key="cmp_right")
        if a != b:
            ra = st.session_state["runs"][a]
            rb = st.session_state["runs"][b]
            lcol, rcol = st.columns(2)
            for col, idx, r in ((lcol, a, ra), (rcol, b, rb)):
                with col:
                    st.markdown(f"**Run {idx} — Prompt**")
                    st.code(r["prompt"])
                    st.markdown(f"**Answer** (score {r['result']['score']})")
                    st.markdown(r["result"]["answer"] or "_no answer_")
        else:
            st.info("Pick two different runs to compare.")
else:
    st.info("Enter a prompt and click **Run Reflexion** to get started.")
