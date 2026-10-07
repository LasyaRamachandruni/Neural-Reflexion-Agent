from pathlib import Path

import pytest

import reflexion_agent
from tests.test_graph import make_llm, make_search

st_testing = pytest.importorskip("streamlit.testing.v1")
UI_APP = str(Path(__file__).resolve().parent.parent / "ui_app.py")


def test_ui_runs_with_fakes(monkeypatch):
    real_build = reflexion_agent.build_graph
    monkeypatch.setattr(
        reflexion_agent, "build_graph",
        lambda max_iterations=2: real_build(llm=make_llm(), search_tool=make_search(), max_iterations=max_iterations),
    )
    at = st_testing.AppTest.from_file(UI_APP, default_timeout=30)
    at.run()
    at.button[0].click().run()
    assert not at.exception
    shown = [m.value for m in at.markdown]
    assert "- [2] Email AI - https://example.org/email-ai" in shown
