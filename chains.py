# chains.py
import datetime
import os
from typing import Tuple

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable

from schema import AnswerQuestion, ReviseAnswer

DEFAULT_MODEL = "gemini-2.5-pro"

# --- Base prompt for both actors ---
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert AI researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. After the reflection, **list 1-3 search queries separately** for researching improvements. Do not include them inside the reflection.
""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(time=lambda: datetime.datetime.now().isoformat())

# --- First responder prompt ---
first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer"
)

revise_instructions = """Revise your previous answer using the new information.
- Max 250 words. Do not exceed.
- Include inline numeric citations like [1], [2] that map to the "references" field.
- The search results are in the tool messages as JSON ({query: {"results": [{title, url, content}]}}).
  Only cite URLs that appear there. Never invent a source or a URL.
- Write each reference as "[n] Title - URL" using the exact URL from the search results.
- If the searches returned nothing useful, give fewer references rather than making them up.
- Do not put a References section inside the answer text; use the "references" field.
- Keep a professional, actionable tone.
"""

revisor_prompt_template = actor_prompt_template.partial(first_instruction=revise_instructions)


def get_api_key() -> str:
    """Gemini key from GEMINI_API_KEY or GOOGLE_API_KEY."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY in your environment/.env")
    return key


def get_default_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
        google_api_key=get_api_key(),
    )


def build_chains(llm=None) -> Tuple[Runnable, Runnable]:
    """Return (first_responder_chain, revisor_chain) for the given chat model.

    The model only needs to support .bind_tools(tools=..., tool_choice=...).
    """
    if llm is None:
        llm = get_default_llm()
    first_responder_chain = first_responder_prompt_template | llm.bind_tools(
        tools=[AnswerQuestion], tool_choice="AnswerQuestion"
    )
    revisor_chain = revisor_prompt_template | llm.bind_tools(
        tools=[ReviseAnswer], tool_choice="ReviseAnswer"
    )
    return first_responder_chain, revisor_chain


__all__ = ["build_chains", "get_default_llm", "get_api_key"]
