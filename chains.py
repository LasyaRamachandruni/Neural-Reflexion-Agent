# chains.py
import os
import datetime
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers.openai_tools import (
    PydanticToolsParser,
    JsonOutputToolsParser,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from schema import AnswerQuestion, ReviseAnswer

# --- Auth (Gemini) ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY in environment/.env")

# --- Parsers (optional, useful if you later validate) ---
pydantic_parser = PydanticToolsParser(tools=[AnswerQuestion])
parser = JsonOutputToolsParser(return_id=True)

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

# --- LLM ---
from langchain_google_genai import ChatGoogleGenerativeAI
import os
llm = ChatGoogleGenerativeAI(
    # pick ONE of these; start with flash-latest:
    model="gemini-2.5-pro",
    # model="gemini-1.5-pro-latest",
    # model="gemini-1.5-pro-002",
    # model="gemini-1.0-pro",          # fallback for older accounts
    api_key=os.getenv("GOOGLE_API_KEY"),
)


# --- Chains ---
first_responder_chain = (
    first_responder_prompt_template
    | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
)

revise_instructions = """Revise your previous answer using the new information.
- Max 250 words. Do not exceed.
- Include inline numeric citations like [1], [2] that map to a "References" list.
- Provide 3–6 references that support specific claims; prefer sources (<= 3 years).
- Avoid generic claims without a citation.
- Keep a professional, actionable tone.
"""


revisor_chain = (
    actor_prompt_template.partial(first_instruction=revise_instructions)
    | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")
)

__all__ = ["first_responder_chain", "revisor_chain"]


