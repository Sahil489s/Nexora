import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
import certifi

# --------------------------------------------------
# Environment
# --------------------------------------------------

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


# --------------------------------------------------
# LangChain / LangGraph
# --------------------------------------------------

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from tools import tools


# --------------------------------------------------
# Data directory
# --------------------------------------------------

Path("data").mkdir(exist_ok=True)


# --------------------------------------------------
# Gemini model configuration
# --------------------------------------------------

# Gemini 3.8 Flash is the current stable model.
DEFAULT_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)


# Models allowed from the frontend.
#
# Gemini 2.5 models are intentionally removed because
# your API key currently does not have access to them.
#
# Gemini 3.8 Flash is the primary model.
ALLOWED_MODELS = {
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
}


# --------------------------------------------------
# System prompt
# --------------------------------------------------

SYSTEM_PROMPT = """
You are a helpful Agentic AI assistant named ShaggyGPT developed by Sahil Sharma, similar to ChatGPT.

You can:

1. Answer normal questions.
2. Use tools when needed.
3. Search uploaded documents using the RAG tool.
4. Search the web for latest/current information using Tavily Search.
5. Remember important user information using the memory tool.
6. Recall memory when useful.
7. Use calculator for math.

Rules:

- If the user asks about latest news, current events, recent updates,
  today's information, current prices, current people, current versions,
  new releases, or anything time-sensitive, use Tavily Search.

- If the user asks about an uploaded document,
  use search_uploaded_documents.

- If the user asks you to remember something,
  use remember_this.

- If the user asks about previous preferences or saved facts,
  use recall_memory.

- Use calculator for math questions.

- When using web search, summarize the results clearly and mention
  that the answer is based on web search results.

- Be clear, helpful, accurate, and concise.
"""


# --------------------------------------------------
# Model validation
# --------------------------------------------------

def normalize_model_name(model_name: str | None) -> str:
    """
    Validate the model selected by the frontend.

    If the model is missing or unsupported,
    Gemini 3.8 Flash is used.
    """

    if not model_name:
        return DEFAULT_MODEL

    model_name = model_name.strip()

    if model_name not in ALLOWED_MODELS:
        return DEFAULT_MODEL

    return model_name


# --------------------------------------------------
# Build LangGraph Agent
# --------------------------------------------------

def build_agent(model_name: str):
    """
    Build a LangGraph agent using the selected Gemini model.
    """

    selected_model = normalize_model_name(model_name)

    print(f"[BappyGPT] Loading Gemini model: {selected_model}")

    # --------------------------------------------------
    # Gemini LLM
    # --------------------------------------------------
    #
    # IMPORTANT:
    # Do NOT use temperature/top_p/top_k with Gemini 3.8.
    #
    # Gemini 3.8 uses thinking levels instead.
    #
    # LangChain's ChatGoogleGenerativeAI integration handles
    # the Gemini API communication.
    # --------------------------------------------------

    llm_kwargs = {
        "model": selected_model,
        "streaming": True,
    }

    # Gemini 3.x supports thinking_level.
    #
    # medium gives a balanced amount of reasoning.
    if selected_model.startswith("gemini-3."):
        llm_kwargs["thinking_level"] = "medium"

    llm = ChatGoogleGenerativeAI(**llm_kwargs)

    # --------------------------------------------------
    # Bind tools
    # --------------------------------------------------

    llm_with_tools = llm.bind_tools(tools)

    # --------------------------------------------------
    # Chatbot node
    # --------------------------------------------------

    def chatbot_node(state: MessagesState):

        messages = [
            SystemMessage(content=SYSTEM_PROMPT)
        ] + state["messages"]

        response = llm_with_tools.invoke(messages)

        return {
            "messages": [response]
        }

    # --------------------------------------------------
    # Tool node
    # --------------------------------------------------

    tool_node = ToolNode(tools)

    # --------------------------------------------------
    # LangGraph workflow
    # --------------------------------------------------

    workflow = StateGraph(MessagesState)

    workflow.add_node(
        "chatbot",
        chatbot_node
    )

    workflow.add_node(
        "tools",
        tool_node
    )

    workflow.add_edge(
        START,
        "chatbot"
    )

    workflow.add_conditional_edges(
        "chatbot",
        tools_condition
    )

    workflow.add_edge(
        "tools",
        "chatbot"
    )

    # --------------------------------------------------
    # SQLite checkpoint database
    # --------------------------------------------------

    conn = sqlite3.connect(
        "data/langgraph_checkpoints.sqlite",
        check_same_thread=False
    )

    checkpointer = SqliteSaver(conn)

    # --------------------------------------------------
    # Compile agent
    # --------------------------------------------------

    return workflow.compile(
        checkpointer=checkpointer
    )


# --------------------------------------------------
# Agent cache
# --------------------------------------------------

_AGENT_CACHE = {}


def get_agent(model_name: str | None = None):
    """
    Return a cached LangGraph agent.

    If the agent doesn't already exist,
    create it and cache it.
    """

    selected_model = normalize_model_name(model_name)

    if selected_model not in _AGENT_CACHE:

        _AGENT_CACHE[selected_model] = build_agent(
            selected_model
        )

    return _AGENT_CACHE[selected_model]