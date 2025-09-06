import json
import re
import pandas as pd
from os import getenv
from dotenv import load_dotenv
from functools import lru_cache
from typing import Literal, Optional, Dict
from typing_extensions import Annotated
from operator import add

import streamlit as st
import yfinance as yf
from rapidfuzz import process, fuzz

# LangChain & LangGraph
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_tavily import TavilySearch

# Local utilities
from yfutil import load_info, load_news, extract_all_statements
from chat_template import (
    prompt_screen_input,
    prompt_web_search,
    prompt_answer_query,
    instruction_template,
)


# --------------------------
# Global State
# --------------------------
STOCK_SYMBOL = ""
STOCK_NAME = ""
STOCK_INFO = {}
STOCK_NEWS = ""
STOCK_STATEMENTS = None


# --------------------------
# Environment & Config
# --------------------------
dotenv_path = ".env"
load_dotenv(dotenv_path)

# Large context model
llm = ChatOpenAI(
    api_key=getenv("OPENROUTER_API_KEY"),
    base_url=getenv("OPENROUTER_BASE_URL"),
    model="gpt-4.1-mini",
)

# Web search (Tavily)
tavily_search = TavilySearch(
    max_results=2,
    search_depth="basic",
)

# Equity list (CSV)
equity_df = pd.read_csv("data/EQUITY_L.csv")  # Columns: SYMBOL, NAME OF COMPANY


# --------------------------
# Utilities
# --------------------------
def update_data():
    """Update global stock variables from session state ticker."""
    global STOCK_SYMBOL, STOCK_NAME, STOCK_INFO, STOCK_NEWS, STOCK_STATEMENTS

    STOCK_SYMBOL = st.session_state.ticker
    STOCK_NAME = equity_df[equity_df['SYMBOL'] == STOCK_SYMBOL[:-3]]['NAME OF COMPANY'].values[0]

    STOCK_INFO = load_info(STOCK_SYMBOL)
    STOCK_NEWS = load_news(STOCK_SYMBOL)
    STOCK_STATEMENTS = extract_all_statements(STOCK_SYMBOL)


def merge_dicts(current: Dict, update: Dict) -> Dict:
    """
    Merge two dictionaries, with `update` taking precedence.
    (Shallow merge – customize if deep merge needed)
    """
    new_dict = current.copy()
    new_dict.update(update)
    return new_dict


# --------------------------
# Graph State
# --------------------------
class GraphState(MessagesState):
    symbol: str = None
    name: str = None
    
    # External info flags
    web_search: bool = False
    news_required: bool = False
    financials_required: bool = False
    
    # Web search query (if needed)
    search_query: str | None = None

    # Aggregated data store
    data: Annotated[Dict, merge_dicts]



# --------------------------
# Nodes
# --------------------------
def screen_input(state: GraphState) -> Literal["update_state", "answer_query"]:
    """Classify input as finance-related or not."""
    chain = prompt_screen_input | llm
    result = chain.invoke({
        "query": state["messages"][-1],
        "stock_name": STOCK_NAME,
    }).content.strip()

    return "update_state" if result == "finance" else "answer_query"


def update_state(state: GraphState) -> GraphState:
    """Update stock info in the state if symbol changed."""
    if state.get("symbol") is None or state["symbol"] != STOCK_SYMBOL:
        state["symbol"] = STOCK_SYMBOL
        state["name"] = STOCK_NAME
        state["data"]["stock_info"] = STOCK_INFO
        state["data"]["stock_news"] = STOCK_NEWS
        state["data"]["stock_statements"] = STOCK_STATEMENTS

        # Keep only latest user message
        state["messages"] = [RemoveMessage(id=m.id) for m in state["messages"][:-1]]
        return state

    # Keep last 4 messages for context
    if len(state["messages"]) > 4:
        state["messages"] = [RemoveMessage(id=m.id) for m in state["messages"][:-4]]
    return state


# --------------------------
# Search Decision
# --------------------------
from pydantic import BaseModel, Field

class InfoDecision(BaseModel):
    """Decision on whether external info is required for answering the query."""
    
    web_search: str = Field(
        description="Whether web search is required. Must be 'required' or 'not_required'",
        pattern="^(required|not_required)$"
    )
    news: str = Field(
        description="Whether company news is required. Must be 'required' or 'not_required'",
        pattern="^(required|not_required)$"
    )
    financial_statement: str = Field(
        description="Whether financial statements are required. Must be 'required' or 'not_required'",
        pattern="^(required|not_required)$"
    )
    search_query: Optional[str] = Field(
        description="Reframed search query if web_search = 'required'",
        default=None
    )


def decide_external_info(state: GraphState) -> dict:
    """Decide which external information (web, news, financials) is needed for the query."""
    system_prompt = PromptTemplate(
        input_variables=["stock_name"],
        template=instruction_template,
    )
    instruction = SystemMessage(content=system_prompt.format(stock_name=state["name"]))

    structured_llm = llm.with_structured_output(InfoDecision)
    decision = structured_llm.invoke([instruction] + [state["messages"][-1]])

    return {
        "web_search": decision.web_search == "required",
        "news_required": decision.news == "required",
        "financials_required": decision.financial_statement == "required",
        "search_query": decision.search_query,
    }



def route_search(state: GraphState) -> Literal["web_search", "answer_query"]:
    return "web_search" if state["web_search"] else "answer_query"


def web_search(state: GraphState) -> dict:
    """Fetch and store web search results in state."""
    search_docs = tavily_search.invoke(state["search_query"])

    formatted_docs = "\n\n---\n\n".join(
        f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
        for doc in search_docs["results"]
    )

    state["data"]["web_info"] = formatted_docs
    return state


def answer_query(state: GraphState) -> GraphState:
    """Generate final answer only with required context to save tokens."""
    chain = prompt_answer_query | llm
    
    # Selectively include context
    news_data = state["data"]["stock_news"] if state.get("news_required") else "Not required"
    financials = (
        json.dumps(state["data"]["stock_statements"], indent=2)
        if state.get("financials_required")
        else "Not required"
    )
    web_results = state["data"]["web_info"] if state.get("web_search") else "Not required"

    result = chain.invoke({
        "stock_name": state.get("name", "Unknown"),
        "messages": state["messages"][-1],
        "context": state["messages"][:-1],
        "stock_parameters": json.dumps(state["data"].get("stock_info", "No info"), indent=2),
        "news_data": news_data,
        "web_results": web_results,
        "financial_statements": financials,
    })

    return {"messages": result}



# --------------------------
# Graph Workflow
# --------------------------
workflow = StateGraph(GraphState)

workflow.add_node("update_state", update_state)
workflow.add_node("answer_query", answer_query)
workflow.add_node("decide_external_info", decide_external_info)
workflow.add_node("web_search", web_search)

workflow.add_conditional_edges(START, screen_input)
workflow.add_edge("update_state", "decide_external_info")
workflow.add_conditional_edges("decide_external_info", route_search)
workflow.add_edge("web_search", "answer_query")
workflow.add_edge("answer_query", END)

# --------------------------
# Compile App
# --------------------------
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)
