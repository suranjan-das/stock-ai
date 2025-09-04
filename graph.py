import pandas as pd
import json
from functools import lru_cache
from dotenv import load_dotenv
from os import getenv
import re
from rapidfuzz import process, fuzz
import yfinance as yf

from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages import RemoveMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_tavily import TavilySearch

from typing import Literal, Optional, Dict
from pydantic import BaseModel, Field
from typing_extensions import Annotated
from operator import add

import streamlit as st
from yfutil import load_info, load_news
from chat_template import prompt_screen_input, prompt_web_search, prompt_answer_query, instruction_template

STOCK_SYMBOL = ""
STOCK_NAME = ""
STOCK_INFO = {}
STOCK_NEWS = ""

# Always resolve relative to this file's directory
dotenv_path = ".env"
load_dotenv(dotenv_path)

# initialize model for large number of input tokens
llm = llm = ChatOpenAI(
  api_key=getenv("OPENROUTER_API_KEY"),
  base_url=getenv("OPENROUTER_BASE_URL"),
  model="gpt-4o",
)

# initialize tavily search
tavily_search = TavilySearch(
    max_results=2,
    search_depth="advanced",
    # time_range="day",
    include_domains=['finance'],
)

# Load your CSV
equity_df = pd.read_csv("data/EQUITY_L.csv")  # Columns: SYMBOL, NAME OF COMPANY

def update_data():
    # Update the global stock variables
    global STOCK_SYMBOL, STOCK_NAME, STOCK_INFO, STOCK_NEWS
    STOCK_SYMBOL = st.session_state.ticker
    STOCK_NAME = equity_df[equity_df['SYMBOL'] == STOCK_SYMBOL[:-3]]['NAME OF COMPANY'].values[0]
    STOCK_INFO = load_info(STOCK_SYMBOL)
    STOCK_NEWS = load_news(STOCK_SYMBOL)

# Custom reducer function for merging dictionaries
def merge_dicts(current: Dict, update: Dict) -> Dict:
    """
    Merge two dictionaries, with update taking precedence for overlapping keys.
    You can customize this logic (e.g., deep merge, append to lists, etc.).
    """
    new_dict = current.copy()  # Create a copy to avoid mutating the original
    new_dict.update(update)    # Merge update into current
    return new_dict

class GraphState(MessagesState):
    symbol: str = None
    name: str = None
    web_search: bool = False
    search_query: str
    data: Annotated[Dict, merge_dicts]

def screen_input(state: GraphState) -> Literal["update_state", "answer_query"]:
    # define the llm chain
    chain = prompt_screen_input | llm
    result = chain.invoke({"query": state["messages"][-1],
                            "stock_name": STOCK_NAME}).content.strip()
    if result == 'finance':
        return "update_state"
    return "answer_query"

def update_state(state: GraphState) -> GraphState:
    if state.get("symbol", None) is None or state["symbol"] != STOCK_SYMBOL:
        state["symbol"] = STOCK_SYMBOL
        state["name"] = STOCK_NAME
        state["data"]["stock_info"] = STOCK_INFO
        state["data"]["stock_news"] = STOCK_NEWS
        state["messages"] = [RemoveMessage(id=m.id) for m in state["messages"][:-1]]
        return state
    return state

class SearchDecision(BaseModel):
    """Decision on whether a web search is required and the query to use"""
    
    decision: str = Field(
        description="Whether a web search is required. Must be 'required' or 'not_required'",
        pattern="^(required|not_required)$"
    )
    search_query: str | None = Field(
        description="The reframed search query for Tavily, included only if decision is 'required'",
        default=None
    )

def need_web_search(state: GraphState) -> dict:
    # Create a PromptTemplate
    system_prompt = PromptTemplate(input_variables=["stock_name"], template=instruction_template)    
    # Example: Pass the user_query parameter from LangGraph state
    instruction = SystemMessage(content=system_prompt.format(stock_name=state["name"]))
    # Search query
    structured_llm = llm.with_structured_output(SearchDecision)
    search_query = structured_llm.invoke([instruction]+[state["messages"][-1]])
    
    if search_query.decision == "required":
        return {"web_search": True, "search_query": search_query.search_query}
    return {"web_search": False, "search_query": None}

def route_search(state: GraphState) -> Literal["web_search", "answer_query"]:
    if state["web_search"]:
        return "web_search"
    return "answer_query"

def web_search(state: GraphState) -> dict:
    # Search
    search_docs = tavily_search.invoke(state["search_query"])

     # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs["results"]
        ]
    )
    state["data"]["web_info"] = formatted_search_docs

    return state

def answer_query(state: GraphState) -> GraphState:
    # define the llm chain
    chain = prompt_answer_query | llm
    result = chain.invoke({
            "stock_name": state["name"] if state.get("name", None) else "Unknown",
            "messages": state["messages"][-1],
            "context": state["messages"][:-1],
            "stock_parameters": state["data"]["stock_info"] if state.get("data", {}).get("stock_info", None) else "No information available",
            "news_data": state["data"]["stock_news"] if state.get("data", {}).get("stock_news", None) else "No information available",
            "web_results": state["data"]["web_info"]  if state.get("data", {}).get("web_info", None) else "No information available",
        })
    return {"messages": result}

# Build the graph
workflow = StateGraph(GraphState)

workflow.add_node("update_state", update_state)
workflow.add_node("answer_query", answer_query)
workflow.add_node("need_web_search", need_web_search)
workflow.add_node("web_search", web_search)

workflow.add_conditional_edges(START, screen_input)
workflow.add_edge("update_state", "need_web_search")
workflow.add_conditional_edges("need_web_search", route_search)
workflow.add_edge("web_search", "answer_query")
workflow.add_edge("answer_query", END)

# Configure memory
checkpointer = MemorySaver()
# Compile the graph with checkpointer
app = workflow.compile(checkpointer=checkpointer)