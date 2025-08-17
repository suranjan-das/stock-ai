import os
import pandas as pd
import json
from functools import lru_cache
import asyncio
from dotenv import load_dotenv
from fuzzywuzzy import process

import yfinance as yf

from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.messages import RemoveMessage
from langchain.prompts import PromptTemplate

from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import MemorySaver

from typing import TypedDict, Literal, Optional, Dict, Union

from chat_template import (
    prompt_extract_symbol,
    prompt_answer_messages_wo_news,
    prompt_answer_messages,
    prompt_route_all,
    prompt_handle_other,
    prompt_news_relevance
)

# Always resolve relative to this file's directory
dotenv_path = ".env"
load_dotenv(dotenv_path)

# Constants
FUZZY_LIMIT = 5
LOW_CONFIDENCE_THRESHOLD = 50  # Minimum score to consider
HIGH_CONFIDENCE_THRESHOLD = 90  # If score ≥ 90, skip LLM

# initialize model for large number of input tokens
llmg = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
# Initialize the LLM
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

# Load CSV of NSE companies
df_path = "./data/EQUITY_L.csv"
df = pd.read_csv(df_path)  # Assumes columns: "Company Name", "Symbol"
company_names = df["NAME OF COMPANY"].tolist()



class GraphState(MessagesState):
    non_stock: bool = False            # True if query is not stock related
    follow_up: Optional[bool] = False   # None until determined; then True/False
    symbol: Optional[str] = None       # Stock symbol, e.g., 'HDFCBANK'
    stock_info: Optional[Dict] = None  # Populated after fetching stock data
    news_required: bool = False
    news_data: Optional[str] = None
    response: Optional[str] = None     # Model's textual reply

def classify_query(state: GraphState) -> dict:
    chain = prompt_route_all | llmg
    result = chain.invoke({"messages": state["messages"]}).content.strip()

    if result == "follow_up":
        return {"follow_up": True, "non_stock": False}

    elif result == "new":
        if len(state["messages"]) > 1:
            # keep only the latest message
            return {
                "follow_up": False,
                "non_stock": False,
                "news_required": False,
                "news_data": None,
                "stock_info": None,
                "messages": [RemoveMessage(id=m.id) for m in state["messages"][:-1]]
            }
        return {"follow_up": False, "non_stock": False}

    else:  # "other"
        return {
            "non_stock": True,
            "follow_up": False,
            "messages": [RemoveMessage(id=m.id) for m in state["messages"][:-1]]
        }

def news_required(state: GraphState) -> dict:
    chain = prompt_news_relevance | llmg
    result = chain.invoke({"messages": state["messages"]}).content.strip()
    if result == "relevant":
        return {"news_required": True}
    else:
        return {"news_required": False}

# Node to extract company symbol for single stock queries
def extract_symbol(state: GraphState) -> dict:
    symbol = "NO_MATCH"
    chain = prompt_extract_symbol | llm
    extracted_name = chain.invoke({"messages": state["messages"]}).content.strip()
    # Fuzzy matching
    matches = process.extract(extracted_name, company_names, limit=FUZZY_LIMIT)
    best_match, best_score = matches[0]
    # High-confidence match
    if best_score >= HIGH_CONFIDENCE_THRESHOLD:
        symbol = df[df["NAME OF COMPANY"] == best_match]["SYMBOL"].values[0]
    
    return {"symbol": symbol}

# Cache important keys CSV so it’s loaded only once
@lru_cache(maxsize=1)
def load_keys_to_keep(csv_path="./important_keys.csv"):
    try:
        keys_df = pd.read_csv(csv_path, header=None)
        return set(keys_df.iloc[0].dropna().tolist())
    except Exception:
        return set()

# function to get news data
def get_stock_news_info(stock) -> str:
    news_string = ""    
    try:
        news = stock.get_news(count=5)
        for article in news:
            news_string += f"Publication Date: {article['content']['pubDate'][:10]}\nsummary:  {article['content']['summary']}\n"
    except Exception as e:
        print(f"error getting news: {e}")
    return news_string

def get_stock_info(state: GraphState) -> GraphState:
    """
    Fetch stock information and optionally recent news for the given symbol.
    Returns a dictionary with 'stock_info' and 'news_data'.
    """

    # Handle missing/invalid symbol
    if state.get("symbol") == "NO_MATCH":
        return {"stock_info": {"data": "No data available"}, "news_data": []}

    stock_symbol = state["symbol"]
    try:
        stock = yf.Ticker(f"{stock_symbol}.NS")
    except Exception:
        return {"stock_info": {"data": "No data available"}, "news_data": []}

    result = {"news_data": state.get("news_data", [])}

    # Handle follow-up queries
    if state.get("follow_up", False):
        if state.get("news_required", False) and not result["news_data"]:
            result["news_data"] = get_stock_news_info(stock)
        return result

    # Handle new queries → fetch structured stock info
    try:
        keys_to_keep = load_keys_to_keep()
        info = stock.info
        filtered_info = {k: v for k, v in info.items() if k in keys_to_keep}
        if not filtered_info:
            filtered_info = {"data": "No data available"}
    except Exception:
        filtered_info = {"data": "No data available"}

    result["stock_info"] = filtered_info

    # Fetch news only if required and not already present
    if state.get("news_required", False) and not result["news_data"]:
        result["news_data"] = get_stock_news_info(stock)
    return result

def route_query(state: GraphState) -> Literal["new", "follow_up", "other"]:
    if state.get("non_stock"):
        return "other"
    elif state.get("follow_up"):
        return "follow_up"
    else:
        return "new"

def answer_user_messages(state: GraphState) -> Optional[Union[dict, GraphState]]:
    # Handle "other" type queries (non-stock, greetings, multi-stock)
    if state.get("non_stock", False):
        chain = prompt_handle_other | llmg
        result = chain.invoke({"message": state["messages"][-1]})
        return {"response": result}

    result = None
    formatted_info = json.dumps(state.get("stock_info", {}), indent=2)
    # Handle single-stock queries (follow_up or new)
    if state.get("news_required", False):
        chain = prompt_answer_messages | llmg
        result = chain.invoke({
            "messages": state["messages"][-1],
            "stock_info": formatted_info,
            "news_required": False,
            "news_data": state["news_data"]
        })
    else:
        chain = prompt_answer_messages_wo_news | llmg
        result = chain.invoke({
            "messages": state["messages"][-1],
            "stock_info": formatted_info,
        })    

    return {"response": result}


# Build the graph
workflow = StateGraph(GraphState)

workflow.add_node("extract_symbol", extract_symbol)
workflow.add_node("get_stock_info", get_stock_info)
workflow.add_node("answer_user_messages", answer_user_messages)
workflow.add_node("classify_query", classify_query)
workflow.add_node("news_required", news_required)

workflow.add_edge(START, "classify_query")
workflow.add_conditional_edges(
    "classify_query",
    route_query,
    {
        "new": "extract_symbol",
        "follow_up": "news_required",
        "other": "answer_user_messages"
    }
)
workflow.add_edge("extract_symbol", "news_required")
workflow.add_edge("news_required", "get_stock_info")
workflow.add_edge("get_stock_info", "answer_user_messages")
workflow.add_edge("answer_user_messages", END)

# Configure memory
checkpointer = MemorySaver()
# Compile the graph with checkpointer
app = workflow.compile(checkpointer=checkpointer)