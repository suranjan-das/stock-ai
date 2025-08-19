from IPython.display import Image, display
import pandas as pd
import json
from functools import lru_cache
from dotenv import load_dotenv
import re
from rapidfuzz import process, fuzz
import yfinance as yf

from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage
from langchain_core.messages import RemoveMessage
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import MemorySaver

from typing import TypedDict, Literal, Optional, List, Dict, Union
from typing_extensions import Annotated
from operator import add

from chat_template import (
    prompt_classify_input_query,
    prompt_extract_symbol,
    prompt_disambiguate,
    prompt_handle_other,
    prompt_answer_query
)

# Always resolve relative to this file's directory
dotenv_path = ".env"
load_dotenv(dotenv_path)

# initialize model for large number of input tokens
llmg = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

# Load CSV of NSE companies
df_path = "./data/EQUITY_L.csv"
df = pd.read_csv(df_path)  # Assumes columns: "Company Name", "Symbol"
company_names = df["NAME OF COMPANY"].tolist()

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
    query_type: Literal["new", "follow_up", "other"]
    symbol: str = None       # Stock symbol, e.g., 'HDFCBANK'
    news_required: bool = False
    data: Annotated[Dict, merge_dicts]  # merge dict updates
    response: str = None     # Model's textual reply

def classify_query(state: GraphState) -> dict:
    chain = prompt_classify_input_query | llmg
    result = chain.invoke({"messages": state["messages"]}).content.strip()

    if result == "new":
        return {"messages": [RemoveMessage(id=m.id) for m in state["messages"][:-1]],
                "query_type": "new",
                "symbol": None,
                "data": {}
               }
    elif result == "follow_up":
        return {"query_type": "follow_up"}
    else:
        return {"query_type": "other"}

def handle_follow_up_query(state: GraphState):
    # for future use
    return

def route_query(state: GraphState) -> Literal["new", "follow_up", "other"]:
    return state["query_type"]

def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # remove punctuation
    text = re.sub(r'\b(ltd|limited|inc|corp|company|co)\b', '', text)  # remove suffixes
    return re.sub(r'\s+', ' ', text).strip()

def extract_symbol(state: GraphState) -> dict:
    query = state["messages"][-1]  # user query
    chain = prompt_extract_symbol | llmg
    extracted_name = chain.invoke({"messages": state["messages"]}).content.strip()

    # Normalize
    extracted_norm = normalize(extracted_name)

    # Try direct symbol match first
    if extracted_norm.upper() in df["SYMBOL"].values:
        return {"symbol": extracted_norm.upper()}

    # Exact match on company name
    for name in df["NAME OF COMPANY"].values:
        if normalize(name) == extracted_norm:
            symbol = df[df["NAME OF COMPANY"] == name]["SYMBOL"].values[0]
            return {"symbol": symbol}

    # Fuzzy match (RapidFuzz)
    matches = process.extract(
        extracted_norm, 
        [normalize(n) for n in company_names], 
        scorer=fuzz.token_sort_ratio,
        limit=3
    )

    best_match, best_score, idx = matches[0]

    if best_score >= 90:  # high confidence
        symbol = df.iloc[idx]["SYMBOL"]
        return {"symbol": symbol}

    elif best_score >= 75:  # medium confidence → ask LLM
        candidates = [df.iloc[m[2]]["NAME OF COMPANY"] for m in matches]
        disambig_chain = prompt_disambiguate | llmg
        final_choice = disambig_chain.invoke({"user_query": extracted_name, "candidates": candidates})
        # pick symbol of chosen company
        chosen = final_choice.content.strip()
        symbol = df[df["NAME OF COMPANY"] == chosen]["SYMBOL"].values[0]
        return {"symbol": symbol}

    return {"symbol": "NO_MATCH"}

# Cache important keys CSV so it’s loaded only once
@lru_cache(maxsize=1)
def load_keys_to_keep(csv_path="./important_keys.csv"):
    try:
        keys_df = pd.read_csv(csv_path, header=None)
        return set(keys_df.iloc[0].dropna().tolist())
    except Exception:
        return set()

def get_stock_info(state: GraphState) -> dict:
    # Handle missing/invalid symbol
    if state.get("symbol") == "NO_MATCH":
        return {"data": {"stock_info": "No data available for this stock."}}

    stock_symbol = state["symbol"]
    try:
        stock = yf.Ticker(f"{stock_symbol}.NS")
    except Exception:
        return {"data": {"stock_info": "No data available for this stock."}}

     # Handle new queries → fetch structured stock info
    try:
        keys_to_keep = load_keys_to_keep()
        info = stock.info
        filtered_info = {k: v for k, v in info.items() if k in keys_to_keep}
    except Exception:
        filtered_info = {"data": {"stock_info": "No data available for this stock."}}

    return {"data": {"stock_info": filtered_info}}

def get_stock_news(state: GraphState) -> dict:
    news_string = ""
    stock_symbol = state["symbol"]
    try:
        stock = yf.Ticker(f"{stock_symbol}.NS")
        news = stock.get_news(count=3)
        for article in news:
            news_string += f"Publication Date: {article['content']['pubDate'][:10]}\nsummary:  {article['content']['summary']}\n"
    except Exception as e:
        print(f"error getting news: {e}")
    return {"data": {"stock_news": news_string}}

def answer_query(state: GraphState) -> dict:
    if state.get("query_type") == "other":
        chain = prompt_handle_other | llmg
        result = chain.invoke({"messages": state["messages"][-1]})
        return {"response": result}
    # define the llm chain
    chain = prompt_answer_query | llmg
    # get stock parameters
    if state["data"]["stock_info"]: 
        formatted_info = json.dumps(state["data"]["stock_info"], indent=2)
    else:
        formatted_info = "No stock info available!"
    # Get news data    
    if state["data"]["stock_news"]:
        news_info = state["data"]["stock_news"]
    else:
        news_info = "Sorry. No news data available!!"
    result = chain.invoke({
            "messages": state["messages"][-1],
            "stock_parameters": formatted_info,
            "news_data": news_info
        })
    return {"response": result}


# Build the graph
workflow = StateGraph(GraphState)

workflow.add_node("classify_query", classify_query)
workflow.add_node("handle_follow_up_query", handle_follow_up_query)
workflow.add_node("extract_symbol", extract_symbol)
workflow.add_node("get_stock_info", get_stock_info)
workflow.add_node("get_stock_news", get_stock_news)
workflow.add_node("answer_query", answer_query)

workflow.add_edge(START, "classify_query")
workflow.add_conditional_edges(
    "classify_query",
    route_query,
    {
        "new": "extract_symbol",
        "follow_up": "handle_follow_up_query",
        "other": "answer_query"
    }
)
workflow.add_edge("extract_symbol", "get_stock_info")
workflow.add_edge("extract_symbol", "get_stock_news")
workflow.add_edge("get_stock_info", "answer_query")
workflow.add_edge("get_stock_news", "answer_query")
workflow.add_edge("handle_follow_up_query", "answer_query")
workflow.add_edge("answer_query", END)

# Configure memory
checkpointer = MemorySaver()
# Compile the graph with checkpointer
app = workflow.compile(checkpointer=checkpointer)