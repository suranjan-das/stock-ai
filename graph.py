import os
import pandas as pd
import json
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
    prompt_answer_messages,
    prompt_route_all,
    prompt_handle_other
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
    follow_up: Optional[bool] = None   # None until determined; then True/False
    symbol: Optional[str] = None       # Stock symbol, e.g., 'HDFCBANK'
    stock_info: Optional[Dict] = None  # Populated after fetching stock data
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
                "messages": [RemoveMessage(id=m.id) for m in state["messages"][:-1]]
            }
        return {"follow_up": False, "non_stock": False}

    else:  # "other"
        return {
            "non_stock": True,
            "follow_up": False,
            "messages": [RemoveMessage(id=m.id) for m in state["messages"][:-1]]
        }


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

# Node to fetch stock data
def get_stock_info(state: GraphState) -> dict:
    if state["symbol"] == "NO_MATCH":
        return {"stock_info": {"data": "No data available"}}  # Response already set in extract_symbol
    try:
        stock_symbol = state["symbol"]
        stock = yf.Ticker(f"{stock_symbol}.NS")
        info = stock.info
        # Read the CSV that contains the keys to keep
        imp_keys__path = "./important_keys.csv"
        keys_df = pd.read_csv(imp_keys__path, header=None)
        keys_to_keep = set(keys_df.iloc[0].dropna().tolist())
        # Filter dictionary
        filtered_info = {k: v for k, v in info.items() if k in keys_to_keep}
    except Exception as e:
        filtered_info = {"data": "No data available"}
    return {"stock_info": filtered_info}

def route_query(state: GraphState) -> Literal["follow_up", "new", "other"]:
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

    # Handle single-stock queries (follow_up or new)
    chain = prompt_answer_messages | llmg
    formatted_info = json.dumps(state.get("stock_info", {}), indent=2)
    result = chain.invoke({
        "messages": state["messages"][-1],
        "stock_info": formatted_info
    })

    return {"response": result}

# Build the graph
workflow = StateGraph(GraphState)

workflow.add_node("extract_symbol", extract_symbol)
workflow.add_node("get_stock_info", get_stock_info)
workflow.add_node("answer_user_messages", answer_user_messages)
workflow.add_node("classify_query", classify_query)

workflow.add_edge(START, "classify_query")
workflow.add_conditional_edges(
    "classify_query",
    route_query,
    {
        "follow_up": "answer_user_messages",
        "new": "extract_symbol",
        "other": "answer_user_messages",
    }
)

workflow.add_edge("extract_symbol", "get_stock_info")
workflow.add_edge("get_stock_info", "answer_user_messages")
workflow.add_edge("answer_user_messages", END)

# Configure memory
checkpointer = MemorySaver()
# Compile the graph with checkpointer
app = workflow.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}
node_to_stream = 'answer_user_messages'

async def chat_loop():
    while True:
        user_input = input("\nYou: ")
        if user_input.lower().strip() in {"exit", "quit"}:
            print("Exiting chat...")
            break

        input_message = HumanMessage(content=user_input)

        print("AI: ", end="", flush=True)

        async for event in app.astream_events({"messages": [input_message]}, config, version="v2"):
            if event["event"] == "on_chat_model_stream" and event['metadata'].get('langgraph_node', '') == node_to_stream:
                data = event["data"]
                print(data["chunk"].content, end="", flush=True)
        print()  # newline after full response

async def main():
    await chat_loop()

if __name__ == "__main__":
    asyncio.run(main())