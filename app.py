import os
import sys
import asyncio
from typing import Dict
import numpy as np
import pandas as pd
import streamlit as st
from langchain.schema import HumanMessage

from yfutil import get_stock_data, stock_chart

# ---------- Add project root to sys.path so we can import graph.py ----------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your LangGraph app
from temp_graph import app  # assuming your LangGraph object is named `app`
from temp_graph import update_data

# Custom CSS to adjust the top padding of the main content area
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem; # Adjust this value as needed
        }
    </style>
    """, unsafe_allow_html=True)

st.set_page_config(page_title="Stock Analyst", layout="wide")

DEFAULT_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "SBIN.NS", "TATAMOTORS.NS",
    "INFY.NS", "ICICIBANK.NS", "ITC.NS", "LT.NS", "MARUTI.NS",
]

# ----------------- Render chart with tabs -----------------
def render_chart_tabs(ticker, key_suffix=""):
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["1D", "1WK", "1M", "6M", "1Y", "5Y", "MAX"])

    with tab1:
        df = get_stock_data(ticker, period="1d")
        if df is not None:
            st.plotly_chart(stock_chart(df, ticker), use_container_width=True, key=f"{ticker}_1d_{key_suffix}")
        else:
            st.warning("No data found.")

    with tab2:
        df = get_stock_data(ticker, period="1wk")
        if df is not None:
            st.plotly_chart(stock_chart(df, ticker), use_container_width=True, key=f"{ticker}_1wk_{key_suffix}")
        else:
            st.warning("No data found.")

    with tab3:
        df = get_stock_data(ticker, period="1mo")
        if df is not None:
            st.plotly_chart(stock_chart(df, ticker), use_container_width=True, key=f"{ticker}_1m_{key_suffix}")
        else:
            st.warning("No data found.")

    with tab4:
        df = get_stock_data(ticker, period="6mo")
        if df is not None:
            st.plotly_chart(stock_chart(df, ticker), use_container_width=True, key=f"{ticker}_6mo_{key_suffix}")
        else:
            st.warning("No data found.")

    with tab5:
        df = get_stock_data(ticker, period="1y")
        if df is not None:
            st.plotly_chart(stock_chart(df, ticker), use_container_width=True, key=f"{ticker}_1y_{key_suffix}")
        else:
            st.warning("No data found.")

    with tab6:
        df = get_stock_data(ticker, period="5y")
        if df is not None:
            st.plotly_chart(stock_chart(df, ticker), use_container_width=True, key=f"{ticker}_5y_{key_suffix}")
        else:
            st.warning("No data found.")

    with tab7:
        df = get_stock_data(ticker, period="max")
        if df is not None:
            st.plotly_chart(stock_chart(df, ticker), use_container_width=True, key=f"{ticker}_max_{key_suffix}")
        else:
            st.warning("No data found.")


# ---------- Session State ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_ticker" not in st.session_state:
    st.session_state.last_ticker = None

# Load your CSV
equity_df = pd.read_csv("data/EQUITY_L.csv")  # Columns: SYMBOL, NAME OF COMPANY

# Build a mapping { "NAME OF COMPANY (SYMBOL)": "SYMBOL" }
options = {f"{row['NAME OF COMPANY']}": row['SYMBOL'] for _, row in equity_df.iterrows()}

# Sidebar
with st.sidebar:
    st.header("⚙️ Controls")

    # searchable by both company and ticker
    selected_display = st.selectbox(
        "Select a company",
        options=list(options.keys()),
        index=list(options.values()).index("MARUTI") if "MARUTI" in options.values() else 0
    )

    # Actual ticker you will use downstream
    ticker = f"{options[selected_display]}.NS"
    st.session_state.ticker = ticker

    st.divider()
    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = []

# ---------- Reset when ticker changes ----------
if st.session_state.last_ticker != ticker:
    st.session_state.messages = []
    st.session_state.last_ticker = ticker
    # Update global state in temp_graph
    update_data()


# ---------- Inject chart as first assistant message ----------
if not any(msg.get("type") == "chart" and msg.get("symbol") == ticker for msg in st.session_state.messages):
    st.session_state.messages.insert(0, {"role": "assistant", "type": "chart", "symbol": ticker})


# ---------- Chat History UI ----------
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg.get("type") == "chart":
            render_chart_tabs(msg["symbol"], key_suffix=str(i))
        else:
            st.markdown(msg["content"])


# ---------- Streaming Logic ----------
async def run_graph_stream(user_message: str):
    config = {"configurable": {"thread_id": "abc123"}}
    node_to_stream = "answer_query"

    async for event in app.astream_events(
        {"messages": [HumanMessage(content=user_message)]},
        config,
        version="v2"
    ):
        if event["event"] == "on_chat_model_stream" and event["metadata"].get("langgraph_node", "") == node_to_stream:
            chunk = event["data"]["chunk"].content
            if chunk:
                yield chunk


# ---------- Chat Input ----------
if user_input := st.chat_input("Ask me about a stock..."):
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()

        async def process_stream():
            full = ""
            async for chunk in run_graph_stream(user_input):
                full += chunk
                placeholder.markdown(full)
            return full

        full_response = asyncio.run(process_stream())

    st.session_state.messages.append({"role": "assistant", "content": full_response})
