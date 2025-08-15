import sys
import os
import asyncio
import streamlit as st
from langchain.schema import HumanMessage

# ---------- Add project root to sys.path so we can import graph.py ----------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your LangGraph app
from graph import app  # assuming your LangGraph object is named `app`

# ---------- Streamlit Config ----------
st.set_page_config(page_title="📈 Stock Analysis Chatbot", page_icon="📊")
st.title("📈 Stock Analysis Chatbot")

# ---------- Session State ----------
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user", "content": ...}, ...]

# ---------- Chat History UI ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- Streaming Logic ----------
async def run_graph_stream(user_message: str):
    """
    Runs the LangGraph app in streaming mode and yields chunks of text.
    """
    config = {"configurable": {"thread_id": "abc123"}}  # customize as needed
    node_to_stream = 'answer_user_messages'

    async for event in app.astream_events(
        {"messages": [HumanMessage(content=user_message)]},
        config,
        version="v2"
    ):
        if event["event"] == "on_chat_model_stream" and event['metadata'].get('langgraph_node', '') == node_to_stream:
            chunk = event["data"]["chunk"].content
            if chunk:
                yield chunk

# ---------- Chat Input ----------
if user_input := st.chat_input("Ask me about a stock..."):
    # Append user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Prepare assistant streaming
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        async def process_stream():
            full = ""
            async for chunk in run_graph_stream(user_input):
                full += chunk
                placeholder.markdown(full)
            return full

        full_response = asyncio.run(process_stream())

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": full_response})
