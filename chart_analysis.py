# chart_analysis.py
import os
import io
import tempfile
from dotenv import load_dotenv
import plotly.graph_objects as go
import plotly.io as pio

pio.kaleido.scope.default_format = "png"
pio.kaleido.scope.default_width = 1200
pio.kaleido.scope.default_height = 800


from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Load env variables
load_dotenv(".env")

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
)

def fig_to_png_bytes(fig):
    """Convert Plotly figure to PNG bytes fully headless (Kaleido)."""
    buf = io.BytesIO()
    fig.write_image(buf, format="png", engine="kaleido")  # Force Kaleido
    buf.seek(0)
    return buf.read()

def analyze_chart(fig: go.Figure, ticker: str, timeframe: str = None, stream_handler=None):
    timeframe_text = f"Timeframe: {timeframe}. " if timeframe else ""

    # Convert figure to PNG bytes headless
    image_bytes = fig_to_png_bytes(fig)

    # Construct prompt
    analysis_prompt = (
        f"You are a Stock Trader specializing in Technical Analysis at a top financial institution. "
        f"Analyze the stock chart for {ticker} based on its candlestick chart and displayed indicators. "
        f"{timeframe_text}"
        f"Provide a detailed explanation of patterns, signals, and trends visible, "
        f"without giving buy/sell recommendations. Keep it concise (~150 words)."
    )

    instruction = SystemMessage(content=analysis_prompt)
    message = HumanMessage(content=[{
        "type": "media",
        "data": image_bytes,
        "mime_type": "image/png"
    }])

    # Streaming mode
    if stream_handler:
        final_output = ""
        for event in llm.stream([instruction, message]):
            if event.type == "token":
                token = event.data
                final_output += token
                stream_handler(token)
        return final_output
    else:
        response = llm.invoke([instruction, message])
        return response.content

