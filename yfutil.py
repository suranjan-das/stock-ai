import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from functools import lru_cache

# ---------------------- Helpers -----------------------------
PERIOD_MAP = {
    "1d": ("1d", "2m"),
    "1wk": ("1wk", "15m"),
    "1mo": ("1mo", "30m"),
    "6mo": ("6mo", "4h"),
    "1y": ("1y", "1d"),
    "5y": ("5y", "1wk"),
    "max": ("max", "1mo"),
}

@lru_cache(maxsize=None)
def get_stock_data(ticker, period="5d"):
    # Map period to (period, interval)
    yf_period, interval = PERIOD_MAP.get(period, ("6mo", "1h"))

    try:
        df = yf.download(ticker, period=yf_period, interval=interval, auto_adjust=False)
        if df.empty:
            return None

        # --- Indicators ---
        close = df["Close"][ticker]
        df["SMA20"] = close.rolling(window=20, min_periods=1).mean()
        df["SMA50"] = close.rolling(window=50, min_periods=1).mean()

        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))

        # Bollinger Bands (20-period SMA ± 2 std dev)
        sma20 = df["SMA20"]
        rolling_std = close.rolling(window=20, min_periods=1).std()
        df["BB_upper"] = sma20 + (rolling_std * 2)
        df["BB_lower"] = sma20 - (rolling_std * 2)

        # Format index for readability
        if interval in ["2m", "15m", "30m", "1h", "4h"]:  
            # intraday → show full datetime
            df.index = df.index.strftime("%d %b %H:%M")
        elif interval in ["1d", "1wk"]:  
            # daily/weekly → show date
            df.index = df.index.strftime("%d %b %Y")
        elif interval in ["1mo"]:  
            # monthly → show year-month
            df.index = df.index.strftime("%m.%Y")
        else:
            # fallback → keep full datetime
            df.index = df.index.strftime("%Y-%m-%d %H:%M")

        return df

    except Exception as e:
        st.error(f"Error loading stock data: {e}")
        return None

# Cache important keys CSV so it’s loaded only once
@lru_cache(maxsize=1)
def load_keys_to_keep(csv_path="./important_keys.csv"):
    try:
        keys_df = pd.read_csv(csv_path, header=None)
        return set(keys_df.iloc[0].dropna().tolist())
    except Exception:
        return set()

@lru_cache(maxsize=None)
def load_info(ticker: str) -> Dict:
    t = yf.Ticker(ticker)
    info = {}
    try:
        keys_to_keep = load_keys_to_keep()
        info = t.info
        info = {k: v for k, v in info.items() if k in keys_to_keep}
    except Exception:
        info = {}
    return info

@lru_cache(maxsize=None)
def load_news(ticker: str) -> str:
    news_string = ""
    stock = yf.Ticker(ticker)
    try:        
        news = stock.get_news(count=3)
        for article in news:
            news_string += f"Publication Date: {article['content']['pubDate'][:10]}\nsummary:  {article['content']['summary']}\n"
    except Exception as e:
        print(f"error getting news: {e}")
    return news_string

def stock_chart(df: pd.DataFrame, symbol: str):
    """Generate stock chart (candlestick or line) with SMA20, SMA50, and RSI(14)."""

    # --- Subplots (Price + RSI) ---
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3], vertical_spacing=0.05,
        subplot_titles=[f"{symbol[:-3]} Price", "RSI (14)"]
    )

    # --- Price chart (candlestick or line) ---
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"][symbol],
            high=df["High"][symbol],
            low=df["Low"][symbol],
            close=df["Close"][symbol],
            name="Candlestick"
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["Close"][symbol],
            mode="lines",
            name="Close Price",
            line=dict(color="blue", width=1.5),
            visible='legendonly'
        ),
        row=1, col=1
    )

    # --- Moving Averages ---
    fig.add_trace(
        go.Scatter(x=df.index, y=df["SMA20"], mode="lines",
                   name="SMA20", line=dict(color="purple", width=1,),
                   visible='legendonly'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["SMA50"], mode="lines",
                   name="SMA50", line=dict(color="orange", width=1),
                   visible='legendonly'),
        row=1, col=1
    )
    # --- Bollinger Bands ---
    fig.add_trace(
        go.Scatter(x=df.index, y=df["BB_upper"], mode="lines",
                   name="BB Upper", line=dict(color="red", width=1),
                   line_dash="dot",
                   visible='legendonly'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["BB_lower"], mode="lines",
                   name="BB Lower", line=dict(color="green", width=1),
                   line_dash="dot",
                   visible='legendonly'),
        row=1, col=1
    )

    # --- RSI ---
    fig.add_trace(
        go.Scatter(x=df.index, y=df["RSI"], mode="lines",
                   name="RSI", line=dict(color="teal", width=1)),
        row=2, col=1
    )
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

    # --- X-axis formatting ---
    fig.update_xaxes(type="category")  # no gaps

    n_ticks = min(len(df), 10)  # show ~10 ticks max
    tick_positions = df.index[::max(len(df) // n_ticks, 1)]
    tick_labels = [str(i) for i in tick_positions]

    for r in [1, 2]:
        fig.update_xaxes(
            tickvals=tick_positions,
            ticktext=tick_labels,
            tickfont=dict(size=10),
            row=r, col=1
        )

    # --- Layout ---
    fig.update_layout(
        height=600,
        showlegend=True,
        xaxis_rangeslider_visible=False,

        # 🔑 Enable vertical hover line like finance apps
        hovermode="x",  
        xaxis=dict(showspikes=True, spikemode="across", spikesnap="cursor", 
        showline=True, spikethickness=1),
        xaxis2=dict(showspikes=True, spikemode="across", spikesnap="cursor", 
        showline=True, spikethickness=1),
    )
    fig.update_yaxes(
        tickprefix="₹",
        tickformat="~s",  # compact form: 1k, 1M, 1B
        row=1, col=1         # <-- only top chart
    )

    return fig
