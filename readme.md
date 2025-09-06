# StockSenseAi

**StockSenseAi** is an AI-powered Python application for interactive stock market analysis. Built using **LangGraph** and **Streamlit**, the app combines real-time financial data, dynamic charting, and large language model (LLM) insights to provide actionable stock information and conversational querying capabilities.  

---

## Features

1. **Stock Selection:**  
   - Choose stocks via a sidebar dropdown.  
   - Supports NSE tickers (e.g., HDFC Bank) and other supported exchanges.

2. **Dynamic Stock Charts:**  
   - Visualize stock price movement over multiple periods: 1D, 1WK, 1M, 6M, 1Y, 5Y, MAX.  
   - Candlestick charts with SMA20, SMA50, Bollinger Bands, and RSI(14) subplot.  
   - Interactive zoom and hover features.

3. **LLM-Powered Chart Analysis:**  
   - On-demand narrative insights about chart trends, momentum, and potential reversals.

4. **Interactive Chat Interface:**  
   - Ask about stock metrics, news, financial statements, or buy/hold sentiment.  
   - Integrated access to news, yFinance data, and AI-generated insights.  

5. **Responsive & Intuitive UI:**  
   - Built on **Streamlit** for smooth user experience.  
   - Reset conversation and context-aware chat maintained throughout interactions.

---

## Technical Overview

- **Frontend:** Streamlit for UI components (sidebar, buttons, tabs, charts, chat input).  
- **Backend:** Python with **LangGraph** to manage workflow nodes and state across interactions.  
- **Data Sources:**  
  - `yFinance` for stock data (quotes, historical prices).  
  - News APIs or web scraping for market news.  
  - Social media platforms (Twitter, Reddit, YouTube) for sentiment analysis (planned/extended features).  
- **Visualization:** Plotly for interactive candlestick charts with technical indicators.  
- **AI Integration:** LLMs orchestrated via LangGraph and LangChain for chart analysis and conversational queries. 
