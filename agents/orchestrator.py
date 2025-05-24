# agents/orchestrator.py
import os
import sys
from pathlib import Path
import json
import yaml
from typing import Dict, Any
import re

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from phi.agent import Agent
from phi.model.groq import Groq
from config.env import GROQ_API_KEY, GROQ_MODEL
from utils.logging import setup_logging
from utils.response import standardize_response

logger = setup_logging()

TOOLS_CONFIG = {
    "text2sql_agent": {
        "intents": [
            "stock", "price", "volume", "market cap", "pe ratio", "dividend yield",
            "52 week high", "52 week low", "dividends", "stock splits",
            "sector", "industry", "country", "highest price", "lowest price",
            "average price", "total volume", "average volume", "highest volume",
            "weekly volume", "daily highlow range", "time series", "histogram",
            "boxplot", "scatter plot", "bar chart", "pie chart", "heatmap",
            "normal return", "chart", "plot", "graph", "visualization", "diagram"
        ],
        "sub_query_template": "{query}",
        "description": "Queries database for stock prices or company info"
    },
    "rag_agent": {
        "intents": [
            "report", "annual report", "financial statement", "balance sheet",
            "income statement", "cash flow", "revenue", "profit", "expense",
            "assets", "liabilities", "equity", "shares", "business", "strategy",
            "performance", "growth"
        ],
        "sub_query_template": "{query}",
        "description": "Summarizes financial reports or documents"
    }
}

def load_metadata() -> dict:
    """Load database schema from metadata_db.yml."""
    metadata_file = BASE_DIR / "config" / "metadata_db.yml"
    try:
        with open(metadata_file, "r") as file:
            metadata = yaml.safe_load(file)
        logger.info("Successfully loaded metadata_db.yml")
        return metadata
    except FileNotFoundError:
        logger.error("metadata_db.yml not found")
        return {}
    except Exception as e:
        logger.error(f"Error loading metadata_db.yml: {str(e)}")
        return {}

metadata = load_metadata()
schema = yaml.dump({k: v for k, v in metadata.items() if k in ["database_description", "tables", "relationships"]}, default_flow_style=False, sort_keys=False)

def create_orchestrator():
    tools_config_json = json.dumps(TOOLS_CONFIG, ensure_ascii=False, indent=2)
    system_prompt = f"""
You are Orchestrator, analyzing financial queries and delegating tasks to text2sql_agent or rag_agent. Return ONLY JSON output with agents, sub-queries, dashboard enablement, tickers, and date range. Do NOT include text, explanations, markdown, or code outside JSON.

Input format: JSON string with "query" (current query) and "chat_history" (list of previous interactions).
- Example input: {{"query": "What is the stock price of Apple?", "chat_history": [{{"role": "user", "content": "Tell me about Apple", "timestamp": "2025-05-24 08:16:45"}}, {{"role": "assistant", "content": "Apple Inc. is a tech company...", "timestamp": "2025-05-24 08:16:50"}}]}}

1. Analyze Chat History for Context:
   - Use chat_history to understand context and relationships between queries.
   - Example: If chat_history contains "Tell me about Apple" and current query is "What is its stock price?", infer "its" refers to Apple.
   - Limit history to the last 5 interactions to avoid token overflow.

2. Analyze Current Query:
   - Match intents:
     {tools_config_json}
   - Prioritize text2sql_agent for data queries (e.g., 'stock', 'price', 'volume', 'chart').
   - Extract:
     - Tickers: e.g., 'AAPL, MSFT' from '(symbol: AAPL)' or names (e.g., 'Apple'); [] for queries not involving specific companies. Use chat_history to infer tickers if query is ambiguous (e.g., "its stock price" after mentioning "Apple").
     - Date range: e.g., 'in 2024' → {{'start_date': '2024-01-01', 'end_date': '2024-12-31'}}; 'on 2025-04-26' → {{'start_date': '2025-04-26', 'end_date': '2025-04-26'}}; null for non-time-based queries. Use chat_history to infer dates if query is ambiguous (e.g., "last year" after mentioning "2024").
   - General query: Set agents=[], message='System supports stock queries, report summaries, and visualizations'.
   - Invalid query: Return error.

3. Delegate:
   - For text2sql_agent: Use original query as sub-query.
   - For rag_agent: Use original query as sub-query.
   - General query: Set agents=[].
   - Invalid query: Return error.

4. Determine Dashboard:
   - Enable dashboard (Dashboard: true) for queries involving 'chart', 'plot', 'graph', 'visualization', or 'diagram'.
   - Disable dashboard (Dashboard: false) for single-value queries (e.g., 'closing price') or rag_agent queries.

5. Output:
   - JSON: {{"status": "success"|"error", "message": "...", "data": {{"agents": ["text2sql_agent"|"rag_agent"], "sub_queries": {{}}, "Dashboard": bool, "tickers": [], "date_range": null|{{start_date, end_date}}}}}}
   - Example: Query: 'Create a bar chart of Caterpillar (CAT) average monthly closing price in 2024'
     - {{"status": "success", "message": "Query analyzed successfully", "data": {{"agents": ["text2sql_agent"], "sub_queries": {{"text2sql_agent": "create a bar chart of Caterpillar (CAT) average monthly closing price in 2024"}}, "Dashboard": true, "tickers": ["CAT"], "date_range": {{"start_date": "2024-01-01", "end_date": "2024-12-31"}}}}}}
   - Example: Query: 'Summarize annual report for Apple'
     - {{"status": "success", "message": "Query analyzed successfully", "data": {{"agents": ["rag_agent"], "sub_queries": {{"rag_agent": "summarize annual report for Apple"}}, "Dashboard": false, "tickers": ["AAPL"], "date_range": null}}}}
"""
    return Agent(
        model=Groq(
            id="llama-3.3-70b-versatile",
            api_key=GROQ_API_KEY,
            timeout=30,
            max_retries=5,
            temperature=0.2,
            max_tokens=1000,
            top_p=0.8,
            response_format={"type": "json_object"}
        ),
        system_prompt=system_prompt,
        debug_mode=True,
    )