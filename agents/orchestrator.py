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
    """Load database schema."""
    metadata = {
        "database_description": "DJIA companies database",
        "tables": {
            "companies": {
                "columns": [
                    {"name": "symbol", "type": "VARCHAR(10)"},
                    {"name": "name", "type": "VARCHAR(255)"},
                    {"name": "sector", "type": "VARCHAR(100)"},
                    {"name": "industry", "type": "VARCHAR(100)"},
                    {"name": "country", "type": "VARCHAR(100)"},
                    {"name": "website", "type": "VARCHAR(255)"},
                    {"name": "market_cap", "type": "DECIMAL(15,2)"},
                    {"name": "pe_ratio", "type": "DECIMAL(10,2)"},
                    {"name": "dividend_yield", "type": "DECIMAL(5,2)"},
                    {"name": "week_high_52", "type": "DECIMAL(10,2)"},
                    {"name": "week_low_52", "type": "DECIMAL(10,2)"},
                    {"name": "description", "type": "TEXT"}
                ]
            },
            "stock_prices": {
                "columns": [
                    {"name": "id", "type": "SERIAL"},
                    {"name": "symbol", "type": "VARCHAR(10)"},
                    {"name": "date", "type": "DATE"},
                    {"name": "close_price", "type": "DECIMAL(10,2)"},
                    {"name": "volume", "type": "BIGINT"},
                    {"name": "high_price", "type": "DECIMAL(10,2)"},
                    {"name": "low_price", "type": "DECIMAL(10,2)"}
                ]
            }
        }
    }
    return metadata

metadata = load_metadata()
schema = yaml.dump({k: v for k, v in metadata.items() if k in ["database_description", "tables"]}, default_flow_style=False, sort_keys=False)

def create_orchestrator():
    tools_config_json = json.dumps(TOOLS_CONFIG, ensure_ascii=False, indent=2)
    system_prompt = f"""
You are Orchestrator, analyzing financial queries and delegating tasks to text2sql_agent or rag_agent. Return ONLY JSON output with agents, sub-queries, dashboard enablement, tickers, and date range. Do NOT include text, explanations, markdown, or code outside JSON.

1. Analyze Query:
   - Match intents:
     {tools_config_json}
   - Prioritize text2sql_agent for data queries (e.g., 'stock', 'price', 'volume', 'chart').
   - Extract:
     - Tickers: e.g., 'AAPL, MSFT' from '(symbol: AAPL)' or names (e.g., 'Apple'); [] for queries not involving specific companies.
     - Date range: e.g., 'in 2024' → {{'start_date': '2024-01-01', 'end_date': '2024-12-31'}}; 'on 2025-04-26' → {{'start_date': '2025-04-26', 'end_date': '2025-04-26'}}; null for non-time-based queries.
   - General query: Set agents=[], message='System supports stock queries, report summaries, and visualizations'.
   - Invalid query: Return error.

2. Delegate:
   - For text2sql_agent: Use original query as sub-query.
   - For rag_agent: Use original query as sub-query.
   - General query: Set agents=[].
   - Invalid query: Return error.

3. Determine Dashboard:
   - Enable dashboard (Dashboard: true) for queries involving 'chart', 'plot', 'graph', 'visualization', or 'diagram'.
   - Disable dashboard (Dashboard: false) for single-value queries (e.g., 'closing price') or rag_agent queries.

4. Output:
   - JSON: {{"status": "success"|"error", "message": "...", "data": {{"agents": ["text2sql_agent"|"rag_agent"], "sub_queries": {{}}, "Dashboard": bool, "tickers": [], "date_range": null|{{start_date, end_date}}}}}}
   - Example: Query: 'Create a bar chart of Caterpillar (CAT) average monthly closing price in 2024'
     - {{"status": "success", "message": "Query analyzed successfully", "data": {{"agents": ["text2sql_agent"], "sub_queries": {{"text2sql_agent": "create a bar chart of Caterpillar (CAT) average monthly closing price in 2024"}}, "Dashboard": true, "tickers": ["CAT"], "date_range": {{"start_date": "2024-01-01", "end_date": "2024-12-31"}}}}}}
   - Example: Query: 'Summarize annual report for Apple'
     - {{"status": "success", "message": "Query analyzed successfully", "data": {{"agents": ["rag_agent"], "sub_queries": {{"rag_agent": "summarize annual report for Apple"}}, "Dashboard": false, "tickers": ["AAPL"], "date_range": null}}}}
"""
    return Agent(
        model=Groq(
            id=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            timeout=30,
            max_retries=5,
            temperature=0.2,
            max_tokens=1000,
            top_p=0.8,
            response_format={"type": "json_object"}
        ),
        system_prompt=system_prompt,
    )