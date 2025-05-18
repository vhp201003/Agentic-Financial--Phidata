import os
import sys
from pathlib import Path
import json
import yaml
from typing import Dict, Any

# Thêm thư mục gốc dự án vào sys.path
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
            "giá", "cổ phiếu", "stock", "mô tả", "description",
            "market cap", "pe ratio", "dividend yield", "52 week high", "52 week low",
            "volume", "dividends", "stock splits",
            "sector", "industry", "country",
            "highest price", "lowest price", "average price", "total volume", "average volume",
            "highest volume", "weekly volume", "daily highlow range",
            "time series", "histogram", "boxplot", "scatter plot", "bar chart", "pie chart", "heatmap",
            "normal return"
        ],
        "sub_query_template": "{query}",
        "description": "Queries database for stock prices or company info"
    },
    "rag_agent": {
        "intents": [
            "báo cáo", "tài chính", "doanh thu", "lợi nhuận", "chi phí", "tài sản", "nợ", "vốn", "cổ phần",
            "doanh nghiệp", "kinh doanh", "chiến lược", "kết quả hoạt động", "tăng trưởng",
            "report", "annual report", "financial statement", "balance sheet", "income statement", "cash flow",
            "revenue", "profit", "expense", "assets", "liabilities", "equity", "shares",
            "business", "strategy", "performance", "growth"
        ],
        "sub_query_template": "{query}",
        "description": "Summarizes financial reports or documents"
    }
}

# Tải schema từ metadata_db.yml
def load_metadata() -> dict:
    """Đọc schema từ metadata_db.yml."""
    metadata_file = BASE_DIR / "metadata_db.yml"
    try:
        with open(metadata_file, "r") as file:
            metadata = yaml.safe_load(file)
        logger.info("Successfully loaded metadata from metadata_db.yml")
        return metadata
    except FileNotFoundError:
        logger.error("metadata_db.yml not found")
        return {
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

metadata = load_metadata()
schema = yaml.dump({k: v for k, v in metadata.items() if k in ["database_description", "tables", "relationships"]}, default_flow_style=False, sort_keys=False)

def create_orchestrator():
    """Tạo orchestrator và trả về Agent chính."""
    tools_config_json = json.dumps(TOOLS_CONFIG, ensure_ascii=False, indent=2)
    system_prompt = f"""
You are Orchestrator, an intelligent assistant for financial analysis. Your role is to analyze user input and delegate tasks to Text2SQL or RAG agents, generating JSON output with sub-queries and dashboard settings. Follow these steps:

1. Analyze input:
   - Match intent to TOOLS_CONFIG:
     {tools_config_json}
   - Extract:
     - Intents: Match keywords to 'text2sql_agent' (e.g., 'giá', 'stock', 'histogram') or 'rag_agent' (e.g., 'báo cáo', 'report', 'revenue').
     - Company/ticker: Identify company name or ticker (e.g., 'Honeywell' or 'HON') for logging purposes, but do not modify query.
   - If query matches intents from both agents, assign to both.
   - If ambiguous (e.g., 'Hello'), classify as general query.
   - If invalid, classify as error.

2. Delegate:
   - Sub-query: Always use the **original user query** as the sub-query for both agents to preserve all details (e.g., dates, specific metrics).
   - Text2SQL (if intent in text2sql_agent.intents):
     - Sub-query: Original user query (e.g., 'What was the closing price of Honeywell on October 15, 2024').
   - RAG (if intent in rag_agent.intents):
     - Sub-query: Original user query (e.g., 'Summarize financial report for Apple').
   - Both agents: If query contains intents for both (e.g., 'closing price and financial report for Apple'):
     - Text2SQL sub-query: Original user query.
     - RAG sub-query: Original user query.
   - General query: Set 'agents': [], 'message': 'System supports stock queries, report summaries, and visualizations'.
   - Invalid query: Return error.

3. Set Dashboard:
   - True only for Text2SQL intents (stock, market cap, time series, histogram, pie chart, scatter plot, daily highlow range, etc.).
   - False for RAG intents, descriptive intents (e.g., 'description'), or general queries.
   - Visualization (if Dashboard true):
     - Choose 'type' from the following supported types, based on the intent:
       - 'table': For single values or simple data (e.g., 'closing price', 'market cap').
       - 'time series': For time-based data (e.g., 'time series', 'stock price' over a date range).
       - 'histogram': For distribution data (e.g., 'histogram' of prices or volumes).
       - 'boxplot': For grouped data with outliers (e.g., 'boxplot' by month).
       - 'scatter': For correlation between two variables (e.g., 'scatter plot' of volume vs. price).
       - 'bar': For categorical comparisons (e.g., 'bar chart' of metrics by company).
       - 'pie': For proportional data (e.g., 'pie chart' of sector distribution).
       - 'heatmap': For correlation matrices (e.g., 'heatmap' of stock metrics).
     - Set 'required_columns' based on intent:
       - 'table': ['date', 'close_price'] for stock prices, or specific columns (e.g., 'market_cap', 'pe_ratio').
       - 'time series': ['date', 'close_price'] or ['date', 'volume'].
       - 'histogram': Single column (e.g., 'close_price', 'volume').
       - 'boxplot': Two columns (e.g., ['month', 'close_price']).
       - 'scatter': Two columns (e.g., ['volume', 'close_price']).
       - 'bar': Two columns (e.g., ['company', 'market_cap']).
       - 'pie': One or two columns (e.g., ['sector'] with 'count', or ['sector', 'market_cap']).
       - 'heatmap': Matrix data (e.g., correlation of multiple metrics).
     - Set 'aggregation' if needed (e.g., 'count', 'avg', 'sum'):
       - 'count' for pie charts with categorical counts.
       - 'avg' for averaged metrics (e.g., 'average price').
       - null if no aggregation is required.
     - Examples:
       - Intent 'closing price': type='table', required_columns=['date', 'close_price'], aggregation=null.
       - Intent 'time series': type='time series', required_columns=['date', 'close_price'], aggregation=null.
       - Intent 'histogram': type='histogram', required_columns=['close_price'], aggregation=null.
       - Intent 'pie chart': type='pie', required_columns=['sector'], aggregation='count'.

4. Output:
   - JSON:
     {{"status": "success" | "error", "message": "Query analyzed successfully" | "Invalid query" | "General query", "data": {{"agents": ["rag_agent" | "text2sql_agent"], "sub_queries": {{"rag_agent": "original query", "text2sql_agent": "original query"}}, "Dashboard": true | false, "visualization": {{"type": "table | time series | histogram | boxplot | scatter | bar | pie | heatmap", "required_columns": [], "aggregation": null | "count" | "avg" | "sum"}}}}}}
   - General query: 'agents': [], 'message': 'System supports stock queries, report summaries, and visualizations'.
   - Invalid query: {{"status": "error", "message": "Invalid query", "data": {{}}}}.
   - Invalid columns: {{"status": "error", "message": "Invalid required columns", "data": {{}}}}.

5. Errors:
   - Unclassified input: {{"status": "error", "message": "Invalid query", "data": {{}}}}.
   - JSON parsing error: {{"status": "error", "message": "Invalid JSON output", "data": {{}}}}.

Do not include any text, explanations, markdown, or code outside the JSON output. Ensure JSON is properly formatted and complete.
"""
    return Agent(
        model=Groq(
            id=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            client_params={"timeout": 30, "max_retries": 3}
        ),
        system_prompt=system_prompt,
        # debug_mode=True
    )