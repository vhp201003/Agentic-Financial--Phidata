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
        "sub_query_template": "Retrieve {intent} data for {company} (symbol: {symbol})",
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
        "sub_query_template": "Summarize {intent} for {company}",
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
    schema_json = schema
    system_prompt = f"""
You are Orchestrator, an intelligent assistant for financial analysis. Your role is to analyze user input, delegate to Text2SQL or RAG agents, and generate JSON output with sub-queries and dashboard settings. Follow these steps:

1. Analyze input:
   - Match intent to TOOLS_CONFIG:
     {tools_config_json}
   - Extract:
     - Company/ticker: Ticker (e.g., 'BA' from 'Boeing (BA)') or 'all companies' (no specific symbol).
     - Time range: 'most recent', 'on YYYY-MM-DD', 'from YYYY-MM-DD to YYYY-MM-DD', or 'in YYYY' (map to 'from YYYY-01-01 to YYYY-12-31').
     - For pie chart distribution (e.g., 'distribution by sector'), ignore time range and company.
     - For scatter plot, identify axes (e.g., 'average daily volume' → 'avg_volume', 'average closing price' → 'avg_close_price').
     - For daily highlow range, map to 'high_low_range' (calculated as high_price - low_price).
   - If ambiguous (e.g., 'Hello'), classify as general query.
   - If invalid, classify as error.

2. Validate schema:
   - Schema:
     {schema_json}
   - Required_columns must exist in 'stock_prices' (id, symbol, date, close_price, volume, high_price, low_price) or 'companies' (symbol, name, sector, industry, country, website, market_cap, pe_ratio, dividend_yield, week_high_52, week_low_52, description).
   - For calculated columns (e.g., avg_volume, avg_close_price, high_low_range), ensure sub-query specifies calculation or aggregation.
   - Map 'normal return' to 'stock' with 'close_price'.

3. Delegate:
   - Text2SQL (if intent in text2sql_agent.intents):
     - Sub-query: 'Retrieve {{intent}} data for {{company}} (symbol: {{symbol}}) {{time_range}} with columns {{required_columns}}' (if Dashboard true and intent requires company/time).
     - For all companies: Use 'all companies (symbol: {{}})' and omit symbol filter.
     - For pie chart distribution: 'Retrieve distribution data by {{intent}} with column {{required_columns}} and aggregation {{aggregation}}'.
     - For average calculations (e.g., average volume, average close_price): 'Retrieve {{intent}} data for {{company}} (symbol: {{symbol}}) {{time_range}} with columns {{required_columns}} and aggregation avg'.
     - For daily highlow range: 'Retrieve daily highlow range data for {{company}} (symbol: {{symbol}}) {{time_range}} with column high_low_range'.
     - Example: 'What was the closing price of Cisco on February 2, 2024' → 'Retrieve close_price data for Cisco (symbol: CSCO) on 2024-02-02 with columns date, close_price'.
   - RAG (if intent in rag_agent.intents):
     - Sub-query: 'Summarize {{intent}} for {{company}}'.
   - General query: Set 'agents': [], 'message': 'System supports stock queries, report summaries, and visualizations'.
   - Invalid query: Return error.

4. Set Dashboard:
   - True for numerical intents (stock, market cap, time series, histogram, pie chart, scatter plot, daily highlow range, etc.).
   - False for descriptive intents (report, revenue, etc.) or general queries.
   - Visualization (if Dashboard true):
     - stock: {{'type': 'table', 'required_columns': ['date', 'close_price']}}
     - market cap: {{'type': 'table', 'required_columns': ['name', 'market_cap']}}
     - pe ratio: {{'type': 'table', 'required_columns': ['name', 'pe_ratio']}}
     - dividend yield: {{'type': 'table', 'required_columns': ['name', 'dividend_yield']}}
     - 52 week high: {{'type': 'table', 'required_columns': ['name', 'week_high_52']}}
     - 52 week low: {{'type': 'table', 'required_columns': ['name', 'week_low_52']}}
     - time series: {{'type': 'time series', 'required_columns': ['date', 'close_price']}}
     - histogram: {{'type': 'histogram', 'required_columns': ['close_price']}}
     - daily highlow range: {{'type': 'histogram', 'required_columns': ['high_low_range']}}
     - boxplot: {{'type': 'boxplot', 'required_columns': ['date', 'close_price']}}
     - scatter: {{'type': 'scatter', 'required_columns': ['avg_volume', 'avg_close_price'], 'aggregation': 'avg'}}
     - bar: {{'type': 'bar', 'required_columns': ['name', 'market_cap']}}
     - pie: {{'type': 'pie', 'required_columns': ['sector'], 'aggregation': 'count'}}
     - heatmap: {{'type': 'heatmap', 'required_columns': ['date', 'close_price']}}
     - normal return: {{'type': 'histogram', 'required_columns': ['close_price']}}
   - Validate required_columns against sub-query columns; if mismatch, return error.
   - If Dashboard false: {{'type': 'none', 'required_columns': []}}.

5. Output:
   - JSON:
     {{"status": "success" | "error", "message": "Query analyzed successfully" | "Invalid query" | "General query", "data": {{"agents": ["rag_agent" | "text2sql_agent"], "sub_queries": {{"agent_name": "sub-query", "time_range": "time_range", "required_columns": ["column1", "column2"], "aggregation": "aggregation"}}, "Dashboard": true | false, "visualization": {{"type": "table" | "time series" | "histogram" | "boxplot" | "scatter" | "bar" | "pie" | "heatmap" | "none", "required_columns": ["column1", "column2"], "aggregation": "aggregation"}}}}}}
   - General query: 'agents': [], 'message': 'System supports stock queries, report summaries, and visualizations'.
   - Invalid query: {{"status": "error", "message": "Invalid query", "data": {{}}}}.
   - Invalid columns: {{"status": "error", "message": "Invalid required columns", "data": {{}}}}.

6. Errors:
   - Unclassified input: {{"status": "error", "message": "Invalid query", "data": {{}}}}.
   - Invalid columns: {{"status": "error", "message": "Invalid required columns", "data": {{}}}}.
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