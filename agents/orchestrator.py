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
            "giá", "cổ phiếu", "stock", "mô tả", "description",
            "market cap", "pe ratio", "dividend yield", "52 week high", "52 week low",
            "volume", "dividends", "stock splits",
            "sector", "industry", "country",
            "highest price", "lowest price", "average price", "total volume", "average volume",
            "highest volume", "weekly volume", "daily highlow range",
            "time series", "histogram", "boxplot", "scatter plot", "bar chart", "pie chart", "heatmap",
            "normal return", "chart", "plot", "graph", "visualization", "diagram"
        ],
        "sub_query_template": "{query}",
        "description": "Queries database for stock prices or company info, and handles data visualization requests"
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

def load_metadata() -> dict:
    """Đọc schema và visualization metadata."""
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

    # Load visualization metadata
    vis_metadata_file = BASE_DIR / "config" / "visualization_metadata.yml"
    try:
        with open(vis_metadata_file, "r") as file:
            vis_metadata = yaml.safe_load(file)
        logger.info("Successfully loaded visualization_metadata.yml")
        metadata["visualization_metadata"] = vis_metadata["visualization_metadata"]
    except FileNotFoundError:
        logger.error("visualization_metadata.yml not found")
        metadata["visualization_metadata"] = []
    except Exception as e:
        logger.error(f"Error loading visualization metadata: {str(e)}")
        metadata["visualization_metadata"] = []

    return metadata

metadata = load_metadata()
schema = yaml.dump({k: v for k, v in metadata.items() if k in ["database_description", "tables"]}, default_flow_style=False, sort_keys=False)
vis_metadata_json = json.dumps(metadata["visualization_metadata"], ensure_ascii=False, indent=2)

def create_orchestrator():
    """Tạo orchestrator và trả về Agent chính."""
    tools_config_json = json.dumps(TOOLS_CONFIG, ensure_ascii=False, indent=2)
    system_prompt = f"""
You are Orchestrator, analyzing financial queries and delegating tasks to text2sql_agent or rag_agent. 
Return ONLY JSON output with visualization type, template name, sub-query, tickers, date range, aggregation, required columns, and UI requirements. Do NOT include text, explanations, markdown, or code outside JSON.

1. Analyze Query:
   - Match intents: 
   {tools_config_json} 
   (text2sql_agent: 'stock', 'chart', 'volume', 'heatmap'; rag_agent: 'report', 'revenue').
   - Prioritize text2sql_agent for data queries (e.g., 'stock', 'price', 'volume', 'chart').
   - Extract:
     - Visualization type: Identify from query (e.g., 'table', 'line_chart', 'pie_chart', 'histogram', 'boxplot', 'scatter', 'heatmap'); null for non-visual queries.
     - Tickers: e.g., 'AAPL, MSFT' from '(symbol: AAPL)' or names (e.g., 'Apple'); [] for queries not involving specific companies (e.g., 'distribution of DJIA companies by sector').
     - Date range: e.g., 'in 2024' → {{'start_date': '2024-01-01', 'end_date': '2024-12-31'}}; 'on 2025-04-26' → {{'start_date': '2025-04-26', 'end_date': '2025-04-26'}}; null for non-time-based queries.
     - Aggregation: 'count' for distributions, 'sum' for proportions, 'avg' for averages, 'null' for direct values or time series.
   - General query: Set agents=[], message='System supports stock queries, report summaries, and visualizations'.
   - Invalid query: Return error.

2. Use Visualization Metadata:
   - Visualization metadata:
   {vis_metadata_json}
   - For text2sql_agent queries:
     - Identify vis_type from query (e.g., 'boxplot' from 'Create a boxplot').
     - Find matching vis_type in visualization metadata.
     - Within vis_type, select a template by matching intent_keywords with query (e.g., 'daily returns' matches template 'daily_returns_boxplot').
     - If no template matches, return error.
     - Extract required_columns and ui_requirements from the selected template.
   - For rag_agent queries, set vis_type=null, required_columns=[], ui_requirements={{}}.

3. Delegate:
   - For text2sql_agent: Use original query as sub-query.
   - For rag_agent: Use original query as sub-query.
   - General query: Set agents=[].
   - Invalid query: Return error.

4. Output:
   - JSON: {{"status": "success"|"error", "message": "...", "data": {{"agents": ["text2sql_agent"|"rag_agent"], "sub_queries": {{}}, "Dashboard": bool(True|False), "vis_type": str|null, "template_name": str|null, "tickers": [], "date_range": null|{{start_date, end_date}}, "aggregation": null|"count"|"sum"|"avg", "required_columns": [], "ui_requirements": {{}}}}}}
   - Dashboard: true for visualization queries (e.g., line_chart, pie_chart), false for single-value queries (e.g., closing price) or rag_agent.
   - Example: Query: 'Create a boxplot of daily returns for Apple in 2024'
     - {{"status": "success", "message": "Query analyzed successfully", "data": {{"agents": ["text2sql_agent"], "sub_queries": {{"text2sql_agent": "create a boxplot of daily returns for Apple in 2024"}}, "Dashboard": true, "vis_type": "boxplot", "template_name": "daily_returns_boxplot", "tickers": ["AAPL"], "date_range": {{"start_date": "2024-01-01", "end_date": "2024-12-31"}}, "aggregation": "null", "required_columns": ["date", "daily_return"], "ui_requirements": {{"group_col": "date", "value_col": "daily_return", "group_transform": "to_month"}}}}}}
   - Example: Query: 'Summarize annual report for Apple'
     - {{"status": "success", "message": "Query analyzed successfully", "data": {{"agents": ["rag_agent"], "sub_queries": {{"rag_agent": "summarize annual report for Apple"}}, "Dashboard": false, "vis_type": null, "template_name": null, "tickers": ["AAPL"], "date_range": null, "aggregation": null, "required_columns": [], "ui_requirements": {{}}}}}}
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
        # debug_mode=True
    )