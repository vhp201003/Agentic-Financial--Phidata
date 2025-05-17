# orchestrator.py
import os
import sys
from pathlib import Path
import json
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
            "time series", "histogram", "boxplot", "scatter plot", "bar chart", "pie chart", "heatmap"
        ],
        "sub_query_template": "Retrieve {intent} data for {company}",
        "description": "Queries database for stock prices or company info"
    },
    "rag_agent": {
        "intents": ["báo cáo", "annual report", "tài chính", "report"],
        "sub_query_template": "Summarize {intent} for {company}",
        "description": "Summarizes financial reports or documents"
    }
}

def create_orchestrator():
    """Tạo orchestrator và trả về Agent chính."""
    return Agent(
        model=Groq(
            id=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            client_params={"timeout": 30, "max_retries": 3}
        ),
        description="Orchestrator: Analyzes user query and delegates to RAG, Text2SQL, or other agents.",
        instructions=[
            f"""
            **Objective**: Assign user query to agents using TOOLS_CONFIG, decide if a dashboard is needed, and specify data structure for Text2SQL Agent.

            **TOOLS_CONFIG**:
            {json.dumps(TOOLS_CONFIG, ensure_ascii=False, indent=2)}

            **Output**:
            {{
              "status": "success" | "error",
              "message": "Query analyzed successfully" | "Invalid query",
              "data": {{
                "agents": ["agent_name"],
                "sub_queries": {{"agent_name": "sub-query"}},
                "Dashboard": true | false,
                "visualization": {{"type": "table" | "time series" | "histogram" | "boxplot" | "scatter" | "bar" | "pie" | "heatmap", "required_columns": ["column1", "column2", ...]}}
              }}
            }} or {{}}

            **Rules**:
            - Match query keywords to TOOLS_CONFIG[agent]['intents'] to select agent.
            - Map intents: 
              - 'giá', 'cổ phiếu', 'stock' → 'stock price'
              - 'mô tả', 'description' → 'description'
              - 'market cap' → 'market cap'
              - 'pe ratio' → 'pe ratio'
              - 'dividend yield' → 'dividend yield'
              - '52 week high' → '52 week high'
              - '52 week low' → '52 week low'
              - 'volume' → 'volume'
              - 'dividends' → 'dividends'
              - 'stock splits' → 'stock splits'
              - 'sector' → 'sector'
              - 'industry' → 'industry'
              - 'country' → 'country'
              - 'highest price' → 'highest price'
              - 'lowest price' → 'lowest price'
              - 'average price' → 'average price'
              - 'total volume' → 'total volume'
              - 'average volume' → 'average volume'
              - 'time series' → 'time series'
              - 'histogram' → 'histogram'
              - 'boxplot' → 'boxplot'
              - 'scatter plot' → 'scatter'
              - 'bar chart' → 'bar'
              - 'pie chart' → 'pie'
              - 'heatmap' → 'heatmap'
              - 'báo cáo', 'tài chính', 'report', 'annual report' → 'report'
            - Map additional keywords: 'chiến lược kinh doanh' to 'business strategy'; 'doanh thu' to 'revenue'; 'lợi nhuận' to 'profit'; 'tóm tắt' to 'summarize'.
            - Create sub-query from TOOLS_CONFIG[agent]['sub_query_template'] using mapped {{intent}}, {{company}}.
            - If query contains additional keywords (e.g., 'chiến lược kinh doanh', 'doanh thu'), append them to sub-query (e.g., 'Summarize report for Apple with business strategy').
            - Latest date ('ngày gần nhất', 'most recent'): Use latest date, no specific date in sub-query.
            - Specific date (e.g., 'ngày DD tháng MM năm YYYY' or 'YYYY-MM-DD'): Append 'on YYYY-MM-DD' to sub-query for text2sql_agent.
            - Specific year (e.g., 'năm YYYY' or 'YYYY'): Append 'in YYYY' to sub-query for rag_agent (e.g., 'Summarize report for Apple in 2024').
            - General queries ('What can you do?', 'Hello', 'Hi'): return {{
              "status": "success",
              "message": "General query",
              "data": {{
                "agents": [],
                "sub_queries": {{}},
                "Dashboard": false,
                "general_response": "System supports stock queries, report summaries"
              }}
            }}.
            - Invalid queries: return {{}}.
            - sub_queries keys must match agents.
            - Sub-queries must always be in English: Translate all Vietnamese keywords to English before creating sub-query.
            - Determine Dashboard and visualization:
              - Set "Dashboard": true for queries involving numerical data (stock price, volume, market cap, etc.) or visualization intents (time series, histogram, etc.).
              - Set "Dashboard": false for descriptive queries (description, business strategy, etc.) or general queries.
              - If "Dashboard": true, determine visualization type and required columns:
                - 'stock price', 'volume', 'market cap', etc.: "visualization": {{"type": "table", "required_columns": ["name", "close_price" | "volume" | "market_cap"]}}.
                - 'time series': "visualization": {{"type": "time series", "required_columns": ["date", "close_price"]}}.
                - 'histogram': "visualization": {{"type": "histogram", "required_columns": ["close_price"]}}.
                - 'boxplot': "visualization": {{"type": "boxplot", "required_columns": ["month", "close_price"]}}.
                - 'scatter plot': "visualization": {{"type": "scatter", "required_columns": ["market_cap", "pe_ratio"]}}.
                - 'bar chart': "visualization": {{"type": "bar", "required_columns": ["name", "market_cap"]}}.
                - 'pie chart': "visualization": {{"type": "pie", "required_columns": ["sector", "count"]}}.
                - 'heatmap': "visualization": {{"type": "heatmap", "required_columns": ["date", "close_price"]}}.
              - If "Dashboard": false, "visualization": {{"type": "none", "required_columns": []}}.

            **Example**:
            Input: 'Plot the time series of Microsoft stock closing price from June 1, 2024 to September 30, 2024'
            Output:
            {{
              "status": "success",
              "message": "Query analyzed successfully",
              "data": {{
                "agents": ["text2sql_agent"],
                "sub_queries": {{
                  "text2sql_agent": "Retrieve time series closing price data for Microsoft on 2024-06-01 to 2024-09-30"
                }},
                "Dashboard": true,
                "visualization": {{
                  "type": "time series",
                  "required_columns": ["date", "close_price"]
                }}
              }}
            }}

            **Constraints**:
            - Return ONLY the specified structure: {{...}} or {{}}.
            - Do NOT wrap the output in markdown code blocks (e.g., ```json ... ```).
            - NO markdown, code fences, code, text, or explanations outside the specified structure.
            - Strictly follow Rules and Output format.
            """
        ],
        debug_mode=True
    )