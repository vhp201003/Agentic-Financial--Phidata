# agents/text_to_sql_agent.py
import os
import sys
from pathlib import Path
import json
import yaml
from typing import Dict, Any
import re

# Thêm thư mục gốc dự án vào sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from phi.agent import Agent
from phi.model.groq import Groq
from config.env import GROQ_API_KEY, GROQ_MODEL
from utils.logging import setup_logging
from utils.response import standardize_response

logger = setup_logging()

def load_metadata() -> dict:
    """Đọc schema và common queries từ metadata_db.yml. Nếu file không tồn tại, tạo file mặc định."""
    metadata_file = BASE_DIR / "metadata_db.yml"
    try:
        with open(metadata_file, "r") as file:
            metadata = yaml.safe_load(file)
        logger.info("Successfully loaded metadata from metadata_db.yml")
        return metadata
    except FileNotFoundError:
        logger.warning("metadata_db.yml not found. Creating a default one.")
        default_metadata = {
            "database_description": "DJIA companies database",
            "tables": {
                "companies": {
                    "description": "Contains information about companies in the DJIA index",
                    "columns": [
                        {"name": "symbol", "type": "VARCHAR(10)", "constraints": "PRIMARY KEY", "description": "Stock ticker symbol (e.g., 'AAPL')"},
                        {"name": "name", "type": "VARCHAR(255)", "description": "Full name of the company (e.g., 'Apple Inc.')"},
                        {"name": "description", "type": "TEXT", "description": "Detailed company description"}
                    ]
                },
                "stock_prices": {
                    "description": "Contains price information for the DJIA companies",
                    "columns": [
                        {"name": "id", "type": "SERIAL", "constraints": "PRIMARY KEY", "description": "Unique identifier"},
                        {"name": "symbol", "type": "VARCHAR(10)", "constraints": "FOREIGN KEY referencing companies.symbol", "description": "Stock ticker symbol"},
                        {"name": "date", "type": "DATE", "description": "Date of the price record (YYYY-MM-DD)"},
                        {"name": "close_price", "type": "DECIMAL(10,2)", "description": "Closing price for the day"}
                    ]
                }
            },
            "relationships": [
                {
                    "name": "companies_to_stock_prices",
                    "type": "one-to-many",
                    "description": "One company can have many price records",
                    "from": {"table": "companies", "column": "symbol"},
                    "to": {"table": "stock_prices", "column": "symbol"}
                }
            ],
            "common_queries": [
                {
                    "name": "Closing price on specific date",
                    "description": "Get the closing price of a company on a specific date",
                    "sql": "SELECT sp.close_price FROM companies c JOIN stock_prices sp ON c.symbol = sp.symbol WHERE c.name ILIKE '%{company}%' AND sp.date = '{date}'"
                },
                {
                    "name": "Average closing price by company",
                    "description": "Calculate the average closing price for a company within a date range",
                    "sql": "SELECT c.name, AVG(sp.close_price) AS avg_close_price FROM companies c JOIN stock_prices sp ON c.symbol = sp.symbol WHERE c.name ILIKE '%{company}%' AND sp.date BETWEEN '{start_date}' AND '{end_date}' GROUP BY c.name"
                },
                {
                    "name": "Compare closing price on specific date",
                    "description": "Compare the closing price of two companies on a specific date",
                    "sql": "SELECT c.name, sp.close_price FROM companies c JOIN stock_prices sp ON c.symbol = sp.symbol WHERE c.name IN ('{company1}', '{company2}') AND sp.date = '{date}'"
                },
                {
                    "name": "Highest closing price in range",
                    "description": "Get the highest closing price of a company within a date range",
                    "sql": "SELECT sp.date, sp.close_price FROM companies c JOIN stock_prices sp ON c.symbol = sp.symbol WHERE c.name ILIKE '%{company}%' AND sp.date BETWEEN '{start_date}' AND '{end_date}' ORDER BY sp.close_price DESC LIMIT 1"
                },
                {
                    "name": "Lowest closing price in range",
                    "description": "Get the lowest closing price of a company within a date range",
                    "sql": "SELECT sp.date, sp.close_price FROM companies c JOIN stock_prices sp ON c.symbol = sp.symbol WHERE c.name ILIKE '%{company}%' AND sp.date BETWEEN '{start_date}' AND '{end_date}' ORDER BY sp.close_price ASC LIMIT 1"
                },
                {
                    "name": "Time series data",
                    "description": "Get the closing price time series for a company within a date range",
                    "sql": "SELECT sp.date, sp.close_price FROM companies c JOIN stock_prices sp ON c.symbol = sp.symbol WHERE c.name ILIKE '%{company}%' AND sp.date BETWEEN '{start_date}' AND '{end_date}' ORDER BY sp.date"
                }
            ]
        }
        with open(metadata_file, "w") as file:
            yaml.dump(default_metadata, file, default_flow_style=False, sort_keys=False)
        logger.info("Created default metadata_db.yml")
        return default_metadata
    except Exception as e:
        logger.error(f"Error loading metadata: {str(e)}")
        return {
            "database_description": "The database schema is not provided. Infer the appropriate tables and columns from the query context.",
            "tables": {},
            "relationships": [],
            "common_queries": []
        }

# Schema và common queries từ metadata_db.yml
metadata = load_metadata()
schema = yaml.dump({k: v for k, v in metadata.items() if k in ["database_description", "tables", "relationships"]}, default_flow_style=False, sort_keys=False)
common_queries = metadata.get("common_queries", [])

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
    "finance_agent": {
        "intents": ["phân tích", "xu hướng", "ROI", "trend", "insight"],
        "sub_query_template": "Analyze {intent} for {company}",
        "description": "Analyzes financial trends or insights"
    },
    "rag_agent": {
        "intents": ["báo cáo", "annual report", "tài chính", "report"],
        "sub_query_template": "Summarize {intent} for {company}",
        "description": "Summarizes financial reports or documents"
    }
}

def create_text_to_sql_agent() -> Agent:
    """Tạo Text2SQL Agent để tạo câu SQL dựa trên schema và common queries."""
    logger.info("Creating Text2SQL Agent")

    return Agent(
        model=Groq(
            id=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            client_params={"timeout": 30, "max_retries": 3}
        ),
        description="Text2SQL Agent: sinh câu SQL từ sub-query, chỉ tạo không thực thi.",
        instructions=[
            f"""
            **Constraints** (Read this first):
            - Return ONLY the specified JSON structure: {{"status": "success" | "error", "message": "SQL query generated successfully" | "Invalid sub-query", "data": {{"tables": ["table_name", ...], "sql_query": "SQL query", "result": []}}}} or {{}}.
            - Output must be compact with NO extra spaces, line breaks, or formatting outside the structure (e.g., no pretty printing).
            - Do NOT wrap the output in markdown code blocks (e.g., ```json ... ```).
            - Absolutely NO markdown, code fences, code, text, or explanations outside the specified structure (e.g., do NOT include "Input", "Visualization metadata", etc.).
            - Strictly follow the output format.

            **Schema**: {schema}

            **Common Queries**:
            {json.dumps(common_queries, ensure_ascii=False, indent=2)}

            **Objective**: Generate SQL query from sub-query using TOOLS_CONFIG['text2sql_agent'], ensuring data structure matches the required columns for visualization. Do NOT execute SQL.

            **TOOLS_CONFIG**:
            {json.dumps({"text2sql_agent": TOOLS_CONFIG["text2sql_agent"]}, ensure_ascii=False, indent=2)}

            **Input**:
            - Sub-query (e.g., 'Retrieve stock price data for Nike and Boeing on 2024-08-01').
            - Visualization metadata from Orchestrator (e.g., {{"type": "table", "required_columns": ["name", "close_price"]}}).

            **Output**:
            {{"status": "success" | "error", "message": "SQL query generated successfully" | "Invalid sub-query", "data": {{"tables": ["table_name", ...], "sql_query": "SQL query", "result": []}}}} or {{}}

            **Rules**:
            - First, check if the sub-query matches any template in Common Queries:
              - If the sub-query contains 'stock price data' or 'closing price' and specifies a specific date (e.g., 'on March 15, 2024'), use the 'Closing price on specific date' template, replacing {{company}} and {{date}}.
              - If the sub-query contains 'average price' and specifies a date range (e.g., 'during Q1 2025'), use the 'Average closing price by company' template, replacing {{company}}, {{start_date}}, and {{end_date}}.
              - If the sub-query contains 'compare' or 'which company had higher' and specifies a specific date (e.g., 'on January 15, 2025'), use the 'Compare closing price on specific date' template, replacing {{company1}}, {{company2}}, and {{date}}.
              - If the sub-query contains 'highest closing price' and specifies a date range (e.g., 'in 2024'), use the 'Highest closing price in range' template, replacing {{company}}, {{start_date}}, and {{end_date}}.
              - If the sub-query contains 'lowest closing price' and specifies a date range (e.g., 'in 2023'), use the 'Lowest closing price in range' template, replacing {{company}}, {{start_date}}, and {{end_date}}.
              - If the sub-query contains 'time series data' or 'plot the time series' and specifies a date range (e.g., 'from June 1, 2024 to September 30, 2024'), use the 'Time series data' template, replacing {{company}}, {{start_date}}, and {{end_date}}.
            - If no matching template is found, generate the SQL query using the following rules:
              - Use schema tables/columns only.
              - Map intents: 'giá', 'cổ phiếu', 'stock' → 'stock price'; 'mô tả', 'description' → 'description'; and others as specified in TOOLS_CONFIG.
              - Determine if {{company}} is a stock ticker symbol or company name:
                - If {{company}} is short (less than 5 characters) and all uppercase (e.g., 'MSFT'), search in 'symbol' column with exact match (e.g., c.symbol = '{{company}}').
                - Otherwise, search in 'name' column with partial match (e.g., c.name ILIKE '%{{company}}%').
              - For queries involving stock data (stock price, volume, dividends, etc.):
                - JOIN companies, stock_prices on symbol using exact match (e.g., c.symbol = sp.symbol).
                - SELECT columns from visualization.required_columns.
                - Add WHERE conditions based on sub-query (e.g., sp.date = 'YYYY-MM-DD' for specific date, sp.date BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD' for time range).
                - For 'most recent', add ORDER BY sp.date DESC LIMIT 1.
              - For average calculations ('average price'):
                - JOIN companies, stock_prices on symbol using exact match (e.g., c.symbol = sp.symbol).
                - SELECT c.name, AVG(sp.close_price) AS avg_close_price.
                - Add WHERE conditions based on sub-query (e.g., sp.date BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD' for time range).
                - Add GROUP BY c.name to group results by company name.
                - Do NOT include ORDER BY unless specified in visualization.
              - For descriptive queries (description, sector, etc.):
                - No join, SELECT from companies table only.
              - For visualization types:
                - Adjust SELECT based on visualization.required_columns (e.g., for 'time series', SELECT sp.date, sp.close_price ORDER BY sp.date; for 'average', SELECT c.name, AVG(sp.close_price) AS avg_close_price GROUP BY c.name).
              - Invalid sub-query: return {{}}.
              - Format sql_query as a single-line string without line breaks.

            **Example**:
            Input: 'Retrieve average stock price data for Apple during Q1 2025'
            Visualization metadata: {{"type": "table", "required_columns": ["name", "avg_close_price"]}}
            Output:
            {{"status": "success", "message": "SQL query generated successfully", "data": {{"tables": ["companies", "stock_prices"], "sql_query": "SELECT c.name, AVG(sp.close_price) AS avg_close_price FROM companies c JOIN stock_prices sp ON c.symbol = sp.symbol WHERE c.name ILIKE 'Apple%' AND sp.date BETWEEN '2025-01-01' AND '2025-03-31' GROUP BY c.name", "result": []}}}}
            """
        ],
        debug_mode=True,
    custom_run=lambda self, sub_query: self.run_with_fallback(sub_query)
    )

def run_with_fallback(self, sub_query: str) -> str:
    """Custom run method to handle incorrect response formats from Groq."""
    logger.info(f"Received sub_query: {sub_query}")
    try:
        response = self.model.response(messages=[{"role": "user", "content": self.instructions[0]}])
        logger.debug(f"Raw response from Groq: {response['content']}")
        
        # Loại bỏ markdown code fences và text ngoài JSON
        response_content = response['content']
        json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
        if json_match:
            cleaned_response = json_match.group(0)
            logger.info(f"Cleaned JSON response: {cleaned_response}")
        else:
            logger.error(f"No JSON found in response: {response_content}")
            return json.dumps({
                "status": "error",
                "message": "No JSON found in response",
                "data": {}
            }, ensure_ascii=False)

        # Parse JSON
        try:
            response_dict = json.loads(cleaned_response)
            logger.info(f"Parsed response: {json.dumps(response_dict, ensure_ascii=False)}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse cleaned response as JSON: {cleaned_response}, error: {str(e)}")
            return json.dumps({
                "status": "error",
                "message": f"Failed to parse response as JSON: {str(e)}",
                "data": {}
            }, ensure_ascii=False)

        # Kiểm tra cấu trúc JSON
        if not isinstance(response_dict, dict) or "status" not in response_dict:
            logger.error(f"Invalid response structure: {response_dict}")
            return json.dumps({
                "status": "error",
                "message": "Invalid response structure",
                "data": {}
            }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": f"Error executing query: {str(e)}",
            "data": {}
        }, ensure_ascii=False)