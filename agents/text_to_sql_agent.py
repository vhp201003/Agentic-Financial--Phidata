# agents/text_to_sql_agent.py
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

def load_metadata() -> str:
    """Đọc schema từ metadata_db.yml. Nếu file không tồn tại, tạo file mặc định."""
    metadata_file = BASE_DIR / "metadata_db.yml"
    try:
        with open(metadata_file, "r") as file:
            metadata = yaml.safe_load(file)
        logger.info("Successfully loaded metadata from metadata_db.yml")
        return yaml.dump(metadata, default_flow_style=False, sort_keys=False)
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
            ]
        }
        with open(metadata_file, "w") as file:
            yaml.dump(default_metadata, file, default_flow_style=False, sort_keys=False)
        logger.info("Created default metadata_db.yml")
        return yaml.dump(default_metadata, default_flow_style=False, sort_keys=False)
    except Exception as e:
        logger.error(f"Error loading metadata: {str(e)}")
        return """
The database schema is not provided. Infer the appropriate tables and columns from the query context.
"""

# Schema từ metadata_db.yml
schema = load_metadata()
TOOLS_CONFIG = {
    "text2sql_agent": {
        "intents": ["giá", "cổ phiếu", "stock", "mô tả", "description"],
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
    """Tạo TextToSQL Agent để tạo câu SQL dựa trên schema."""
    logger.info("Creating TextToSQL Agent")

    return Agent(
        model=Groq(
            id=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            client_params={"timeout": 30, "max_retries": 3}
        ),
        description="Text2SQL Agent: sinh câu SQL từ sub-query, chỉ tạo không thực thi.",
        instructions=[
            f"""
            **Schema**: {schema}

            **Objective**: Generate SQL query from sub-query using TOOLS_CONFIG['text2sql_agent']. Do NOT execute SQL.

            **TOOLS_CONFIG**:
            {json.dumps({"text2sql_agent": TOOLS_CONFIG["text2sql_agent"]}, ensure_ascii=False, indent=2)}

            **Output**: JSON array:
            {{
            "status": "success" | "error",
            "message": "SQL query generated successfully" | "Invalid sub-query",
            "data": {{
                "tables": ["table_name", ...],
                "sql_query": "SQL query",
                "result": []
                }}
            }} or {{}}

            **Rules**:
            - Use schema tables/columns only.
            - Map intents: 'giá', 'cổ phiếu' to 'stock price'; 'mô tả' to 'description'.
            - Stock prices ('stock price'): JOIN companies, stock_prices on symbol, SELECT c.name, sp.close_price, LIKE '%company%'.
            - Latest price: ORDER BY sp.date DESC LIMIT 1.
            - Specific date (e.g., 'on YYYY-MM-DD'): Add WHERE sp.date = 'YYYY-MM-DD'.
            - Yearly summary (e.g., 'in YYYY'): Add EXTRACT(YEAR FROM sp.date)=year.
            - Description ('description'): SELECT c.name, c.description, no join, LIKE '%company%'.
            - Invalid sub-query or no SQL: return [].

            **Constraints**:
            - Return ONLY JSON array: {{...}} or {{}}.
            - NO markdown, code fences, code, text, or explanations.
            - Strictly follow Rules and Output format.
            """
        ],
        debug_mode=True
    )