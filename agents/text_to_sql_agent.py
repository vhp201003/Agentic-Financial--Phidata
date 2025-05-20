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

def load_metadata() -> dict:
    """Đọc schema và visualization metadata."""
    metadata = {
        "database_description": "Dow Jones Industrial Average (DJIA) companies database",
        "tables": {
            "companies": {
                "description": "Contains basic company information and metrics",
                "columns": [
                    {"name": "symbol", "type": "VARCHAR(10)", "constraints": "PRIMARY KEY", "description": "Stock ticker symbol (e.g., 'AAPL')"},
                    {"name": "name", "type": "VARCHAR(255)", "description": "Full name of the company (e.g., 'Apple Inc.')"},
                    {"name": "sector", "type": "VARCHAR(100)", "description": "Economic sector classification (e.g., 'Technology')"},
                    {"name": "industry", "type": "VARCHAR(100)", "description": "Specific industry within sector (e.g., 'Consumer Electronics')"},
                    {"name": "country", "type": "VARCHAR(100)", "description": "Country where company is headquartered (e.g., 'United States')"},
                    {"name": "website", "type": "VARCHAR(255)", "description": "Company's official website URL"},
                    {"name": "market_cap", "type": "BIGINT", "description": "Market capitalization value in USD"},
                    {"name": "pe_ratio", "type": "DECIMAL(10,2)", "description": "Price-to-earnings ratio"},
                    {"name": "dividend_yield", "type": "DECIMAL(5,2)", "description": "Annual dividend yield percentage"},
                    {"name": "week_high_52", "type": "DECIMAL(10,2)", "description": "Highest stock price in the last 52 weeks"},
                    {"name": "week_low_52", "type": "DECIMAL(10,2)", "description": "Lowest stock price in the last 52 weeks"},
                    {"name": "description", "type": "TEXT", "description": "Detailed company description"}
                ]
            },
            "stock_prices": {
                "description": "Contains historical price information for stocks",
                "columns": [
                    {"name": "id", "type": "SERIAL", "constraints": "PRIMARY KEY", "description": "Unique identifier"},
                    {"name": "symbol", "type": "VARCHAR(10)", "constraints": "FOREIGN KEY referencing companies.symbol", "description": "Stock ticker symbol"},
                    {"name": "date", "type": "DATE", "description": "Date of the price record (YYYY-MM-DD)"},
                    {"name": "open_price", "type": "DECIMAL(10,2)", "description": "Opening price for the day"},
                    {"name": "high_price", "type": "DECIMAL(10,2)", "description": "Highest price during the day"},
                    {"name": "low_price", "type": "DECIMAL(10,2)", "description": "Lowest price during the day"},
                    {"name": "close_price", "type": "DECIMAL(10,2)", "description": "Closing price for the day"},
                    {"name": "volume", "type": "BIGINT", "description": "Trading volume for the day"},
                    {"name": "dividends", "type": "DECIMAL(10,2)", "description": "Dividends paid"},
                    {"name": "stock_splits", "type": "DECIMAL(10,2)", "description": "Stock split ratio"}
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

metadata = load_metadata()
SCHEMA = yaml.dump(
    {k: v for k, v in metadata.items() if k in ["database_description", "tables", "relationships"]},
    default_flow_style=False,
    sort_keys=False
)

ERROR_MESSAGES = {
    "missing_date": "Không tạo được câu SQL: thiếu thông tin thời gian",
    "invalid_query": "Không tạo được câu SQL: truy vấn không hợp lệ",
    "missing_template": "Không tạo được câu SQL: không tìm thấy template",
    "invalid_columns": "Không tạo được câu SQL: cột không hợp lệ"
}

def create_text_to_sql_agent() -> Agent:
    """Tạo Text2SQL Agent để phân tích truy vấn và chọn template SQL."""
    logger.info("Creating Text2SQL Agent")

    tools_config_json = json.dumps({"text2sql_agent": TOOLS_CONFIG["text2sql_agent"]}, ensure_ascii=False, indent=2)
    system_prompt = f"""
You are Text2SQL Agent, generating plain SQL for a PostgreSQL financial database. Follow these steps to generate SQL queries based on the metadata. Return ONLY the SQL query or error message, no explanations, no markdown (e.g., avoid ```sql, ```json, or ```).

- **Database Schema**:
{SCHEMA}

- **Generate SQL**:
  1. **Extract Metadata**:
     - Extract template_name, tickers, date_range, and required_columns from metadata.
     - Use template_name to find the corresponding SQL template in visualization_metadata.

  2. **Validate Required Columns**:
     - Check the required_columns (e.g., ['month', 'close_price']) from metadata.
     - Ensure the SQL query returns EXACTLY these columns.
     - If the template's SQL does not match required_columns, adjust the query:
       - Example: If required_columns=['month', 'close_price'] but template returns 'date', modify to use TO_CHAR(date, 'YYYY-MM') AS month and GROUP BY month.

  3. **Adjust for Grouping**:
     - If required_columns includes 'month' but template returns 'date', group the data by month:
       - Use TO_CHAR(date, 'YYYY-MM') AS month in SELECT.
       - Add GROUP BY TO_CHAR(date, 'YYYY-MM') to the query.
       - Apply appropriate aggregation (e.g., AVG(close_price)) for other columns.
     - Preserve other parts of the query (e.g., WHERE, ORDER BY) unless they conflict with grouping.

  4. **Check Schema**:
     - Verify columns exist in schema.
     - Use JOIN for columns not in target table (e.g., 'name' from 'companies').

  5. **Handle Tickers**:
     - Use metadata.tickers, uppercase (e.g., 'AAPL').
     - If empty, query by company name with LIKE (e.g., '%Apple%') and JOIN.

  6. **Handle Date Ranges**:
     - Use BETWEEN for ranges.
     - For missing data, use subquery for nearest prior date.

  7. **Generate Query**:
     - Start with the SQL template from visualization_metadata.
     - Adjust the SELECT clause to match required_columns.
     - Add GROUP BY if required (e.g., for 'month').
     - Format single-line SQL with semicolon, no markdown or JSON formatting.
     - Example: SELECT TO_CHAR(date, 'YYYY-MM') AS month, AVG(close_price) AS close_price FROM stock_prices WHERE symbol = 'DIS' AND date BETWEEN '2024-01-01' AND '2024-12-31' GROUP BY TO_CHAR(date, 'YYYY-MM') ORDER BY TO_CHAR(date, 'YYYY-MM');

- **Errors**:
  - Missing date: '{ERROR_MESSAGES["missing_date"]}'
  - Invalid query: '{ERROR_MESSAGES["invalid_query"]}'
  - Missing template: '{ERROR_MESSAGES["missing_template"]}'
  - Invalid columns: '{ERROR_MESSAGES["invalid_columns"]}'

Examples:
1. Metadata: template_name='daily_returns_boxplot', tickers=['AAPL'], date_range={{'start_date': '2024-01-01', 'end_date': '2024-12-31'}}, required_columns=['date', 'daily_return']
   SQL: SELECT date, (close_price - LAG(close_price) OVER (PARTITION BY symbol ORDER BY date)) / LAG(close_price) OVER (PARTITION BY symbol ORDER BY date) AS daily_return FROM stock_prices WHERE symbol = 'AAPL' AND date BETWEEN '2024-01-01' AND '2024-12-31' ORDER BY date;
2. Metadata: template_name='monthly_prices_boxplot', tickers=['DIS'], date_range={{'start_date': '2024-01-01', 'end_date': '2024-12-31'}}, required_columns=['month', 'close_price']
   SQL: SELECT TO_CHAR(date, 'YYYY-MM') AS month, AVG(close_price) AS close_price FROM stock_prices WHERE symbol = 'DIS' AND date BETWEEN '2024-01-01' AND '2024-12-31' GROUP BY TO_CHAR(date, 'YYYY-MM') ORDER BY TO_CHAR(date, 'YYYY-MM');
"""
    return Agent(
        model=Groq(
            id=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            timeout=30,
            max_retries=5,
            temperature=0.2,
            max_tokens=1000,
            top_p=0.8
        ),
        system_prompt=system_prompt,
        custom_run=lambda self, sub_query, metadata=None: self.run_with_fallback(sub_query, metadata),
        # debug_mode=True,
    )

def run_with_fallback(self, sub_query: str, metadata: dict = None) -> str:
    logger.info(f"Received sub_query: {sub_query}, metadata: {metadata}")
    try:
        templates = metadata.get('visualization_metadata', [])
        template_name = metadata.get('template_name', None)
        tickers = metadata.get('tickers', [])
        date_range = metadata.get('date_range', None)
        required_columns = metadata.get('required_columns', [])
        valid_columns = [col['name'] for table in metadata['tables'].values() for col in table['columns']] + [
            'avg_volume', 'avg_close_price', 'high_low_range', 'proportion', 'count', 'total_dividends',
            'name', 'daily_return', 'avg_daily_volume', 'avg_closing_price', 'month'
        ]

        if not template_name:
            logger.error("Missing template_name in metadata")
            return ERROR_MESSAGES["missing_template"]

        template = None
        for vis_type_entry in templates:
            for t in vis_type_entry['templates']:
                if t['name'] == template_name:
                    template = t
                    break
            if template:
                break

        if not template:
            logger.error(f"No template found for template_name: {template_name}")
            return ERROR_MESSAGES["missing_template"]

        params = {
            'ticker': tickers[0] if tickers else '',
            'tickers': ','.join(f"'{t}'" for t in tickers) if tickers else '',
            'start_date': date_range['start_date'] if date_range else '2024-01-01',
            'end_date': date_range['end_date'] if date_range else '2024-12-31',
        }

        sql_query = template['sql'].format(**params)
        sql_query = re.sub(r'[\n\r\t]+', ' ', sql_query).strip()
        if not sql_query.endswith(';'):
            sql_query += ';'

        # Điều chỉnh SELECT clause khi vis_type là null
        if metadata.get('vis_type') is None:
            stock_prices_columns = [col['name'] for col in metadata['tables']['stock_prices']['columns']]
            if all(col in stock_prices_columns for col in required_columns):
                select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_query, re.IGNORECASE)
                if select_match:
                    current_select = select_match.group(1)
                    new_select = ', '.join(required_columns)
                    sql_query = sql_query.replace(current_select, new_select)
                    logger.info(f"Adjusted SQL for vis_type null: {sql_query}")
            else:
                logger.error(f"Required columns {required_columns} not all in stock_prices columns")
                return "Không tạo được câu SQL: cột yêu cầu không hợp lệ cho truy vấn đơn giản"

        logger.info(f"Generated SQL: {sql_query}")
        return sql_query

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return f"Không tạo được câu SQL: {str(e)}"