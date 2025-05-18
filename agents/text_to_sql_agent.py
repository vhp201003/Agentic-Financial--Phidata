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
                        {"name": "sector", "type": "VARCHAR(100)", "description": "Industry sector (e.g., 'Technology')"},
                        {"name": "industry", "type": "VARCHAR(100)", "description": "Specific industry (e.g., 'Consumer Electronics')"},
                        {"name": "country", "type": "VARCHAR(100)", "description": "Country where company is headquartered (e.g., 'USA')"},
                        {"name": "website", "type": "VARCHAR(255)", "description": "Company website URL (e.g., 'https://www.apple.com')"},
                        {"name": "market_cap", "type": "DECIMAL(15,2)", "description": "Market capitalization in USD"},
                        {"name": "pe_ratio", "type": "DECIMAL(10,2)", "description": "Price-to-earnings ratio"},
                        {"name": "dividend_yield", "type": "DECIMAL(5,2)", "description": "Dividend yield percentage"},
                        {"name": "week_high_52", "type": "DECIMAL(10,2)", "description": "52-week high stock price"},
                        {"name": "week_low_52", "type": "DECIMAL(10,2)", "description": "52-week low stock price"},
                        {"name": "description", "type": "TEXT", "description": "Detailed company description"}
                    ]
                },
                "stock_prices": {
                    "description": "Contains price information for the DJIA companies",
                    "columns": [
                        {"name": "id", "type": "SERIAL", "constraints": "PRIMARY KEY", "description": "Unique identifier"},
                        {"name": "symbol", "type": "VARCHAR(10)", "constraints": "FOREIGN KEY referencing companies.symbol", "description": "Stock ticker symbol"},
                        {"name": "date", "type": "DATE", "description": "Date of the price record (YYYY-MM-DD)"},
                        {"name": "close_price", "type": "DECIMAL(10,2)", "description": "Closing price for the day"},
                        {"name": "volume", "type": "BIGINT", "description": "Trading volume for the day"},
                        {"name": "high_price", "type": "DECIMAL(10,2)", "description": "Highest price for the day"},
                        {"name": "low_price", "type": "DECIMAL(10,2)", "description": "Lowest price for the day"}
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
                    "sql": "SELECT {required_columns} FROM companies c JOIN stock_prices sp ON c.symbol = sp.symbol WHERE c.name ILIKE '%{company}%' AND sp.date = '{date}'"
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
                },
                {
                    "name": "Boxplot data by month",
                    "description": "Get closing prices grouped by month for a company within a date range",
                    "sql": "SELECT EXTRACT(MONTH FROM sp.date) AS month, sp.close_price FROM companies c JOIN stock_prices sp ON c.symbol = sp.symbol WHERE c.name ILIKE '%{company}%' AND sp.date BETWEEN '{start_date}' AND '{end_date}' ORDER BY month"
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
schema = yaml.dump({k: v for k, v in metadata.items() if k in ["database_description", "tables", "relationships"]}, default_flow_style=False, sort_keys=False)
common_queries = metadata.get("common_queries", [])

def create_text_to_sql_agent() -> Agent:
    """Tạo Text2SQL Agent để tạo câu SQL dựa trên schema và common queries."""
    logger.info("Creating Text2SQL Agent")

    tools_config_json = json.dumps({"text2sql_agent": TOOLS_CONFIG["text2sql_agent"]}, ensure_ascii=False, indent=2)
    common_queries_json = json.dumps(common_queries, ensure_ascii=False, indent=2)
    schema_json = schema
    system_prompt = f"""
You are Text2SQL Agent, an intelligent assistant specializing in generating SQL queries for a PostgreSQL financial database. Your role is to create a plain, syntactically correct SQL query based on the input query, using the provided schema and common queries, without executing the SQL. Follow these steps using chain-of-thought reasoning:

1. Analyze the query:
   - Match query to intents in TOOLS_CONFIG['text2sql_agent']:
     {tools_config_json}
   - Intents include: 'giá', 'cổ phiếu', 'stock', 'market cap', 'pe ratio', 'dividend yield', 'volume', 'time series', 'histogram', 'boxplot', 'scatter plot', 'bar chart', 'pie chart', 'heatmap', 'daily highlow range', etc.
   - Map intents: 'giá', 'cổ phiếu', 'stock' to 'stock price'; 'mô tả', 'description' to 'description'; 'daily highlow range' to 'high_low_range' (high_price - low_price); others as specified.
   - Extract company and ticker symbol:
     - If query contains '(symbol: <ticker>)' (e.g., 'Retrieve stock price data for Nike (symbol: NKE)'), extract ticker (e.g., 'NKE') and use c.symbol = '<ticker>' in WHERE clause.
     - If query specifies 'all companies (symbol: {{}})', query all companies without filtering symbol.
     - Otherwise, use company name for partial match (e.g., 'Nike'). For names with single quotes (e.g., 'McDonald''s'), escape by doubling the single quote (e.g., 'McDonald''s') for PostgreSQL compatibility.
     - Clean company name to avoid duplication (e.g., remove repeated 'Corporation' from 'Microsoft Corporation Corporation').
   - Extract date or time range:
     - 'on YYYY-MM-DD': Specific date (e.g., 'on 2024-08-01').
     - 'from YYYY-MM-DD to YYYY-MM-DD': Date range (e.g., 'from 2024-06-01 to 2024-09-30').
     - 'in YYYY': Year (e.g., 'in 2024'), map to BETWEEN 'YYYY-01-01' AND 'YYYY-12-31'.
     - 'most recent': Latest date, use ORDER BY sp.date DESC LIMIT 1.
     - If no time range is specified and intent requires time (e.g., stock price, daily highlow range), return error: 'Không tạo được câu SQL: thiếu thông tin thời gian'.
   - Extract required columns and aggregation:
     - Parse query for 'with column <column_list>' or 'with columns <column_list> and aggregation <aggregation>' (e.g., 'with columns date, close_price').
     - If no columns specified, infer columns based on intent (e.g., 'stock price' → 'date, close_price'; 'description' → 'description'; 'histogram' → 'close_price' or 'high_low_range').
     - Validate columns against schema:
       {schema_json}
     - If columns are not in schema or calculated columns (e.g., avg_volume, avg_close_price, high_low_range), ensure calculation is specified.
     - If aggregation is specified (e.g., 'count', 'avg'), apply COUNT(*) or AVG() in SQL.

2. Match common queries:
   - Check if query matches a template in common queries:
     {common_queries_json}
   - If matched, use the template and replace placeholders:
     - 'Closing price on specific date': Replace {{company}} (escape single quotes, e.g., 'McDonald''s'), {{date}}, and adjust SELECT to include all columns specified in query (e.g., sp.date, sp.close_price).
     - 'Average closing price by company': Replace {{company}}, {{start_date}}, {{end_date}}.
     - 'Compare closing price on specific date': Replace {{company1}}, {{company2}}, {{date}}.
     - 'Highest closing price in range': Replace {{company}}, {{start_date}}, {{end_date}}.
     - 'Lowest closing price in range': Replace {{company}}, {{start_date}}, {{end_date}}.
     - 'Time series data': Replace {{company}}, {{start_date}}, {{end_date}}.
     - 'Boxplot data by month': Replace {{company}}, {{start_date}}, {{end_date}}.
   - For 'in YYYY', set start_date='YYYY-01-01', end_date='YYYY-12-31'.
   - If ticker is provided, replace company name match with c.symbol = '<ticker>'.
   - If required columns are specified in query (e.g., 'with columns date, close_price'), override default SELECT columns with specified columns.

3. Generate SQL query if no common query matches:
   - Use schema:
     {schema}
   - Validate required columns against schema or calculated columns (e.g., avg_volume, avg_close_price, high_low_range).
   - For stock data (stock price, volume, etc.):
     - JOIN companies, stock_prices on c.symbol = sp.symbol.
     - SELECT all columns specified in required_columns or inferred columns (e.g., sp.date, sp.close_price for stock price).
     - For aggregation 'avg' (e.g., avg_volume, avg_close_price):
       - SELECT c.symbol, c.name, AVG(sp.close_price) AS avg_close_price, AVG(sp.volume) AS avg_volume
       - GROUP BY c.symbol, c.name
     - For daily highlow range:
       - SELECT sp.date, (sp.high_price - sp.low_price) AS high_low_range
       - JOIN companies, stock_prices on c.symbol = sp.symbol
       - WHERE c.symbol = '<ticker>' and sp.date BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD'
     - WHERE conditions:
       - If 'all companies (symbol: {{}})', omit symbol filter.
       - Otherwise, c.symbol = '<ticker>' (if ticker provided) or c.name ILIKE '%<company>%' (escape single quotes in <company>, e.g., 'McDonald''s').
       - For specific date: sp.date = 'YYYY-MM-DD'.
       - For date range: Always use sp.date BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD' (e.g., sp.date BETWEEN '2024-06-01' AND '2024-09-30').
       - For 'in YYYY': sp.date BETWEEN 'YYYY-01-01' AND 'YYYY-12-31'.
       - For 'most recent': ORDER BY sp.date DESC LIMIT 1.
   - For distribution queries (e.g., 'distribution data by sector'):
     - SELECT c.<grouping_column>, COUNT(*) AS count
     - FROM companies c
     - GROUP BY c.<grouping_column>
     - Example: SELECT c.sector, COUNT(*) AS count FROM companies c GROUP BY c.sector
   - For average calculations (average price, average volume):
     - SELECT c.symbol, c.name, AVG(sp.close_price) AS avg_close_price, AVG(sp.volume) AS avg_volume
     - JOIN companies, stock_prices on c.symbol = sp.symbol
     - WHERE conditions as above
     - GROUP BY c.symbol, c.name
   - For boxplot:
     - SELECT EXTRACT(MONTH FROM sp.date) AS month, sp.close_price
     - JOIN companies, stock_prices on c.symbol = sp.symbol
     - WHERE conditions as above
     - ORDER BY month
   - For descriptive queries (description, sector):
     - SELECT from companies table only (e.g., c.description)
     - Example: SELECT c.description FROM companies c WHERE c.name ILIKE '%<company>%' (escape single quotes in <company>)

4. Format and validate the output:
   - Return ONLY a plain SQL query string or error message as plain text, no JSON, no markdown code blocks.
   - Ensure SQL is syntactically correct for PostgreSQL:
     - Escape single quotes in string literals by doubling them (e.g., 'McDonald''s' instead of 'McDonald\'s').
     - Use single '%' for ILIKE patterns (e.g., ILIKE '%McDonald''s%').
     - Remove line breaks, tabs, and extra spaces; format as a single-line string.
     - For date ranges, always use BETWEEN instead of >= and <= (e.g., sp.date BETWEEN '2024-06-01' AND '2024-09-30').
   - Validate SQL:
     - Check for invalid characters (e.g., double '%', unescaped quotes).
     - Ensure WHERE clauses are properly formatted (e.g., no trailing AND, no missing conditions).
     - Ensure date ranges use BETWEEN (e.g., reject queries with >= and <= for ranges).
   - Example: SELECT sp.date, sp.close_price FROM companies c JOIN stock_prices sp ON c.symbol = sp.symbol WHERE c.symbol = 'MSFT' AND sp.date BETWEEN '2024-06-01' AND '2024-09-30' ORDER BY sp.date
   - If invalid: Return 'Không tạo được câu SQL: truy vấn không hợp lệ'

5. Error handling:
   - If query is invalid or cannot be processed, return: 'Không tạo được câu SQL: truy vấn không hợp lệ'
   - If schema is insufficient, return: 'Không tạo được câu SQL: thiếu thông tin schema'
   - If date format is invalid, return: 'Không tạo được câu SQL: định dạng ngày không hợp lệ'
   - If time range is missing for time-based intents, return: 'Không tạo được câu SQL: thiếu thông tin thời gian'
   - If required columns are invalid, return: 'Không tạo được câu SQL: cột không hợp lệ'
   - If SQL syntax is invalid (e.g., unescaped quotes, extra '%'), return: 'Không tạo được câu SQL: lỗi cú pháp SQL'
   - If date range uses >= and <= instead of BETWEEN, return: 'Không tạo được câu SQL: phải dùng BETWEEN cho khoảng ngày'

Do not include any text, explanations, markdown, or code outside the plain SQL query or error message. Ensure the SQL query is syntactically correct for PostgreSQL.
"""
    return Agent(
        model=Groq(
            id=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            client_params={"timeout": 30, "max_retries": 3}
        ),
        system_prompt=system_prompt,
        debug_mode=True,
        custom_run=lambda self, sub_query: self.run_with_fallback(sub_query)
    )

def run_with_fallback(self, sub_query: str) -> str:
    """Custom run method to handle incorrect response formats from Groq and ensure valid SQL."""
    logger.info(f"Received sub_query: {sub_query}")
    try:
        # Validate sub-query for time range if required
        if 'stock price' in sub_query.lower() or 'time series' in sub_query.lower() or 'histogram' in sub_query.lower() or 'average' in sub_query.lower() or 'daily highlow range' in sub_query.lower():
            if not re.search(r'(on \d{{4}}-\d{{2}}-\d{{2}}|from \d{{4}}-\d{{2}}-\d{{2}} to \d{{4}}-\d{{2}}-\d{{2}}|in \d{{4}}|most recent)', sub_query):
                logger.error(f"Sub-query missing time range: {sub_query}")
                return "Không tạo được câu SQL: thiếu thông tin thời gian"

        # Validate required columns and aggregation
        columns_match = re.search(r'with column ([\w,\s]+)|with columns ([\w,\s]+)(?: and aggregation (\w+))?', sub_query)
        if columns_match:
            columns = [col.strip() for col in (columns_match.group(1) or columns_match.group(2)).split(',')]
            aggregation = columns_match.group(3) if columns_match.group(3) else None
            valid_columns = [col['name'] for table in metadata['tables'].values() for col in table['columns']] + ['avg_volume', 'avg_close_price', 'high_low_range', 'count']
            valid_aggregations = ['count', 'avg', 'sum', 'min', 'max', None]
            for col in columns:
                if col not in valid_columns:
                    logger.error(f"Invalid column in sub-query: {col}")
                    return f"Không tạo được câu SQL: cột không hợp lệ: {col}"
            if aggregation not in valid_aggregations:
                logger.error(f"Invalid aggregation in sub_query: {aggregation}")
                return f"Không tạo được câu SQL: aggregation không hợp lệ: {aggregation}"

        # Clean company name to avoid duplication
        ticker_match = re.search(r'\(symbol:\s*(\w+)\)', sub_query)
        if ticker_match:
            ticker = ticker_match.group(1).upper()
            logger.info(f"Extracted ticker: {ticker}")
            # Remove ticker from company name to avoid duplication
            sub_query_clean = re.sub(r'\(symbol:\s*\w+\)', '', sub_query).strip()
            # Remove repeated 'Corporation' or similar terms
            sub_query_clean = re.sub(r'\b(Corporation|Inc\.|Corp\.)\s+\1\b', r'\1', sub_query_clean, flags=re.IGNORECASE)
            logger.info(f"Cleaned sub_query: {sub_query_clean}")
        else:
            sub_query_clean = sub_query

        response = self.model.response(messages=[{"role": "user", "content": sub_query_clean}])
        logger.debug(f"Raw response from Groq: {response['content']}")

        response_content = response['content']
        sql_match = re.search(r'(SELECT|Không tạo được câu SQL:.*)', response_content, re.DOTALL)
        if not sql_match:
            logger.error(f"No SQL query or error message found in response: {response_content}")
            return "Không tạo được câu SQL: không tìm thấy câu truy vấn hợp lệ"

        cleaned_response = sql_match.group(0)
        # Chuẩn hóa SQL: loại bỏ line breaks, tabs, và chuẩn hóa ILIKE
        cleaned_response = re.sub(r'[\n\r\t]', ' ', cleaned_response).strip()
        # Sửa lỗi % thừa trong ILIKE
        cleaned_response = re.sub(r'ILIKE\s*\'%%([^%]+)%%\'', r"ILIKE '%\1%'", cleaned_response)
        # Sửa lỗi escape dấu nháy đơn
        cleaned_response = re.sub(r'\\\'', "''", cleaned_response)
        # Thay >= và <= bằng BETWEEN cho date range
        cleaned_response = re.sub(r"sp\.date\s*>=\s*'(\d{4}-\d{2}-\d{2})'\s*AND\s*sp\.date\s*<=\s*'(\d{4}-\d{2}-\d{2})'", r"sp.date BETWEEN '\1' AND '\2'", cleaned_response)

        # Validate SQL syntax
        if 'ILIKE' in cleaned_response and '%%' in cleaned_response:
            logger.error(f"Invalid ILIKE pattern in SQL: {cleaned_response}")
            return "Không tạo được câu SQL: lỗi cú pháp SQL trong ILIKE"
        if '>= ' in cleaned_response or '<= ' in cleaned_response:
            logger.error(f"Invalid date range syntax, must use BETWEEN: {cleaned_response}")
            return "Không tạo được câu SQL: phải dùng BETWEEN cho khoảng ngày"
        if cleaned_response.startswith('SELECT') and not cleaned_response.endswith(';'):
            cleaned_response += ';'
        if 'WHERE' in cleaned_response and cleaned_response.endswith('AND'):
            logger.error(f"Invalid trailing AND in SQL: {cleaned_response}")
            return "Không tạo được câu SQL: lỗi cú pháp SQL"

        logger.info(f"Cleaned SQL response: {cleaned_response}")
        return cleaned_response
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}")
        return f"Không tạo được câu SQL: {str(e)}"