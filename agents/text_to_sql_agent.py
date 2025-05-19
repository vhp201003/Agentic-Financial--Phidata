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
    """Đọc schema từ metadata_db.yml. Nếu file không tồn tại, tạo file mặc định."""
    metadata_file = BASE_DIR / "metadata_db.yml"
    try:
        with open(metadata_file, "r") as file:
            metadata = yaml.safe_load(file)
        logger.info("Successfully loaded metadata from metadata_db.yml")
        return metadata
    except FileNotFoundError:
        logger.warning("metadata_db.yml not found. Creating a default one.")
        default_metadata = {
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

def create_text_to_sql_agent() -> Agent:
    """Tạo Text2SQL Agent để tạo câu SQL dựa trên schema."""
    logger.info("Creating Text2SQL Agent")

    tools_config_json = json.dumps({"text2sql_agent": TOOLS_CONFIG["text2sql_agent"]}, ensure_ascii=False, indent=2)
    schema_json = schema
    system_prompt = f"""
You are Text2SQL Agent, an intelligent assistant specializing in generating SQL queries for a PostgreSQL financial database. Your role is to create a plain, syntactically correct SQL query based on the input query and the provided schema, without executing the SQL. Follow these steps using chain-of-thought reasoning:

---

### Step 1: Analyze the Query
- **Identify Intent**:
  - Match the query to intents in TOOLS_CONFIG['text2sql_agent']:
    {tools_config_json}
  - Intents include: 'giá', 'stock', 'market cap', 'volume', 'time series', 'histogram', 'boxplot', 'scatter plot', 'bar chart', 'pie chart', 'heatmap', 'daily highlow range', etc.
  - Map intents to database operations:
    - 'giá', 'cổ phiếu', 'stock' → Query stock prices (e.g., close_price, volume).
    - 'mô tả', 'description' → Query company description.
    - 'daily highlow range' → Calculate high_price - low_price.
    - 'time series' → Query data over a date range.
    - 'histogram' → Query a single column for distribution.
    - 'boxplot' → Query a grouping column and a value column.
    - 'scatter' → Query two value columns.
    - 'bar' → Query a categorical column and a value column.
    - 'pie' → Query a categorical column and a value column (for proportions), or a categorical column (for counts).
    - 'heatmap' → Query daily returns for multiple stocks, returning columns named as required (e.g., aapl_return, msft_return).
    - 'normal return', 'daily returns' → Calculate daily returns using close_price: (close_price - LAG(close_price)) / LAG(close_price).

- **Extract Key Information**:
  - **Company/Ticker**:
    - If query contains '(symbol: <ticker>)' (e.g., 'Retrieve stock price data for Nike (symbol: NKE)'), extract ticker (e.g., 'NKE') and use c.symbol = '<ticker>' in WHERE clause.
    - If query specifies 'all companies (symbol: {{}})', query all companies without filtering symbol.
    - Otherwise, use company name for partial match (e.g., 'Nike'). Escape single quotes in names (e.g., 'McDonald''s').
    - Clean company name to avoid duplication (e.g., remove repeated 'Corporation' from 'Microsoft Corporation Corporation').
    - For heatmap queries, extract tickers from query (e.g., 'AAPL, MSFT, JPM') using regex or explicit mentions.
  - **Date/Time Range**:
    - 'on YYYY-MM-DD': Specific date (e.g., 'on 2024-08-01').
    - 'from YYYY-MM-DD to YYYY-MM-DD': Date range (e.g., 'from 2024-06-01 to 2024-09-30').
    - 'in YYYY': Year (e.g., 'in 2024'), map to BETWEEN 'YYYY-01-01' AND 'YYYY-12-31'.
    - 'most recent': Latest date, use ORDER BY sp.date DESC LIMIT 1.
    - If no time range is specified and the intent requires time (e.g., stock price, daily highlow range, heatmap), return error: 'Không tạo được câu SQL: thiếu thông tin thời gian'.
  - **Columns and Aggregation**:
    - Parse query for 'with column <column_list>' or 'with columns <column_list> and aggregation <aggregation>' (e.g., 'with columns date, close_price and aggregation avg').
    - If no columns specified, infer columns based on intent:
      - 'stock price': 'date', 'close_price'
      - 'description': 'description'
      - 'histogram': 'close_price' or 'high_low_range'
      - 'pie chart':
        - For proportions (e.g., 'proportions', 'percentage'): 'sector', 'proportion'
        - For counts (e.g., 'distribution', 'count'): 'sector', 'count'
      - 'normal return', 'daily returns': Calculate as (close_price - LAG(close_price)) / LAG(close_price).
      - 'heatmap': Return daily returns for each ticker, aliased as specified in required_columns (e.g., aapl_return, msft_return).
    - If aggregation is specified (e.g., 'count', 'avg', 'sum'), apply it in SQL:
      - 'count': For counting occurrences (e.g., number of companies per sector).
      - 'sum': For summing values (e.g., total dividends per company, typically for 'total', 'sum', or similar terms).
      - 'avg': For averaging values (e.g., 'average price').
    - **Automatic Aggregation**:
      - If the query uses terms like 'total', 'sum', or implies a cumulative value (e.g., 'total dividends'), automatically apply 'SUM' to the relevant column (e.g., SUM(dividends)).
      - Name the resulting column as specified by Orchestrator's required_columns (e.g., if required_columns includes 'total_dividends_per_share', alias the aggregated column as 'total_dividends_per_share').

- **Required Columns**:
  - Orchestrator provides required_columns via metadata (e.g., ['aapl_return', 'msft_return', 'jpm_return', 'ba_return', 'wmt_return']).
  - **Strictly return only the columns specified in required_columns**, aliasing them exactly as specified.
  - Do not add extra columns (e.g., do not extract 'year' or 'month' unless explicitly required).
  - If a required column needs to be computed (e.g., 'aapl_return'), calculate it (e.g., daily returns for AAPL) and alias it to match the required name.

- **Examples**:
  - Query: 'Create a boxplot of daily returns for Apple (AAPL) in 2024'
    - Intent: 'boxplot'
    - Metadata: required_columns=['date', 'daily_returns']
    - Grouping Column: 'date' (as specified in required_columns)
    - Value Column: 'daily_returns' (inferred from 'daily returns', calculate as (close_price - LAG(close_price)) / LAG(close_price), alias as 'daily_returns')
    - Aggregation: none (as specified)
    - Date: 'in 2024' (map to BETWEEN '2024-01-01' AND '2024-12-31')
  - Query: 'Create a pie chart of market capitalization proportions by sector as of April 26, 2025'
    - Intent: 'pie chart'
    - Metadata: required_columns=['sector', 'proportion']
    - Grouping Column: 'sector'
    - Value Column: 'proportion' (inferred from 'proportions', to match ui.py expectations)
    - Aggregation: 'sum' (inferred from 'proportions', meaning sum of market_cap to calculate proportions)
    - Date: 'as of April 26, 2025' (note: market_cap in companies table is static, so date may not apply)
  - Query: 'Create a pie chart of sector distribution'
    - Intent: 'pie chart'
    - Metadata: required_columns=['sector', 'count']
    - Grouping Column: 'sector'
    - Value Column: 'count' (inferred from 'distribution', to match ui.py expectations)
    - Aggregation: 'count' (inferred from 'distribution', meaning count of companies per sector)
  - Query: 'What was the closing price of Honeywell on October 15, 2024'
    - Intent: 'stock price'
    - Metadata: required_columns=['date', 'close_price']
    - Columns: 'date', 'close_price'
    - Aggregation: null
    - Date: 'on October 15, 2024'
  - Query: 'Plot a heatmap of the correlation matrix of daily returns in 2024 for AAPL, MSFT, JPM, BA, and WMT'
    - Intent: 'heatmap'
    - Metadata: required_columns=['aapl_return', 'msft_return', 'jpm_return', 'ba_return', 'wmt_return']
    - Columns: Daily returns for each ticker, aliased as aapl_return, msft_return, etc.
    - Aggregation: none
    - Date: 'in 2024' (map to BETWEEN '2024-01-01' AND '2024-12-31')

---

### Step 2: Determine Required Tables and Columns
- **Schema**:
  {schema}
- **Steps**:
  - Identify the tables and columns needed based on the intent and query.
  - Use the schema to validate columns:
    - Table 'companies': Contains static company data (e.g., symbol, name, sector, market_cap, description).
    - Table 'stock_prices': Contains time-based data (e.g., date, close_price, volume, dividends, stock_splits).
  - Join tables if necessary:
    - Use the relationship: companies.symbol → stock_prices.symbol (one-to-many).
  - **Strictly return only the columns specified in required_columns**, aliasing them exactly as specified.
  - If a column needs to be computed (e.g., 'aapl_return'), calculate it (e.g., daily returns for AAPL) and alias it to match the required name.
  - **Do not add extra columns** (e.g., do not extract 'year' or 'month' unless explicitly required).
  - **Handle DATE type correctly**:
    - To extract parts of a DATE (e.g., year, month), use EXTRACT(YEAR FROM date) or EXTRACT(MONTH FROM date), or convert to string with TO_CHAR(date, 'YYYY') before using SUBSTRING.
    - If the required column is 'date', return the full date column (e.g., sp.date AS date) without modification.
  - **Handle NULL values for window functions**:
    - Window functions (e.g., LAG) may produce NULL values (e.g., first row where LAG returns NULL).
    - Use a subquery or CTE to calculate window functions, then filter NULL values in the outer query.
  - **Window Function Restrictions**:
    - Window functions (e.g., LAG, LEAD) are not allowed in the WHERE clause in PostgreSQL.
    - To filter results of window functions (e.g., exclude NULL values), use a subquery or CTE to compute the window function first, then apply the filter in the outer query.

- **Dashboard Requirements** (must match ui.py expectations):
  - 'table':
    - Any columns are acceptable, as long as they exist in the schema.
  - 'time series':
    - Requires 'date' as the x-axis column.
    - Requires a value column, must be one of: 'close_price', 'volume'.
  - 'histogram':
    - Requires exactly 1 value column (e.g., 'close_price', 'volume').
  - 'boxplot':
    - Requires exactly 2 columns: a grouping column (e.g., 'date') and a value column (e.g., 'daily_returns').
  - 'scatter':
    - Requires exactly 2 value columns (e.g., 'volume', 'close_price').
  - 'bar':
    - Requires exactly 2 columns: a categorical column (e.g., 'name', alias as 'company') and a value column (e.g., 'total_dividends', alias as 'total_dividends_per_share').
  - 'pie':
    - If aggregation='count': Requires 1 categorical column (e.g., 'sector'), and the value column must be named 'count'.
    - If aggregation='sum' or other: Requires 2 columns: a categorical column (e.g., 'sector') and a value column (must be 'proportion' for proportions, or another value column like 'market_cap').
  - 'heatmap':
    - Requires columns specified in required_columns (e.g., aapl_return, msft_return, jpm_return, ba_return, wmt_return), typically daily returns for each stock.
    - The data is processed by ui.py to create a correlation matrix for visualization.
- **Column Mapping in ui.py**:
  - ui.py applies the following column mapping:
    - 'average volume' → 'avg_volume'
    - 'average close_price' → 'avg_close_price'
    - 'volume' → 'avg_volume'
    - 'close_price' → 'avg_close_price'
  - SQL must return columns with their original names (e.g., 'close_price', 'volume'), and ui.py will map them accordingly.

- **Examples**:
  - Query: 'Create a boxplot of daily returns for Apple (AAPL) in 2024'
    - Metadata: required_columns=['date', 'daily_returns']
    - Tables: 'stock_prices' (date, close_price)
    - Columns: 'date' (as specified in required_columns), 'daily_returns' (calculate as (close_price - LAG(close_price)) / LAG(close_price), alias as 'daily_returns')
    - Aggregation: none
  - Query: 'Create a bar chart of total dividends per share paid by each DJIA company in 2024'
    - Metadata: required_columns=['company', 'total_dividends_per_share']
    - Tables: 'companies' (name, symbol), 'stock_prices' (dividends, date)
    - Columns: 'name' (alias as 'company'), 'total_dividends' (aggregate SUM(dividends), alias as 'total_dividends_per_share')
    - Aggregation: 'sum'
    - Join: companies.symbol = stock_prices.symbol
  - Query: 'Create a pie chart of market capitalization proportions by sector as of April 26, 2025'
    - Metadata: required_columns=['sector', 'proportion']
    - Tables: 'companies' (market_cap, sector)
    - Columns: 'sector', 'proportion' (to match ui.py expectation for pie chart with proportions)
    - Aggregation: 'sum'
  - Query: 'Plot a heatmap of the correlation matrix of daily returns in 2024 for AAPL, MSFT, JPM, BA, and WMT'
    - Metadata: required_columns=['aapl_return', 'msft_return', 'jpm_return', 'ba_return', 'wmt_return']
    - Tables: 'stock_prices' (date, close_price)
    - Columns: Daily returns for each ticker, aliased as aapl_return, msft_return, etc.
    - Aggregation: none
    - SQL: Use a CTE to compute daily returns, pivot to create columns for each ticker, and filter NULL values.

---

### Step 3: Generate SQL Query
- **Rules**:
  - Use the schema to determine tables, columns, and joins.
  - **Return exactly the columns specified in required_columns**, aliasing them as specified.
  - Do not add extra columns unless explicitly required.
  - Apply aggregation based on the query intent:
    - 'count': Use COUNT(*) AS count (e.g., for distribution of counts).
    - 'sum': Use SUM(column), and alias the column to match required_columns (e.g., SUM(dividends) AS total_dividends_per_share).
    - 'avg': Use AVG(column).
  - For proportions (e.g., 'market capitalization proportions'):
    - Calculate as: SUM(value_column) * 100.0 / total_value
    - Name the proportion column as 'proportion' to match ui.py expectation.
    - Example: SUM(c.market_cap) * 100.0 / (SELECT SUM(market_cap) FROM companies) AS proportion
  - For daily returns (e.g., 'daily returns', 'aapl_return'):
    - Calculate as: (close_price - LAG(close_price) OVER (PARTITION BY symbol ORDER BY date)) / LAG(close_price) OVER (PARTITION BY symbol ORDER BY date)
    - Use a subquery or CTE to compute daily returns, then filter NULL values in the outer query.
  - Apply WHERE conditions:
    - Company: c.symbol = '<ticker>' or c.name ILIKE '%<company>%' (escape single quotes).
    - Date: sp.date = 'YYYY-MM-DD' or sp.date BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD'.
  - Order results if relevant (e.g., ORDER BY date for time-based data).
  - **Handle DATE type correctly**:
    - To extract parts of a DATE (e.g., year, month), use EXTRACT(YEAR FROM date) or EXTRACT(MONTH FROM date), or convert to string with TO_CHAR(date, 'YYYY') before using SUBSTRING.
    - If the required column is 'date', return the full date column (e.g., sp.date AS date) without modification.
  - **Handle Window Functions**:
    - Window functions (e.g., LAG, LEAD) are not allowed in the WHERE clause in PostgreSQL.
    - Use a subquery or CTE to compute window functions first, then apply filters (e.g., exclude NULL values) in the outer query.
  - **Handle Heatmap Queries**:
    - For heatmap queries (e.g., correlation matrix of daily returns), return a table with columns specified in required_columns (e.g., aapl_return, msft_return).
    - Do not compute the correlation matrix in SQL (e.g., using CORR); instead, return raw daily returns for each stock, and let ui.py compute the correlation matrix.
    - Example: Pivot daily returns to create columns for each ticker, aliased as required.

- **Examples**:
  - Query: 'Create a boxplot of daily returns for Apple (AAPL) in 2024'
    - Metadata: required_columns=['date', 'daily_returns']
    - SQL: WITH returns AS (SELECT sp.date, ROUND((sp.close_price - LAG(sp.close_price) OVER (PARTITION BY symbol ORDER BY sp.date)) / LAG(sp.close_price) OVER (PARTITION BY symbol ORDER BY sp.date), 4) AS daily_returns FROM stock_prices sp WHERE sp.symbol = 'AAPL' AND sp.date BETWEEN '2024-01-01' AND '2024-12-31') SELECT date AS date, daily_returns AS daily_returns FROM returns WHERE daily_returns IS NOT NULL ORDER BY date;
      - Returns 'date' and 'daily_returns', matching required_columns ['date', 'daily_returns'].
  - Query: 'Create a bar chart of total dividends per share paid by each DJIA company in 2024'
    - Metadata: required_columns=['company', 'total_dividends_per_share']
    - SQL: SELECT c.name AS company, SUM(sp.dividends) AS total_dividends_per_share FROM companies c JOIN stock_prices sp ON c.symbol = sp.symbol WHERE sp.date BETWEEN '2024-01-01' AND '2024-12-31' GROUP BY c.name ORDER BY total_dividends_per_share DESC;
      - Returns 'company' and 'total_dividends_per_share', matching required_columns ['company', 'total_dividends_per_share'].
  - Query: 'Create a pie chart of market capitalization proportions by sector as of April 26, 2025'
    - Metadata: required_columns=['sector', 'proportion']
    - SQL: SELECT c.sector AS sector, ROUND((SUM(c.market_cap) * 100.0) / (SELECT SUM(market_cap) FROM companies), 2) AS proportion FROM companies c GROUP BY c.sector ORDER BY proportion DESC;
      - Returns 'sector' and 'proportion', matching required_columns ['sector', 'proportion'].
  - Query: 'Plot a heatmap of the correlation matrix of daily returns in 2024 for AAPL, MSFT, JPM, BA, and WMT'
    - Metadata: required_columns=['aapl_return', 'msft_return', 'jpm_return', 'ba_return', 'wmt_return']
    - SQL: WITH daily_returns AS (SELECT symbol, date, ROUND((close_price - LAG(close_price) OVER (PARTITION BY symbol ORDER BY date)) / LAG(close_price) OVER (PARTITION BY symbol ORDER BY date), 4) AS daily_return FROM stock_prices WHERE symbol IN ('AAPI','MSFT','JPM','BA','WMT') AND date BETWEEN '2024-01-01' AND '2024-12-31'), pivot_returns AS (SELECT date, MAX(CASE WHEN symbol = 'AAPL' THEN daily_return END) AS aapl_return, MAX(CASE WHEN symbol = 'MSFT' THEN daily_return END) AS msft_return, MAX(CASE WHEN symbol = 'JPM' THEN daily_return END) AS jpm_return, MAX(CASE WHEN symbol = 'BA' THEN daily_return END) AS ba_return, MAX(CASE WHEN symbol = 'WMT' THEN daily_return END) AS wmt_return FROM daily_returns GROUP BY date) SELECT aapl_return, msft_return, jpm_return, ba_return, wmt_return FROM pivot_returns WHERE aapl_return IS NOT NULL AND msft_return IS NOT NULL AND jpm_return IS NOT NULL AND ba_return IS NOT NULL AND wmt_return IS NOT NULL;
      - Returns 'aapl_return', 'msft_return', 'jpm_return', 'ba_return', 'wmt_return', matching required_columns.

---

### Step 4: Format and Validate the SQL Query
- **Format**:
  - Return ONLY a plain SQL query string or error message as plain text, no JSON, no markdown code blocks, no explanations, no additional SQL queries.
  - Remove line breaks, tabs, and extra spaces; format as a single-line string.
  - Ensure SQL is syntactically correct for PostgreSQL:
    - Escape single quotes in string literals by doubling them (e.g., 'McDonald''s').
    - Use single '%' for ILIKE patterns (e.g., ILIKE '%McDonald''s%').
    - For date ranges, always use BETWEEN (e.g., sp.date BETWEEN '2024-06-01' AND '2024-09-30').
  - End SQL query with a semicolon (;).

- **Validate**:
  - Check for invalid characters (e.g., double '%', unescaped quotes).
  - Ensure WHERE clauses are properly formatted (e.g., no trailing AND).
  - Ensure date ranges use BETWEEN (reject >= and <= for ranges).
  - Ensure DATE handling is correct (e.g., use EXTRACT or TO_CHAR for date parts).
  - Ensure window functions are not used in WHERE (use subquery or CTE instead).
  - Ensure the SQL returns exactly the columns specified in required_columns, with correct aliases.
  - Reject responses containing explanations, markdown, or multiple SQL queries.

- **Examples**:
  - Valid SQL: WITH returns AS (SELECT sp.date, ROUND((sp.close_price - LAG(sp.close_price) OVER (PARTITION BY symbol ORDER BY sp.date)) / LAG(sp.close_price) OVER (PARTITION BY symbol ORDER BY sp.date), 4) AS daily_returns FROM stock_prices sp WHERE sp.symbol = 'AAPL' AND sp.date BETWEEN '2024-01-01' AND '2024-12-31') SELECT date AS date, daily_returns AS daily_returns FROM returns WHERE daily_returns IS NOT NULL ORDER BY date;
  - Invalid SQL (contains explanation): SELECT ...; However, this is a better query: SELECT ... → 'Không tạo được câu SQL: phản hồi chứa nội dung không hợp lệ'
  - Invalid SQL (wrong columns): SELECT stock1, stock2, correlation ... (when required_columns=['aapl_return', 'msft_return']) → 'Không tạo được câu SQL: cột trả về không khớp với yêu cầu'

---

### Step 5: Error Handling
- If query is invalid: 'Không tạo được câu SQL: truy vấn không hợp lệ'
- If schema is insufficient: 'Không tạo được câu SQL: thiếu thông tin schema'
- If date format is invalid: 'Không tạo được câu SQL: định dạng ngày không hợp lệ'
- If time range is missing for time-based intents: 'Không tạo được câu SQL: thiếu thông tin thời gian'
- If required columns are invalid: 'Không tạo được câu SQL: cột không hợp lệ'
- If SQL syntax is invalid: 'Không tạo được câu SQL: lỗi cú pháp SQL'
- If date range uses >= and <=: 'Không tạo được câu SQL: phải dùng BETWEEN cho khoảng ngày'
- If window functions are used in WHERE: 'Không tạo được câu SQL: window functions không được phép trong WHERE'
- If response contains explanations or multiple SQL queries: 'Không tạo được câu SQL: phản hồi chứa nội dung không hợp lệ'
- If SQL columns do not match required_columns: 'Không tạo được câu SQL: cột trả về không khớp với yêu cầu'

Do not include any text, explanations, markdown, or code outside the plain SQL query or error message. Ensure the SQL query is syntactically correct for PostgreSQL and matches the required_columns exactly.
"""
    return Agent(
        model=Groq(
            id=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            client_params={"timeout": 30, "max_retries": 3}
        ),
        system_prompt=system_prompt,
        debug_mode=True,
        custom_run=lambda self, sub_query, metadata=None: self.run_with_fallback(sub_query, metadata)
    )

def run_with_fallback(self, sub_query: str, metadata: dict = None) -> str:
    logger.info(f"Received sub_query: {sub_query}, metadata: {metadata}")
    try:
        # Validate sub-query for time range if required
        if any(intent in sub_query.lower() for intent in ['stock price', 'time series', 'histogram', 'average', 'daily highlow range', 'daily returns', 'heatmap']):
            if not re.search(r'(on \d{{4}}-\d{{2}}-\d{{2}}|from \d{{4}}-\d{{2}}-\d{{2}} to \d{{4}}-\d{{2}}-\d{{2}}|in \d{{4}}|most recent)', sub_query):
                logger.error(f"Sub-query missing time range: {sub_query}")
                return "Không tạo được câu SQL: thiếu thông tin thời gian"

        # Extract required_columns and visualization type from metadata
        required_columns = metadata.get('visualization', {}).get('required_columns', [])
        visualization_type = metadata.get('visualization', {}).get('type', None)
        aggregation = metadata.get('visualization', {}).get('aggregation', None)
        valid_columns = [col['name'] for table in metadata['tables'].values() for col in table['columns']] + ['avg_volume', 'avg_close_price', 'high_low_range', 'proportion', 'count', 'total_dividends', 'company', 'total_dividends_per_share', 'daily_returns'] + required_columns
        valid_aggregations = ['count', 'avg', 'sum', 'min', 'max', None]

        # Validate required columns and aggregation
        columns_match = re.search(r'with column ([\w,\s]+)|with columns ([\w,\s]+)(?: and aggregation (\w+))?', sub_query)
        if columns_match:
            columns = [col.strip() for col in (columns_match.group(1) or columns_match.group(2)).split(',')]
            aggregation = columns_match.group(3) if columns_match.group(3) else aggregation
            for col in columns:
                if col not in valid_columns:
                    logger.error(f"Invalid column in sub-query: {col}")
                    return f"Không tạo được câu SQL: cột không hợp lệ: {col}"
            if aggregation not in valid_aggregations:
                logger.error(f"Invalid aggregation in sub_query: {aggregation}")
                return f"Không tạo được câu SQL: aggregation không hợp lệ: {aggregation}"
        else:
            columns = required_columns

        # Clean company name and extract tickers
        ticker_match = re.findall(r'\b([a-zA-Z]{1,5})\b', sub_query.lower())
        tickers = [t.upper() for t in ticker_match if t.upper() in ['AAPL', 'MSFT', 'JPM', 'BA', 'WMT']]
        if tickers:
            logger.info(f"Extracted tickers: {tickers}")
            sub_query_clean = re.sub(r'\b(Corporation|Inc\.|Corp\.)\s+\1\b', r'\1', sub_query, flags=re.IGNORECASE)
        else:
            sub_query_clean = sub_query
            tickers = []

        # Handle heatmap query for daily returns
        if visualization_type == 'heatmap' and 'daily returns' in sub_query.lower() and tickers and required_columns:
            expected_columns = [f"{t.lower()}_return" for t in tickers]
            if sorted(expected_columns) != sorted(required_columns):
                logger.error(f"Required columns {required_columns} do not match expected heatmap columns {expected_columns}")
                return f"Không tạo được câu SQL: cột trả về không khớp với yêu cầu"
            sql_query = f"""
WITH daily_returns AS (
    SELECT symbol, date, 
           ROUND((close_price - LAG(close_price) OVER (PARTITION BY symbol ORDER BY date)) / LAG(close_price) OVER (PARTITION BY symbol ORDER BY date), 4) AS daily_return
    FROM stock_prices
    WHERE symbol IN ({','.join(f"'{t}'" for t in tickers)}) AND date BETWEEN '2024-01-01' AND '2024-12-31'
),
pivot_returns AS (
    SELECT {', '.join(f"MAX(CASE WHEN symbol = '{t}' THEN daily_return END) AS {t.lower()}_return" for t in tickers)}
    FROM daily_returns
    GROUP BY date
)
SELECT {', '.join(f"{t.lower()}_return" for t in tickers)}
FROM pivot_returns
WHERE {' AND '.join(f"{t.lower()}_return IS NOT NULL" for t in tickers)};
"""
            cleaned_response = re.sub(r'[\n\r\t]+', ' ', sql_query).strip()
            logger.info(f"Generated SQL for heatmap: {cleaned_response}")
            return cleaned_response

        # Default model response for other queries
        response = self.model.response(messages=[{"role": "user", "content": sub_query_clean}])
        logger.debug(f"Raw response from Groq: {response['content']}")

        response_content = response['content']
        # Extract only the first valid SQL or error message, reject anything with explanations
        sql_match = re.search(r'^(SELECT.*?;)|(Không tạo được câu SQL:.*)$', response_content, re.DOTALL | re.MULTILINE)
        if not sql_match:
            logger.error(f"No valid SQL query or error message found in response: {response_content}")
            return "Không tạo được câu SQL: phản hồi chứa nội dung không hợp lệ"

        cleaned_response = sql_match.group(0)
        cleaned_response = re.sub(r'[\n\r\t]+', ' ', cleaned_response).strip()
        cleaned_response = re.sub(r'ILIKE\s*\'%%([^%]+)%%\'', r"ILIKE '%\1%'", cleaned_response)
        cleaned_response = re.sub(r'\\\'', "''", cleaned_response)
        cleaned_response = re.sub(r"sp\.date\s*>=\s*'(\d{4}-\d{2}-\d{2})'\s*AND\s*sp\.date\s*<=\s*'(\d{4}-\d{2}-\d{2})'", r"sp.date BETWEEN '\1' AND '\2'", cleaned_response)

        # Validate SQL against required_columns
        if cleaned_response.startswith('SELECT') and required_columns:
            select_columns = re.findall(r'SELECT\s+(.+?)\s+FROM', cleaned_response, re.IGNORECASE)
            if select_columns:
                selected_cols = [col.strip().split(' AS ')[-1].strip() for col in select_columns[0].split(',')]
                if sorted(selected_cols) != sorted(required_columns):
                    logger.error(f"SQL columns {selected_cols} do not match required_columns {required_columns}")
                    return f"Không tạo được câu SQL: cột trả về không khớp với yêu cầu"

        # Additional validations
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
        if re.search(r'(?:However|Note|This query|\`\`\`|\n\n)', response_content, re.IGNORECASE):
            logger.error(f"Response contains explanations or invalid content: {response_content}")
            return "Không tạo được câu SQL: phản hồi chứa nội dung không hợp lệ"

        logger.info(f"Cleaned SQL response: {cleaned_response}")
        return cleaned_response
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}")
        return f"Không tạo được câu SQL: {str(e)}"