# agents/text_to_sql_agent.py
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
    """Load database schema and visualized templates."""
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

    vis_template_file = BASE_DIR / "config" / "visualized_template.yml"
    try:
        with open(vis_template_file, "r") as file:
            vis_template = yaml.safe_load(file)
        logger.info("Successfully loaded visualized_template.yml")
        metadata["visualized_template"] = vis_template["visualized_template"]
    except FileNotFoundError:
        logger.error("visualized_template.yml not found")
        metadata["visualized_template"] = []
    except Exception as e:
        logger.error(f"Error loading visualized_template: {str(e)}")
        metadata["visualized_template"] = []

    return metadata

metadata = load_metadata()
SCHEMA = yaml.dump(
    {k: v for k, v in metadata.items() if k in ["database_description", "tables", "relationships"]},
    default_flow_style=False,
    sort_keys=False
)
vis_template_json = json.dumps(metadata["visualized_template"], ensure_ascii=False, indent=2)

ERROR_MESSAGES = {
    "missing_date": "Cannot generate SQL: missing date information",
    "invalid_query": "Cannot generate SQL: invalid query",
    "missing_template": "Cannot generate SQL: template not found",
    "invalid_template": "Cannot generate SQL: invalid template configuration"
}

def create_text_to_sql_agent() -> Agent:
    """Create Text2SQL Agent to generate valid PostgreSQL queries."""
    logger.info("Creating Text2SQL Agent")

    system_prompt = f"""
You are Text2SQL Agent, generating valid PostgreSQL queries for a financial database. Your task is to select an SQL template from visualized templates based on the query and populate it with provided parameters to produce a syntactically correct query. Return ONLY the SQL query or an error message, no explanations, no markdown.

- **Database Schema**:
{SCHEMA}

- **Visualized Templates**:
{vis_template_json}

- **Generate SQL**:
  1. **Select Template**:
     - Match query with intent_keywords in visualized_template to select the appropriate template (e.g., 'average monthly price' matches 'bar_chart_monthly_price').
     - If no template matches, return error: '{ERROR_MESSAGES["missing_template"]}'

  2. **Extract Metadata**:
     - Use tickers and date_range from metadata.
     - If no tickers, extract company name from query (e.g., 'Apple') and use LIKE clause.
     - If no date_range, default to '2024-01-01' and '2024-12-31'.

  3. **Populate Template**:
     - Replace placeholders in the template (e.g., {{ticker}}, {{start_date}}, {{end_date}}).
     - For tickers, use uppercase (e.g., 'AAPL') and format as a comma-separated list if multiple (e.g., 'AAPL','MSFT').

  4. **Ensure Syntax**:
     - Format the query as a single line with a semicolon at the end.
     - Remove excessive whitespace or newlines.
     - Ensure the query is valid for PostgreSQL.

- **Errors**:
  - Missing date: '{ERROR_MESSAGES["missing_date"]}'
  - Invalid query: '{ERROR_MESSAGES["invalid_query"]}'
  - Missing template: '{ERROR_MESSAGES["missing_template"]}'
  - Invalid template: '{ERROR_MESSAGES["invalid_template"]}'

Examples:
1. Query: 'Create a bar chart of Caterpillar (CAT) average monthly closing price in 2024'
   Metadata: tickers=['CAT'], date_range={{'start_date': '2024-01-01', 'end_date': '2024-12-31'}}
   SQL: SELECT EXTRACT(MONTH FROM date) AS month, AVG(close_price) AS avg_close_price FROM stock_prices WHERE symbol = 'CAT' AND date BETWEEN '2024-01-01' AND '2024-12-31' GROUP BY EXTRACT(MONTH FROM date) ORDER BY month;
2. Query: 'Show company info for Microsoft'
   Metadata: tickers=['MSFT']
   SQL: SELECT symbol, name, sector, market_cap FROM companies WHERE symbol = 'MSFT';
"""
    return Agent(
        model=Groq(
            id=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            timeout=30,
            max_retries=5,
            temperature=0.6,
            max_tokens=500,
            top_p=0.8
        ),
        system_prompt=system_prompt,
        custom_run=lambda self, sub_query, metadata=None: self.run_with_fallback(sub_query, metadata),
        debug_mode=True,
    )

def run_with_fallback(self, sub_query: str, metadata: dict = None) -> str:
    logger.info(f"Received sub_query: {sub_query}, metadata: {metadata}")
    try:
        templates = metadata.get('visualized_template', [])
        tickers = metadata.get('tickers', [])
        date_range = metadata.get('date_range', None)

        # Select template based on query
        template = None
        query_lower = sub_query.lower()
        for t in templates:
            for keyword in t.get('intent_keywords', []):
                if keyword in query_lower:
                    template = t
                    break
            if template:
                break

        if not template:
            logger.error("No template found for query")
            return ERROR_MESSAGES["missing_template"]

        if 'sql' not in template:
            logger.error(f"Invalid template configuration for {template['name']}")
            return ERROR_MESSAGES["invalid_template"]

        params = {
            'ticker': tickers[0].upper() if tickers else '',
            'tickers': ','.join(f"'{t.upper()}'" for t in tickers) if tickers else '',
            'start_date': date_range['start_date'] if date_range else '2024-01-01',
            'end_date': date_range['end_date'] if date_range else '2024-12-31',
        }

        # Handle case where no tickers are provided
        if not tickers and '{ticker}' in template['sql']:
            company_name = query_lower
            company_match = re.search(r'(apple|microsoft|boeing|caterpillar|...)', company_name)
            if company_match:
                params['ticker'] = f"%{company_match.group(0)}%"
                sql_query = template['sql'].replace("symbol = '{ticker}'", "name LIKE '{ticker}'")
            else:
                logger.error("No company name or ticker provided")
                return ERROR_MESSAGES["invalid_query"]
        else:
            sql_query = template['sql'].format(**params)

        # Format SQL: remove excessive whitespace, ensure semicolon
        sql_query = re.sub(r'\s+', ' ', sql_query).strip()
        if not sql_query.endswith(';'):
            sql_query += ';'

        logger.info(f"Generated SQL: {sql_query}")
        return sql_query

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return f"Cannot generate SQL: {str(e)}"