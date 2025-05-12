import os
import sys
from pathlib import Path
import time
import json
import re
from phi.model.groq import Groq
from config.env import Groq_API_KEY, GROQ_MODEL
from utils.logging import setup_logging

logger = setup_logging()

class RateLimiter:
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.interval = 60.0 / requests_per_minute
        self.last_request = 0.0

    def wait(self):
        now = time.time()
        elapsed = now - self.last_request
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self.last_request = time.time()

class TextToSQLAgent:
    def __init__(self):
        self.rate_limiter = RateLimiter(requests_per_minute=3)
        self.model = Groq(
            id=GROQ_MODEL,
            api_key=Groq_API_KEY,
            client_params={"timeout": 30, "max_retries": 3}
        )

    def run(self, query: str) -> str:
        """Hàm tùy chỉnh để phân tích sub-query và thực thi truy vấn SQL."""
        logger.info(f"Processing sub-query: {query}")

        # Schema của cơ sở dữ liệu
        database_schema = """
        The database contains only the following tables:

        1. companies
           - symbol (VARCHAR(10), PRIMARY KEY): Stock ticker symbol (e.g., 'AAPL')
           - name (VARCHAR(255)): Company name (e.g., 'Apple Inc.')
           - sector (VARCHAR(100)): Industry sector (e.g., 'Technology')
           - industry (VARCHAR(100)): Specific industry (e.g., 'Consumer Electronics')
           - country (VARCHAR(100)): Country of headquarters (e.g., 'United States')
           - website (VARCHAR(255)): Company website (e.g., 'www.apple.com')
           - market_cap (BIGINT): Market capitalization in USD
           - pe_ratio (DECIMAL(10,2)): Price-to-earnings ratio
           - dividend_yield (DECIMAL(5,2)): Dividend yield percentage
           - week_high_52 (DECIMAL(10,2)): 52-week high stock price
           - week_low_52 (DECIMAL(10,2)): 52-week low stock price
           - description (TEXT): Company description

        2. stock_prices
           - id (SERIAL, PRIMARY KEY): Unique identifier
           - symbol (VARCHAR(10), FOREIGN KEY referencing companies.symbol): Stock ticker symbol
           - date (DATE): Date of the stock price
           - open_price (DECIMAL(10,2)): Opening price
           - high_price (DECIMAL(10,2)): Highest price of the day
           - low_price (DECIMAL(10,2)): Lowest price of the day
           - close_price (DECIMAL(10,2)): Closing price
           - volume (BIGINT): Trading volume
           - dividends (DECIMAL(10,2)): Dividends paid
           - stock_splits (DECIMAL(10,2)): Stock split ratio

        Only generate SQL queries using these tables and their columns. Do not use any other tables or columns not listed above.
        Use 'symbol' to join 'companies' and 'stock_prices' tables.
        """

        # Bước 1: Phân tích sub-query và trả về JSON chứa tables và query
        analysis_prompt = f"""
        {database_schema}

        Analyze the following sub-query: "{query}"
        - Determine which tables from the schema ('companies', 'stock_prices') are needed to answer the query.
        - Generate a SQL query to answer the sub-query.
        - Always join the 'companies' and 'stock_prices' tables using 'symbol' to map company names to their ticker symbols.
        - For stock price queries, fetch the most recent record by ordering by date DESC and limiting to 1.
        - If the query mentions 'Apple', assume it refers to the company 'Apple Inc.' and use this in the WHERE clause to find the corresponding symbol (e.g., 'AAPL').
        - Use the correct column 'close_price' for stock prices. Do not use 'current_price' or 'price'.
        - Do not use any column named 'company'; instead, use 'name' from the 'companies' table to identify the company and join with 'stock_prices' using 'symbol'.
        - Ensure the SQL query uses proper syntax for conditions (e.g., use '=' for comparisons, not just a value like 'symbol 'AAPL'').
        - Return a JSON object with the following structure:
        {{
            "status": "success" or "error",
            "message": "A brief message about the analysis",
            "data": {{
                "tables": A list of table names used in the query,
                "query": The generated SQL query as a string
            }}
        }}
        - If the sub-query cannot be analyzed or a valid SQL query cannot be generated, return a JSON object with status 'error'.
        Example success response:
        {{
            "status": "success",
            "message": "SQL query generated successfully",
            "data": {{
                "tables": ["companies", "stock_prices"],
                "query": "SELECT s.close_price, s.date, s.open_price, s.high_price, s.low_price, s.volume FROM stock_prices s JOIN companies c ON s.symbol = c.symbol WHERE c.name = 'Apple Inc.' ORDER BY s.date DESC LIMIT 1"
            }}
        }}
        Example error response:
        {{
            "status": "error",
            "message": "Unable to generate SQL query: invalid sub-query",
            "data": {{
                "tables": [],
                "query": ""
            }}
        }}
        """
        logger.info("Sending analysis prompt to Grok model")
        analysis_response = self.model.response(messages=[{"role": "user", "content": analysis_prompt}])
        logger.info(f"Received analysis response: {analysis_response['content']}")
        
        # Trích xuất JSON từ phản hồi
        try:
            sql_plan = json.loads(analysis_response['content'])
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from analysis: {analysis_response['content']}, error: {str(e)}")
            return json.dumps({
                "status": "error",
                "message": "Could not analyze sub-query.",
                "data": None
            })

        # Ghi log SQL plan
        logger.info(f"SQL Plan: {json.dumps(sql_plan)}")

        # Kiểm tra status của sql_plan
        if sql_plan.get("status") != "success":
            logger.warning(f"SQL analysis failed: {sql_plan.get('message', 'Unknown error')}")
            return json.dumps({
                "status": "error",
                "message": sql_plan.get("message", "SQL analysis failed"),
                "data": None
            })

        # Bước 2: Thực thi truy vấn SQL
        sql_query = sql_plan.get("data", {}).get("query", "")
        if not sql_query:
            logger.error("No SQL query generated.")
            return json.dumps({
                "status": "error",
                "message": "No SQL query generated.",
                "data": None
            })

        from tools.sql_tool import CustomSQLTool
        sql_tool = CustomSQLTool()
        logger.info(f"Executing SQL query: {sql_query}")
        result_json = sql_tool.run(sql_query)
        
        # Phân tích JSON từ tool
        try:
            result = json.loads(result_json)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from SQL tool: {result_json}, error: {str(e)}")
            return json.dumps({
                "status": "error",
                "message": "Invalid response from SQL tool.",
                "data": None
            })

        logger.info(f"SQL execution result: {result_json}")

        # Nếu có lỗi, trả về JSON lỗi
        if result["status"] == "error":
            return json.dumps(result)

        # Trả về JSON với kết quả
        return json.dumps({
            "status": "success",
            "message": "Query executed successfully",
            "data": result["data"],
            "source": "database"
        })