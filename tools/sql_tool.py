import os
import sys
from pathlib import Path
from phi.tools import Toolkit
from sqlalchemy import create_engine, text
from config.env import DATABASE_URL
import pandas as pd
from utils.logging import setup_logging
from utils.validators import validate_database_url

logger = setup_logging()

class CustomSQLTool(Toolkit):
    def __init__(self):
        super().__init__(name="sql_tool")
        try:
            validate_database_url(DATABASE_URL)
            self.engine = create_engine(DATABASE_URL)
            self.register(self.run)
            logger.info("SQL tool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SQL tool: {str(e)}")
            raise

    def table_exists(self, table_name: str) -> bool:
        """Kiểm tra xem bảng có tồn tại trong cơ sở dữ liệu không."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = :table_name)"),
                    {"table_name": table_name.lower()}
                )
                return result.scalar()
        except Exception as e:
            logger.error(f"Error checking table existence: {str(e)}")
            return False

    def run(self, query: str) -> str:
        """Run a SQL query on the financial database.

        Args:
            query (str): The SQL query to execute.

        Returns:
            str: The result of the query in markdown table format or error message.
        """
        try:
            # Chỉ cho phép truy vấn các bảng companies và stock_prices
            allowed_tables = ["companies", "stock_prices"]
            query_lower = query.lower()
            invalid_tables = [t for t in ["financial_data", "employees", "customers"] if t in query_lower]
            if invalid_tables or (any(kw in query_lower for kw in ["from", "join"]) and not any(t in query_lower for t in allowed_tables)):
                logger.warning(f"Query references unsupported table(s): {invalid_tables or 'unknown'}")
                return "Error: Query references unsupported table(s). Only 'companies' and 'stock_prices' are allowed."

            # Kiểm tra sự tồn tại của bảng
            for table in allowed_tables:
                if table in query_lower and not self.table_exists(table):
                    logger.warning(f"Table {table} does not exist in the database")
                    return f"Error: Table {table} does not exist in the database."

            with self.engine.connect() as conn:
                result = pd.read_sql_query(text(query), conn)
                return result.to_markdown(index=False)
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return f"Error executing query: {str(e)}"