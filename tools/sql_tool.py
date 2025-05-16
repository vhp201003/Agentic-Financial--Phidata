# tools/sql_tool.py
import os
import sys
from pathlib import Path
import json

# Thêm thư mục gốc dự án vào sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

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

    def run(self, query: str) -> str:
        """Run a SQL query on the financial database and return JSON.

        Args:
            query (str): The SQL query to execute.

        Returns:
            str: JSON string with status, message, and data (result as JSON records).
        """
        try:
            with self.engine.connect() as conn:
                result = pd.read_sql_query(text(query), conn)
                result_json = result.to_dict(orient='records') if not result.empty else []
                return json.dumps({
                    "status": "success",
                    "message": "Query executed successfully",
                    "data": {
                        "result": result_json
                    }
                }, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return json.dumps({
                "status": "error",
                "message": f"Error executing query: {str(e)}",
                "data": {}
            }, ensure_ascii=False)