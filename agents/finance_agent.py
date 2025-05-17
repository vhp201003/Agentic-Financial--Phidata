# agents/finance_agent.py
import os
import sys
from pathlib import Path
import json

# Thêm thư mục gốc dự án vào sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from phi.agent import Agent
from phi.model.groq import Groq
from config.env import GROQ_API_KEY, GROQ_MODEL
from utils.logging import setup_logging

logger = setup_logging()

def create_finance_agent() -> Agent:
    logger.info("Creating Finance Agent")

    def custom_run(self, query: str) -> str:
        """Hàm tùy chỉnh để xử lý truy vấn tài chính và trả về JSON."""
        logger.info(f"Finance Agent processing query: {query}")
        try:
            response = self.model.response(messages=[{"role": "user", "content": query}])
            return json.dumps({
                "status": "success",
                "message": "Query processed successfully",
                "data": {
                    "result": response['content']
                }
            })
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return json.dumps({
                "status": "error",
                "message": f"Error processing query: {str(e)}",
                "data": {}
            })

    return Agent(
        model=Groq(
            id=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            client_params={"timeout": 30, "max_retries": 3}
        ),
        description="You are a Finance Agent that provides financial insights.",
        instructions=["Provide financial insights based on user queries."],
        custom_run=custom_run,
        debug_mode=False
    )