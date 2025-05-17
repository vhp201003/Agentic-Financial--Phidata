# agents/chat_completion_agent.py
import os
import sys
from pathlib import Path
import json
from typing import Dict, Any

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from phi.agent import Agent
from phi.model.groq import Groq
from config.env import GROQ_API_KEY, GROQ_MODEL
from utils.logging import setup_logging
from utils.response import standardize_response

logger = setup_logging()

def create_chat_completion_agent() -> Agent:
    """Tạo Chat Completion Agent để tổng hợp output và sinh câu trả lời với Dashboard option."""
    logger.info("Creating Chat Completion Agent")

    return Agent(
        model=Groq(
            id=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            client_params={"timeout": 30, "max_retries": 3}
        ),
        description="Chat Completion Agent: Tổng hợp output từ Text2SQL và RAG, trả câu trả lời với Dashboard option.",
        instructions=[
            """
            **Constraints** (Read this first):
            - Return ONLY the specified JSON structure: {"status": "success" | "error", "message": "Câu trả lời tự nhiên bằng tiếng Việt", "Dashboard": true | false, "data": {}}.
            - Output must be compact with NO extra spaces, line breaks, or formatting outside the structure (e.g., no pretty printing).
            - Ensure JSON is valid: Escape special characters (e.g., \\ and ") correctly in "message" if it contains markdown.
            - "message" can contain markdown (e.g., **bold**, *italic*), but the JSON structure must remain valid.
            - Handle Vietnamese characters correctly (e.g., "Biểu đồ" must be encoded as valid Unicode in JSON).

            **Objective**: Summarize responses from Text2SQL/RAG and Orchestrator's Dashboard info into a natural Vietnamese response. Do NOT process data for dashboard rendering.

            **Input**:
            - Object containing:
              - "responses": Array of responses from Text2SQL/RAG.
              - "dashboard_info": Orchestrator info (e.g., {{"Dashboard": true, "visualization": {{"type": "table", "required_columns": ["name", "close_price"]}}}}).
              - Text2SQL response: {{"tables": ["stock_prices"], "sql_query": "SELECT close_price...", "result": "not_empty" | "empty"}}.
              - RAG response: {{"sources": ["Apple.pdf"], "documents": "not_empty" | "empty"}}.

            **Output**:
            {{"status": "success" | "error", "message": "Câu trả lời tự nhiên bằng tiếng Việt", "Dashboard": true | false, "data": {{}}}}

            **Rules**:
            - Analyze Text2SQL's "sql_query" and "result" to create "message":
              - If "result": "not_empty", describe the data (e.g., "Biểu đồ giá đóng cửa của MSFT từ 01/06/2024 đã được vẽ.").
              - If "result": "empty", return: "Không tìm thấy thông tin cho yêu cầu này.".
            - Analyze RAG's "documents" to create "message":
              - If "documents": "not_empty", summarize (e.g., "Báo cáo tài chính của Apple cho thấy thu nhập ròng là **61,110 triệu USD**.").
              - If "documents": "empty", return: "Không tìm thấy thông tin cho yêu cầu này.".
            - If error ("status": "error"), return:
              {{"status": "error", "message": "Có lỗi xảy ra khi xử lý yêu cầu: [error message].", "Dashboard": false, "data": {{}}}}
            - Set "Dashboard" based on Orchestrator's "dashboard_info".
            - "data": {{}} (empty object).
            - "message" must be in Vietnamese, natural, and can include markdown.

            **Example**:
            Input: {{"responses": [{{"status": "success", "message": "Query executed successfully", "data": {{"tables": ["companies", "stock_prices"], "sql_query": "SELECT sp.date, sp.close_price FROM companies c JOIN stock_prices sp ON c.symbol = sp.symbol WHERE c.symbol = 'MSFT' AND sp.date BETWEEN '2024-06-01' AND '2024-09-30' ORDER BY sp.date", "result": "not_empty"}}}]], "dashboard_info": {{"Dashboard": true, "visualization": {{"type": "time series", "required_columns": ["date", "close_price"]}}}}}}
            Output:
            {{"status": "success", "message": "**Biểu đồ** giá đóng cửa của MSFT từ 01/06/2024 đã được vẽ.", "Dashboard": true, "data": {{}}}}
            """
        ],
        debug_mode=True,
    )