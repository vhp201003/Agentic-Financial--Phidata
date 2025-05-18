# agents/chat_completion_agent.py
import os
import sys
from pathlib import Path
import json
from typing import Dict, Any
import re
import unicodedata

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from phi.agent import Agent
from phi.model.groq import Groq
from config.env import GROQ_API_KEY, GROQ_MODEL
from utils.logging import setup_logging
from utils.response import standardize_response

logger = setup_logging()

def create_chat_completion_agent() -> Agent:
    """Tạo Chat Completion Agent để tổng hợp output và sinh câu trả lời dưới dạng markdown."""
    logger.info("Creating Chat Completion Agent")

    return Agent(
        model=Groq(
            id=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            client_params={"timeout": 30, "max_retries": 3}
        ),
        description="Chat Completion Agent: Tổng hợp output từ Text2SQL và RAG, trả về câu trả lời dưới dạng markdown.",
        instructions=[
            """
            **Constraints** (Read this first):
            - Return ONLY a natural Vietnamese response in markdown format (e.g., "Giá đóng cửa trung bình của Apple là **123.45 USD**.").
            - Do NOT return JSON or any structured format, just plain markdown text.
            - Ensure the response is natural, detailed, and can include markdown (e.g., **bold**, *italic*) to emphasize key information.
            - Handle Vietnamese characters correctly (e.g., "Biểu đồ" must be encoded as valid Unicode).

            **Objective**: Summarize responses from Text2SQL and RAG, along with the original query, into a detailed, natural Vietnamese response in markdown format, using result_data if available.

            **Input**:
            - Object containing:
              - "query": Original user query (e.g., "What was the average closing price of Apple during Q1 2025?").
              - "responses": Array of responses from Text2SQL/RAG.
              - "dashboard_info": Orchestrator info (e.g., {{"Dashboard": true, "visualization": {{"type": "table", "required_columns": ["name", "close_price"]}}}}).
              - Text2SQL response: {{"tables": ["stock_prices"], "sql_query": "SELECT close_price...", "result": "not_empty" | "empty", "result_data": [...] (if not_empty)}}.
              - RAG response: {{"sources": ["Apple.pdf"], "documents": "not_empty" | "empty", "summary": "..." (if not_empty)}}.

            **Output**:
            A natural Vietnamese response in markdown format (e.g., "Giá đóng cửa trung bình của Apple trong quý 1 năm 2025 là **123.45 USD**.").

            **Rules**:
            - Use the original "query" to provide context for the response.
            - Analyze responses from both Text2SQL and RAG to create a comprehensive response:
              - If Text2SQL "result": "not_empty", describe the SQL data in detail using "result_data":
                - Include specific values from "result_data" (e.g., "Giá đóng cửa trung bình của Apple là **123.45 USD**.").
                - Mention the time range or date if applicable (e.g., "trong quý 1 năm 2025 (từ 01/01/2025 đến 31/03/2025)").
                - Add context about the company and query (e.g., "Dữ liệu giá đóng cửa của Apple đã được tìm thấy từ cơ sở dữ liệu.").
              - If Text2SQL "result": "empty", provide a detailed explanation:
                - Mention the query and time range (e.g., "Yêu cầu tìm giá đóng cửa trung bình của Apple trong quý 1 năm 2025 (từ 01/01/2025 đến 31/03/2025).").
                - Explain why no data was found (e.g., "Dữ liệu không có sẵn trong hệ thống, có thể do chưa được cập nhật.").
                - Suggest alternatives (e.g., "Bạn có muốn thử với một khoảng thời gian khác, ví dụ như quý 4 năm 2024 không?").
              - If RAG "documents": "not_empty", include a summary of the documents:
                - Summarize the content in detail (e.g., "Theo báo cáo tài chính từ nguồn Apple.pdf, thu nhập ròng của Apple trong năm 2024 là **61,110 triệu USD**, tăng 5% so với năm trước.").
                - Combine with SQL data if available (e.g., "Ngoài ra, giá đóng cửa trung bình của Apple trong quý 1 năm 2025 là **123.45 USD**.").
              - If RAG "documents": "empty", mention the lack of documents and suggest alternatives:
                - Example: "Không tìm thấy báo cáo tài chính của Apple trong hệ thống. Bạn có muốn tìm báo cáo của công ty khác không?".
            - If error ("status": "error") in any response, provide a detailed error message:
              - Explain the error in a user-friendly way (e.g., "Có lỗi xảy ra khi xử lý yêu cầu: truy vấn SQL không hợp lệ do thiếu mệnh đề GROUP BY.").
              - Suggest a solution (e.g., "Vui lòng thử lại với yêu cầu khác, ví dụ: 'Giá đóng cửa của Apple vào ngày 01/01/2025'.").
            - The response must be in Vietnamese, natural, detailed (at least 3–4 sentences), and can include markdown to emphasize key information.

            **Example 1**:
            Input: {{"query": "What was the average closing price of Apple during Q1 2025?", "responses": [{{"status": "success", "message": "Query executed successfully", "data": {{"tables": ["companies", "stock_prices"], "sql_query": "SELECT c.name, AVG(sp.close_price) AS avg_close_price FROM companies c JOIN stock_prices sp ON c.symbol = sp.symbol WHERE c.name ILIKE 'Apple%' AND sp.date BETWEEN '2025-01-01' AND '2025-03-31' GROUP BY c.name", "result": "not_empty", "result_data": [{{"name": "Apple Inc.", "avg_close_price": 123.45}}]}}}, {{"status": "success", "message": "Documents found", "data": {{"sources": ["Apple.pdf"], "documents": "not_empty", "summary": "Theo báo cáo tài chính, Apple đạt thu nhập ròng 61,110 triệu USD trong năm 2024, tăng 5% so với năm trước."}}}}], "dashboard_info": {{"Dashboard": true, "visualization": {{"type": "table", "required_columns": ["name", "avg_close_price"]}}}}}}
            Output:
            Yêu cầu của bạn về giá đóng cửa trung bình của Apple trong quý 1 năm 2025 đã được xử lý. Dữ liệu giá đóng cửa của Apple đã được tìm thấy từ cơ sở dữ liệu, với giá đóng cửa trung bình là **123.45 USD** trong khoảng thời gian từ 01/01/2025 đến 31/03/2025. Ngoài ra, theo báo cáo tài chính từ nguồn Apple.pdf, thu nhập ròng của Apple trong năm 2024 là **61,110 triệu USD**, tăng 5% so với năm trước. Một bảng dữ liệu chi tiết đã được hiển thị để bạn tham khảo.

            **Example 2**:
            Input: {{"query": "What was the average closing price of Apple during Q1 2025?", "responses": [{{"status": "success", "message": "SQL query generated successfully", "data": {{"tables": ["companies", "stock_prices"], "sql_query": "SELECT c.name, AVG(sp.close_price) AS avg_close_price FROM companies c JOIN stock_prices sp ON c.symbol = sp.symbol WHERE c.name ILIKE 'Apple%' AND sp.date BETWEEN '2025-01-01' AND '2025-03-31' GROUP BY c.name", "result": "empty", "result_data": []}}}, {{"status": "success", "message": "Documents found", "data": {{"sources": ["Apple.pdf"], "documents": "not_empty", "summary": "Theo báo cáo tài chính, Apple đạt thu nhập ròng 61,110 triệu USD trong năm 2024, tăng 5% so với năm trước."}}}}], "dashboard_info": {{"Dashboard": true, "visualization": {{"type": "table", "required_columns": ["name", "avg_close_price"]}}}}}}
            Output:
            Yêu cầu tìm giá đóng cửa trung bình của Apple trong quý 1 năm 2025 (từ 01/01/2025 đến 31/03/2025) đã được xử lý. Tuy nhiên, không tìm thấy dữ liệu giá trong khoảng thời gian này, có thể do dữ liệu chưa được cập nhật. Dù vậy, theo báo cáo tài chính từ nguồn Apple.pdf, thu nhập ròng của Apple trong năm 2024 là **61,110 triệu USD**, tăng 5% so với năm trước. Bạn có muốn thử với một khoảng thời gian khác, ví dụ như quý 4 năm 2024 không?
            """
        ],
        debug_mode=True,
    )