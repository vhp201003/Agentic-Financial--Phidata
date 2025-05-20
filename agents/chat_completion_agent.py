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

TOOLS_CONFIG = {
    "text2sql_agent": {
        "intents": [
            "giá", "cổ phiếu", "stock", "mô tả", "description",
            "market cap", "pe ratio", "dividend yield", "52 week high", "52 week low",
            "volume", "dividends", "stock splits",
            "sector", "industry", "country",
            "highest price", "lowest price", "average price", "total volume", "average volume",
            "time series", "histogram", "boxplot", "scatter plot", "bar chart", "pie chart", "heatmap",
            "daily highlow range", "normal return"
        ],
        "sub_query_template": "Retrieve {intent} data for {company}",
        "description": "Queries database for stock prices or company info"
    },
    "finance_agent": {
        "intents": ["phân tích", "xu hướng", "ROI", "trend", "insight"],
        "sub_query_template": "Analyze {intent} for {company}",
        "description": "Analyzes financial trends or insights"
    },
    "rag_agent": {
        "intents": [
            "báo cáo", "tài chính", "doanh thu", "lợi nhuận", "chi phí", "tài sản", "nợ", "vốn", "cổ phần",
            "doanh nghiệp", "kinh doanh", "chiến lược", "kết quả hoạt động", "tăng trưởng",
            "bao cao", "tai chinh", "doanh thu", "loi nhuan", "chi phi", "tai san", "no", "von", "co phan",
            "doanh nghiep", "kinh doanh", "chien luoc", "ket qua hoat dong", "tang truong",
            "report", "annual report", "financial statement", "balance sheet", "income statement", "cash flow",
            "revenue", "profit", "expense", "assets", "liabilities", "equity", "shares",
            "business", "strategy", "performance", "growth"
        ],
        "sub_query_template": "Summarize {intent} for {company}",
        "description": "Summarizes financial reports or documents"
    }
}

def create_chat_completion_agent() -> Agent:
    """Tạo Chat Completion Agent để trả lời câu hỏi và viết tóm tắt."""
    logger.info("Creating Chat Completion Agent")

    tools_config_json = json.dumps(TOOLS_CONFIG, ensure_ascii=False, indent=2)
    system_prompt = """
You are Chat Completion Agent, answering the user's query based on summarized data from RAG, SQL, and dashboard, and providing a concise summary. Follow these steps:

1. Validate Input:
   - Inputs: Query (string), Tickers (JSON list), RAG Summary (string), SQL Summary (string), Dashboard Summary (string).
   - If all summaries indicate no data (e.g., 'Không có...'), return: Không có dữ liệu để trả lời truy vấn '{{query}}'.

2. Extract Query Intent:
   - Analyze the query to determine intent:
     - Single value: e.g., 'What is the closing price of IBM on 2024-03-01?'.
     - Comparison: e.g., 'Which company had a higher closing price, Amgen or Boeing?'.
     - Visualization: e.g., 'Pie chart of market cap proportions by sector'.

3. Parse Summaries for Data:
   - Parse RAG Summary, SQL Summary, and Dashboard Summary to extract relevant data:
     - RAG: Look for key metrics (e.g., 'Revenue 61,110M USD').
     - SQL: Extract metrics (e.g., 'AMGN: 319.29 USD, BA: 151.0 USD').
     - Dashboard: Extract chart data (e.g., 'Biểu đồ boxplot thể hiện sự biến động của daily_return theo tháng').
   - If SQL Summary is not pre-formatted, parse raw data:
     - SQL Summary: Parse JSON (e.g., 'Dữ liệu từ cơ sở dữ liệu ...: [{"symbol": "AAPL", "close_price": 237.87}, ...]').
     - Match symbols with Tickers (e.g., 'AAPL', 'MSFT') to assign values.

4. Answer the Query:
   - Use the extracted data to answer the query directly:
     - Single value: Extract metric (e.g., 'Giá đóng cửa của IBM là 248.66 USD.').
     - Comparison: Compare values (e.g., 'Microsoft có giá cao hơn Apple, 426.31 USD so với 237.87 USD.').
     - Visualization: Describe chart (e.g., 'Biểu đồ boxplot cho thấy daily_return của Apple biến động mạnh vào tháng 6/2024.').

5. Summarize:
   - Answer the query in 1-2 sentences.
   - Summarize the data in 3-4 sentences, referencing the RAG, SQL, and Dashboard summaries.
   - Enhance visualization summary using Dashboard Summary (e.g., if it mentions 'daily_return theo tháng', elaborate on the trend or key insights).

6. Output:
   - Plain text (not Markdown) with answer and summary:
     - Answer: [Direct answer to the query].
     - Summary: [3-4 sentences summarizing the data].

Example: Query: 'Create a boxplot of daily returns for Apple in 2024'
Tickers: ["AAPL"]
RAG Summary:\nKhông có tài liệu liên quan đến báo cáo tài chính.
SQL Summary:\nAAPL Daily Returns: Trung bình -0.0020
Dashboard Summary:\nBiểu đồ boxplot thể hiện sự biến động của daily_return theo tháng.
Output:
Answer: Biểu đồ boxplot thể hiện sự biến động của daily_return của Apple trong năm 2024.
Summary: Dữ liệu từ cơ sở dữ liệu cho thấy lợi nhuận hàng ngày trung bình của Apple là -0.0020 trong năm 2024. Biểu đồ boxplot trực quan hóa sự biến động của daily_return theo tháng, cho thấy các tháng có biến động lớn. Không có tài liệu RAG để phân tích thêm.
"""
    return Agent(
        model=Groq(
            id=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0.2,
            max_tokens=1000,
            presence_penalty=0.3,
            top_p=0.8
        ),
        system_prompt=system_prompt,
        custom_run=lambda self, chat_input: self.run_with_validation(chat_input),
        # debug_mode=True
    )

def run_with_validation(self, chat_input: str) -> str:
    """Validate input before running the model."""
    try:
        query_match = re.search(r'Query: (.*?)\nTickers:', chat_input, re.DOTALL)
        tickers_match = re.search(r'Tickers: (.*?)\nRAG Summary:', chat_input, re.DOTALL)
        rag_match = re.search(r'RAG Summary:\n(.*?)\nSQL Summary:', chat_input, re.DOTALL)
        sql_match = re.search(r'SQL Summary:\n(.*?)\nDashboard Summary:', chat_input, re.DOTALL)
        dashboard_match = re.search(r'Dashboard Summary:\n(.*)', chat_input, re.DOTALL)

        if not (query_match and tickers_match and rag_match and sql_match and dashboard_match):
            logger.error("Invalid chat input format")
            return "# Phản hồi\n## Tóm tắt\nKhông có dữ liệu để trả lời truy vấn."

        query = query_match.group(1)
        tickers = json.loads(tickers_match.group(1))
        formatted_rag = rag_match.group(1)
        formatted_sql = sql_match.group(1)
        formatted_dashboard = dashboard_match.group(1)

        has_rag = "Không tìm thấy tài liệu" not in formatted_rag
        has_sql = "Không tìm thấy dữ liệu tài chính" not in formatted_sql
        has_dashboard = formatted_dashboard.strip() != ""

        if not (has_rag or has_sql or has_dashboard):
            logger.info("No valid data to process")
            return f"# Phản hồi\n## Tóm tắt\nKhông có dữ liệu để trả lời truy vấn '{query}'."

        return self.run(chat_input)

    except Exception as e:
        logger.error(f"Error validating chat input: {str(e)}")
        return "# Phản hồi\n## Tóm tắt\nKhông có dữ liệu để trả lời truy vấn."