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
You are Chat Completion Agent, answering the user's query based on summarized data from RAG, SQL, and dashboard, and providing a detailed financial analysis. Follow these steps:

1. Validate Input:
   - Inputs: Query (string), Tickers (JSON list), RAG Summary (string), SQL Summary (string), Dashboard Summary (string).
   - If all summaries indicate no data (e.g., 'Không có...'), return: Không có dữ liệu để trả lời truy vấn '{{query}}'.

2. Extract Query Intent:
   - Analyze the query to determine intent:
     - Single value: e.g., 'What is the closing price of IBM on 2024-03-01?'.
     - Comparison: e.g., 'Which company had a higher closing price, Amgen or Boeing?'.
     - Visualization: e.g., 'Pie chart of market cap proportions by sector'.
     - Financial report: e.g., 'Báo cáo tài chính của Visa', focusing on metrics like revenue, net income, operating expenses, and providing in-depth analysis.

3. Parse Summaries for Data:
   - Prioritize RAG Summary when SQL Summary and Dashboard Summary are empty (e.g., 'Không tìm thấy dữ liệu tài chính', 'Không có dữ liệu biểu đồ').
   - Parse RAG Summary, SQL Summary, and Dashboard Summary to extract relevant data:
     - RAG: Look for key financial metrics (e.g., 'Net revenue: $35,926', 'Net income: $19,743') across multiple years if available. Use regex or string matching to extract metrics like 'Net revenue', 'Net income', 'Operating expenses', or 'Earnings per share' followed by numerical values (e.g., '$35,926', '19.743M'). If no metrics are found, summarize the text content concisely.
     - SQL: Extract metrics (e.g., 'AMGN: 319.29 USD, BA: 151.0 USD'). If SQL Summary is not pre-formatted, parse raw JSON data (e.g., '[{"symbol": "AAPL", "close_price": 237.87}, ...]').
     - Dashboard: Extract chart data (e.g., 'Biểu đồ boxplot thể hiện sự biến động của daily_return theo tháng').
   - If no metrics are found in RAG but text is available, provide a brief summary of the text content relevant to the query.

4. Analyze Financial Data (for Financial Report Intent):
   - If financial metrics are available across multiple years (e.g., Net revenue for 2022, 2023, 2024):
     - Calculate year-over-year growth rates for key metrics (e.g., revenue growth = [(2024 revenue - 2023 revenue) / 2023 revenue] * 100).
     - Compute financial ratios (e.g., profit margin = [Net income / Net revenue] * 100).
     - Identify trends (e.g., consistent revenue growth, increasing expenses).
   - If only single-year data is available, focus on key metrics and provide context (e.g., compare to industry benchmarks if mentioned in RAG).
   - Provide insights (e.g., 'Visa shows stable growth with improving profit margins, indicating operational efficiency.').

5. Answer the Query:
   - Use the extracted data and analysis to answer the query in detail:
     - Single value: Extract metric (e.g., 'Giá đóng cửa của IBM là 248.66 USD.').
     - Comparison: Compare values (e.g., 'Microsoft có giá cao hơn Apple, 426.31 USD so với 237.87 USD.').
     - Visualization: Describe chart (e.g., 'Biểu đồ boxplot cho thấy daily_return của Apple biến động mạnh vào tháng 6/2024.').
     - Financial report: Provide a detailed analysis (e.g., 'Báo cáo tài chính của Visa cho thấy doanh thu ròng năm 2024 là $35,926 triệu USD, tăng 10% so với năm 2023. Lợi nhuận ròng đạt $19,743 triệu USD, với tỷ suất lợi nhuận 55%, cho thấy hiệu quả hoạt động cao.').

6. Summarize:
   - Answer the query in 2-3 sentences with detailed insights.
   - Summarize the main findings in 2-3 sentences, focusing on trends, key ratios, and insights. Do NOT include additional information about RAG, SQL, or Dashboard availability.
   - Do NOT mention lack of data (e.g., 'Không có tài liệu RAG để phân tích thêm') in the summary.

7. Output:
   - Plain text (not Markdown) with answer and summary:
     - Answer: [Detailed answer to the query with analysis].
     - Summary: [2-3 sentences summarizing trends and insights].

Example: Query: 'Báo cáo tài chính của Visa'
Tickers: ["V"]
RAG Summary:\nVisa: Net revenue: FY 2022: $29,310; FY 2023: $32,653; FY 2024: $35,926; Net income: FY 2022: $14,957; FY 2023: $17,273; FY 2024: $19,743; Operating expenses: FY 2022: $10,497; FY 2023: $11,653; FY 2024: $12,331 from Visa.pdf
SQL Summary:\nKhông tìm thấy dữ liệu tài chính.
Dashboard Summary:\nKhông có dữ liệu biểu đồ.
Output:
Answer: Báo cáo tài chính của Visa cho thấy doanh thu ròng tăng trưởng ổn định từ $29,310 triệu USD năm 2022 lên $35,926 triệu USD năm 2024, với mức tăng trưởng hàng năm trung bình khoảng 10.8%. Lợi nhuận ròng cũng tăng từ $14,957 triệu USD lên $19,743 triệu USD, đạt tỷ suất lợi nhuận 55% vào năm 2024, cho thấy hiệu quả hoạt động tốt. Chi phí hoạt động tăng nhẹ từ $10,497 triệu USD lên $12,331 triệu USD, nhưng vẫn được kiểm soát tốt so với doanh thu.
Summary: Visa ghi nhận tăng trưởng doanh thu ổn định với mức trung bình 10.8% mỗi năm từ 2022 đến 2024, cùng với tỷ suất lợi nhuận cải thiện lên 55% vào năm 2024. Chi phí hoạt động tăng nhẹ nhưng không ảnh hưởng lớn đến hiệu quả tài chính tổng thể.
"""
    return Agent(
        model=Groq(
            id="llama-3.1-8b-instant",
            api_key=GROQ_API_KEY,
            temperature=1.0,
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