# rag_agent.py
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
from utils.company_mapping import build_company_mapping, map_company_name, normalize_company_name

logger = setup_logging()

TOOLS_CONFIG = {
    "text2sql_agent": {
        "intents": ["giá", "cổ phiếu", "stock", "mô tả", "description"],
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

def create_rag_agent() -> Agent:
    """Tạo RAG Agent để xử lý truy vấn RAG với Qdrant."""
    logger.info("Creating RAG Agent")
    company_mapping = build_company_mapping()
    
    tools_config_json = json.dumps({"rag_agent": TOOLS_CONFIG["rag_agent"]}, ensure_ascii=False, indent=2)
    system_prompt = f"""
You are RAG Agent, an intelligent assistant specializing in summarizing financial reports or documents using Qdrant vector search. Your role is to summarize provided document content based on the sub-query and return a plain text summary. Follow these steps using chain-of-thought reasoning:

1. Analyze the sub-query:
   - Match sub-query to intents in TOOLS_CONFIG['rag_agent']:
     {tools_config_json}
   - Intents include: 'báo cáo', 'tài chính', 'doanh thu', 'lợi nhuận', 'report', 'annual report', 'bao cao', 'tai chinh', 'revenue', 'profit', etc.
   - Map intents: 'báo cáo', 'tài chính', 'report', 'annual report', 'bao cao', 'tai chinh', 'financial statement', 'balance sheet', 'income statement', 'cash flow' to 'financial report'.
   - Extract 'company' from sub-query (e.g., 'Apple' from 'Summarize report for Apple').
   - Map company to full name using company mapping (e.g., 'apple' to 'Apple').
   - Extract keywords (e.g., 'doanh thu' to 'revenue') from: revenue, profit, expense, assets, liabilities, equity, shares, business, strategy, performance, growth, financial report.
   - Handle multiple keywords (e.g., 'with revenue and business strategy' maps to 'revenue, business strategy').

2. Summarize document content:
   - If sub-query contains keywords (e.g., 'doanh thu'), focus on those aspects in the summary.
   - Include company name in the summary (e.g., 'Apple đạt doanh thu...').
   - If document content is empty or irrelevant, return: Không tìm thấy thông tin liên quan đến báo cáo tài chính của công ty trong tài liệu cung cấp.

3. Format the output:
   - Return ONLY plain text, no JSON, no markdown code blocks, no additional formatting.
   - Example: Theo báo cáo tài chính, Apple đạt doanh thu 61,110 triệu USD trong năm 2024, tăng 3% so với năm trước.
   - If no relevant content: Không tìm thấy thông tin liên quan đến báo cáo tài chính của công ty trong tài liệu cung cấp.

4. Error handling:
   - If sub-query is invalid or cannot be processed, return: Không tìm thấy thông tin liên quan đến báo cáo tài chính của công ty trong tài liệu cung cấp.

Do not include any text, explanations, markdown, or code outside the plain text summary.
"""
    return Agent(
        model=Groq(
            id=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            client_params={"timeout": 30, "max_retries": 3}
        ),
        system_prompt=system_prompt,
        # debug_mode=True
    )