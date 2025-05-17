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
        "intents": ["báo cáo", "annual report", "tài chính", "report"],
        "sub_query_template": "Summarize {intent} for {company}",
        "description": "Summarizes financial reports or documents"
    }
}

def create_rag_agent() -> Agent:
    """Tạo RAG Agent để xử lý truy vấn RAG với Qdrant."""
    logger.info("Creating RAG Agent")
    # Tạo ánh xạ công ty từ thư mục PDF
    company_mapping = build_company_mapping()
    
    return Agent(
        model=Groq(
            id=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            client_params={"timeout": 30, "max_retries": 3}
        ),
        description="RAG Agent: Summarizes financial reports or documents using Qdrant vector search.",
        instructions=[
            f"""
            **Objective**: Create RAG query from sub-query, return structured output only. Do not execute query.

            **TOOLS_CONFIG**: {json.dumps({"rag_agent": TOOLS_CONFIG["rag_agent"]}, ensure_ascii=False, indent=2)}

            **Keywords**: revenue, profit, business strategy, trend, business performance, financial metrics, financial report

            **Output**:
            {{
              "status": "success" | "error",
              "message": "RAG query generated successfully" | "Invalid sub-query" | "Company not found",
              "data": {{
                "rag_query": "sub-query",
                "company": "company name",
                "description": "keyword or null",
                "result": [],
                "suggestion": "suggestion or null"
              }}
            }} or {{}}

            **Rules**:
            - Match sub-query to intents: {TOOLS_CONFIG['rag_agent']['intents']}.
            - Map intents: 'báo cáo', 'tài chính', 'report', 'annual report' → 'financial report'.
            - Extract 'company' from sub-query (e.g., 'Apple' from 'Summarize report for Apple').
            - Map company to full name via Python (e.g., 'apple' → 'Apple').
            - Extract keywords (e.g., 'doanh thu' → 'revenue') from Keywords list.
            - Generate 'rag_query':
              - No keywords: 'Summarize financial report for {{company}}'
              - With keywords: 'Summarize financial report for {{company}} focusing on {{description}}'
              - With year (e.g., 'in 2024'): Append ' in {{year}}'
            - Set 'description' to keyword (e.g., 'revenue') or null.
            - Invalid sub-query: Return {{}}.
            - Unmapped company: Set 'company' to raw name, 'message' to 'Company not found', 'suggestion' to 'Try full name like Apple Inc.'.

            **Example**:
            Input: 'Summarize report for Apple with doanh thu'
            Output:
            {{
              "status": "success",
              "message": "RAG query generated successfully",
              "data": {{
                "rag_query": "Summarize financial report for Apple focusing on revenue",
                "company": "Apple",
                "description": "revenue",
                "result": [],
                "suggestion": null
              }}
            }}

            **Constraints**:
            - Return ONLY the specified structure: {{...}} or {{}}.
            - Do NOT wrap the output in markdown code blocks (e.g., ```json ... ```).
            - NO markdown, code fences, code, text, or explanations outside the specified structure.
            - Strictly follow Rules and Output format.
            """
        ],
        debug_mode=True
    )