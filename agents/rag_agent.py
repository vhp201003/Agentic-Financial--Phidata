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
            **Objective**: Generate RAG query from sub-query using TOOLS_CONFIG['rag_agent']. Do NOT execute query.

            **TOOLS_CONFIG**:
            {json.dumps({"rag_agent": TOOLS_CONFIG["rag_agent"]}, ensure_ascii=False, indent=2)}

            **Company Mapping** (from PDF filenames):
            {json.dumps(company_mapping, ensure_ascii=False, indent=2)}

            **Output**: JSON object:
            {{
              "status": "success" | "error",
              "message": "RAG query generated successfully" | "Invalid sub-query" | "Company not found",
              "data": {{
                "rag_query": "sub-query",
                "company": "company name",
                "description": "additional keyword",
                "result": [],
                "suggestion": "optional suggestion if company not found"
              }}
            }} or {{}}

            **Rules**:
            - Match sub-query to {TOOLS_CONFIG['rag_agent']['intents']}. 
            - Map intents: 'báo cáo', 'tài chính' to 'report'; 'mô tả' to 'description'.
            - Set 'rag_query' to the sub-query as-is (e.g., 'Summarize report for Apple with business strategy in 2024').
            - Extract 'company' from sub-query (e.g., 'Apple' from 'Summarize report for Apple with business strategy').
            - Use Company Mapping to convert extracted company to full name (e.g., 'apple' → 'Apple').
            - Extract 'description' from sub-query if it contains additional keywords (e.g., 'business strategy' from 'with business strategy').
            - Invalid sub-query: return {{}}.
            - If company not in Company Mapping, set 'company' to extracted name, set 'message' to 'Company not found', and add 'suggestion' (e.g., 'Try full name like Apple Inc. or check PDF filenames').

            **Constraints**:
            - Return ONLY JSON object: {{...}} or {{}}.
            - NO markdown, code fences, code, text, or explanations.
            - Strictly follow Rules and Output format.
            """
        ],
        debug_mode=True
    )