import os
import sys
from pathlib import Path
import json
from typing import Dict, Any

# Thêm thư mục gốc dự án vào sys.path
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
        "intents": ["giá", "cổ phiếu", "stock", "mô tả", "description"],
        "sub_query_template": "Retrieve {intent} data for {company}",
        "description": "Queries database for stock prices or company info"
    },
    "rag_agent": {
        "intents": ["báo cáo", "annual report", "tài chính", "report"],
        "sub_query_template": "Summarize {intent} for {company}",
        "description": "Summarizes financial reports or documents"
    }
}

def create_orchestrator():
    """Tạo orchestrator và trả về Agent chính."""
    return Agent(
        model=Groq(
            id=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            client_params={"timeout": 30, "max_retries": 3}
        ),
        description="Orchestrator: Analyzes user query and delegates to RAG, Text2SQL, or other agents.",
        instructions=[
            f"""
            **Objective**: Assign user query to agents using TOOLS_CONFIG.

            **TOOLS_CONFIG**:
            {json.dumps(TOOLS_CONFIG, ensure_ascii=False, indent=2)}

            **Output**: JSON object:
            {{
              "status": "success" | "error",
              "message": "Query analyzed successfully" | "Invalid query",
              "data": {{
                "agents": ["agent_name"],
                "sub_queries": {{"agent_name": "sub-query"}}
              }}
            }} or {{}}

            **Rules**:
            - Match query keywords to TOOLS_CONFIG[agent]['intents'] to select agent.
            - Map intents: 'giá', 'cổ phiếu' to 'stock price'; 'mô tả' to 'description'; 'báo cáo', 'tài chính' to 'report'.
            - Map additional keywords: 'chiến lược kinh doanh' to 'business strategy'; 'doanh thu' to 'revenue'; 'lợi nhuận' to 'profit'; 'tóm tắt' to 'summarize'.
            - Create sub-query from TOOLS_CONFIG[agent]['sub_query_template'] using mapped {{intent}}, {{company}}.
            - If query contains additional keywords (e.g., 'chiến lược kinh doanh', 'doanh thu'), append them to sub-query (e.g., 'Summarize report for Apple with business strategy').
            - Latest date ('ngày gần nhất', 'most recent'): Use latest date, no specific date in sub-query.
            - Specific date (e.g., 'ngày DD tháng MM năm YYYY' or 'YYYY-MM-DD'): Append 'on YYYY-MM-DD' to sub-query for text2sql_agent.
            - Specific year (e.g., 'năm YYYY' or 'YYYY'): Append 'in YYYY' to sub-query for rag_agent (e.g., 'Summarize report for Apple in 2024').
            - General queries ('What can you do?', 'Hello', 'Hi'): return {{
              "status": "success",
              "message": "General query",
              "data": {{
                "agents": [],
                "sub_queries": {{}},
                "general_response": "System supports stock queries, report summaries"
              }}
            }}.
            - Invalid queries: return {{}}.
            - sub_queries keys must match agents.
            - Sub-queries must always be in English: Translate all Vietnamese keywords to English before creating sub-query.

            **Constraints**:
            - Return ONLY JSON object: {{...}} or {{}}.
            - NO markdown, code fences, code, text, or explanations.
            - Strictly follow Rules and Output format.
            """
        ],
        debug_mode=True
    )