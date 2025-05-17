from fastapi import FastAPI
from pydantic import BaseModel
from agents.orchestrator import create_orchestrator
from agents.rag_agent import create_rag_agent
from agents.text_to_sql_agent import create_text_to_sql_agent
# from agents.finance_agent import create_finance_agent
from agents.chat_completion_agent import create_chat_completion_agent  # Đảm bảo import đúng
from tools.sql_tool import CustomSQLTool
from tools.rag_tool import CustomRAGTool
from flow.orchestrator_flow import orchestrator_flow
from utils.logging import setup_logging
import json
import uvicorn

logger = setup_logging()

app = FastAPI()

# Khởi tạo các agent
rag_agent = create_rag_agent()  # Gọi class trực tiếp
text_to_sql_agent = create_text_to_sql_agent()  # Gọi class trực tiếp
# finance_agent = create_finance_agent()  # Gọi class trực tiếp
chat_completion_agent = create_chat_completion_agent()  # Sửa: Gọi hàm để tạo instance
orchestrator = create_orchestrator()  # Gọi hàm để tạo instance
sql_tool = CustomSQLTool()
rag_tool = CustomRAGTool()

class QueryRequest(BaseModel):
    query: str

@app.post("/team")
async def query_team(request: QueryRequest):
    logger.info(f"Received query for Agent Team: {request.query}")
    response = orchestrator_flow(request.query, orchestrator, text_to_sql_agent, sql_tool, rag_agent, rag_tool, chat_completion_agent)
    return {"response": json.dumps(response, ensure_ascii=False)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)