from fastapi import FastAPI
from pydantic import BaseModel
from Data_Platform.Final.financial_agent_system.agents.orchestrator import create_agent_team
from agents.rag_agent import create_rag_agent
from agents.text_to_sql_agent import create_text_to_sql_agent
from agents.finance_agent import create_finance_agent
from utils.logging import setup_logging

from tools.rag_tool import CustomRAGTool
from tools.sql_tool import CustomSQLTool

logger = setup_logging()

app = FastAPI()

rag_tool = CustomRAGTool()
sql_tool = CustomSQLTool()

# Khởi tạo các agent
rag_agent = create_rag_agent(rag_tool)
text_to_sql_agent = create_text_to_sql_agent(sql_tool)
finance_agent = create_finance_agent()
team_agent = create_agent_team()

class QueryRequest(BaseModel):
    query: str

@app.post("/rag")
async def query_rag(request: QueryRequest):
    response = rag_agent.run(request.query)
    return {"response": response}

@app.post("/sql")
async def query_sql(request: QueryRequest):
    response = text_to_sql_agent.run(request.query)
    return {"response": response}

@app.post("/finance")
async def query_finance(request: QueryRequest):
    response = finance_agent.run(request.query)
    return {"response": response}

@app.post("/team")
async def query_team(request: QueryRequest):
    logger.info(f"Received query for Agent Team: {request.query}")
    response = team_agent.run(request.query)
    return {"response": response}