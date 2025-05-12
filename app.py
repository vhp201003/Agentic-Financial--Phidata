from fastapi import FastAPI
from pydantic import BaseModel
from agents.agent_team import create_agent_team
from agents.rag_agent import RAGAgent
from agents.text_to_sql_agent import TextToSQLAgent
from agents.finance_agent import FinanceAgent
from utils.logging import setup_logging

logger = setup_logging()

app = FastAPI()

# Khởi tạo các agent
rag_agent = RAGAgent()
text_to_sql_agent = TextToSQLAgent()
finance_agent = FinanceAgent()
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