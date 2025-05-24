# app.py
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import asyncio
import json
import queue
import re
from agents.orchestrator import create_orchestrator
from agents.text_to_sql_agent import create_text_to_sql_agent
from agents.chat_completion_agent import create_chat_completion_agent
from agents.visualize_agent import create_visualize_agent
from tools.sql_tool import CustomSQLTool
from tools.rag_tool import CustomRAGTool
from flow.orchestrator_flow import orchestrator_flow
from utils.logging import setup_logging, get_collected_logs
from utils.company_mapping import build_company_mapping
from pydantic import BaseModel
import uvicorn

logger = setup_logging()

app = FastAPI()

# Initialize agents and tools
text_to_sql_agent = create_text_to_sql_agent()
chat_completion_agent = create_chat_completion_agent()
visualize_agent = create_visualize_agent()
orchestrator = create_orchestrator()
sql_tool = CustomSQLTool()
rag_tool = CustomRAGTool()

# Load valid companies from company_mapping
VALID_COMPANIES = build_company_mapping()

def normalize_company_name(query):
    logger.info(f"Original query: {query}")
    normalized_query = re.sub(r'[\-\s]+', ' ', query.lower())

    import unicodedata
    def remove_accents(text):
        return ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))
    
    normalized_no_accents = remove_accents(normalized_query)
    standardized_query = normalized_query
    for valid_key, valid_name in VALID_COMPANIES.items():
        pattern = r'\b' + re.escape(valid_key) + r'\b'
        if re.search(pattern, normalized_no_accents, flags=re.IGNORECASE):
            standardized_query = re.sub(pattern, valid_name, standardized_query, flags=re.IGNORECASE)

    logger.info(f"Normalized query: {standardized_query}")
    return standardized_query

async def process_query_generator(query: str):
    try:
        logger.info(f"Received query: {query}")
        normalized_query = normalize_company_name(query)
        thinking_queue = queue.Queue()

        async def run_orchestrator():
            return await asyncio.to_thread(
                orchestrator_flow,
                normalized_query,
                orchestrator,
                text_to_sql_agent,
                sql_tool,
                rag_tool,
                chat_completion_agent,
                thinking_queue=thinking_queue
            )

        task = asyncio.create_task(run_orchestrator())
        while not task.done():
            try:
                message = thinking_queue.get_nowait()
                yield f"event: thinking\ndata: {json.dumps({'message': message}, ensure_ascii=False)}\n\n"
            except queue.Empty:
                await asyncio.sleep(0.1)

        result = await task
        # logger.info(f"Orchestrator result: {json.dumps(result, ensure_ascii=False)}")

        if result["status"] == "error":
            yield f"event: error\ndata: {json.dumps({'message': result['message']}, ensure_ascii=False)}\n\n"
            return

        yield f"event: result\ndata: {json.dumps(result, ensure_ascii=False)}\n\n"
    except Exception as e:
        logger.error(f"Error in process_query_generator: {str(e)}")
        yield f"event: error\ndata: {json.dumps({'message': f'Internal server error: {str(e)}'}, ensure_ascii=False)}\n\n"

@app.get("/process_query")
async def process_query(request: Request, query: str):
    logger.info(f"Accessing /process_query with query: {query}")
    return StreamingResponse(process_query_generator(query), media_type="text/event-stream")

class QueryRequest(BaseModel):
    query: str

@app.post("/team")
async def query_team(request: QueryRequest):
    logger.info(f"Received query for Agent Team: {request.query}")
    normalized_query = normalize_company_name(request.query)
    response = orchestrator_flow(normalized_query, orchestrator, text_to_sql_agent, sql_tool, rag_tool, chat_completion_agent)

    if response["status"] == "error" or response["data"].get("result") is None or not response["data"]["result"]:
        response = {
            "status": "error",
            "message": "Sorry, an error occurred while processing your request. Please try again later.",
            "data": {},
            "logs": get_collected_logs()
        }

    return {"response": json.dumps(response, ensure_ascii=False)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010)