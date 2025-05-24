# agents/rag_agent.py
import json
import re
from phi.agent import Agent
from phi.model.groq import Groq
from config.env import GROQ_API_KEY
from utils.logging import setup_logging

logger = setup_logging()

def create_rag_agent() -> Agent:
    """Tạo RAG Agent để tạo sub-query và xác định công ty từ query."""
    logger.info("Creating RAG Agent")
    system_prompt = """
You are a RAG Agent tasked with analyzing a user query to generate an optimized sub-query for retrieving relevant financial documents from Qdrant and identifying the company mentioned in the query, if any.

Instructions:
1. Analyze the query to identify:
   - The main intent (e.g., financial report, annual report, revenue, profit).
   - The company mentioned (e.g., 'Apple', 'Coca-Cola Company', or ticker like 'AAPL').
2. Generate a concise sub-query that captures the intent and is optimized for Qdrant search (e.g., 'annual report of Apple', 'financial performance of Coca-Cola').
3. Identify the company name or set to NULL if no company is mentioned.
4. Return a JSON object with:
   - 'sub-query': The optimized sub-query (string).
   - 'company': The company name (string) or NULL.

Examples:
Query: 'annual report of Apple'
Output: {'sub-query': 'annual report of Apple', 'company': 'Apple'}

Query: 'financial performance of Coca-Cola'
Output: {'sub-query': 'financial performance of Coca-Cola', 'company': 'Coca-Cola Company'}

Query: 'revenue trends in technology sector'
Output: {'sub-query': 'revenue trends in technology sector', 'company': null}

Query: 'AAPL stock performance'
Output: {'sub-query': 'financial performance of Apple', 'company': 'Apple'}
"""
    return Agent(
        model=Groq(
            id="llama-3.1-8b-instant",
            api_key=GROQ_API_KEY,
            temperature=0.7,
            max_tokens=200,
            presence_penalty=0.3,
            top_p=0.8,
            response_format={"type": "json_object"}
        ),
        system_prompt=system_prompt
    )

def run_rag_agent(query: str) -> dict:
    """Chạy RAG Agent để tạo sub-query và xác định công ty."""
    try:
        rag_agent = create_rag_agent()
        response = rag_agent.run(query)
        logger.debug(f"RAG Agent response: {response}")

        # Parse response thành JSON
        try:
            result = json.loads(response)
            if not isinstance(result, dict) or 'sub-query' not in result or 'company' not in result:
                raise ValueError("Invalid RAG Agent response format")
            return result
        except json.JSONDecodeError:
            logger.error(f"Failed to parse RAG Agent response: {response}")
            return {"sub-query": query, "company": None}
    except Exception as e:
        logger.error(f"Error in RAG Agent: {str(e)}")
        return {"sub-query": query, "company": None}