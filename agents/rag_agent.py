from phi.model.groq import Groq
from config.env import Groq_API_KEY, GROQ_MODEL
from utils.logging import setup_logging

logger = setup_logging()

class RAGAgent:
    def __init__(self):
        self.model = Groq(
            id=GROQ_MODEL,
            api_key=Groq_API_KEY,
            client_params={"timeout": 30, "max_retries": 3}
        )

    def run(self, query: str) -> str:
        """Generate summaries or retrieve financial documents and return the result in JSON format."""
        logger.info(f"RAG Agent processing query: {query}")

        # Prompt để yêu cầu Grok trả về JSON
        prompt = f"""
        You are a retrieval-augmented generation (RAG) agent that summarizes financial reports or retrieves financial documents.
        Analyze the following query: "{query}"
        - If the query asks for financial reports, documents, or summaries (e.g., "báo cáo tài chính của Apple", "summarize Apple's financial report"), provide a summary or document details.
        - If the query cannot be handled, return an error message.
        Return a JSON object with the following structure:
        {{
            "status": "success" or "error",
            "message": "A brief message about the response",
            "data": {{
                "sources": ["doc1", "doc2"],
                "documents": {{"doc1": "Document content", "doc2": "Document content"}},
                "summary": "A summary of the financial report or document"
            }}
        }}
        Example success response:
        {{
            "status": "success",
            "message": "RAG query executed successfully",
            "data": {{
                "sources": ["doc1", "doc2"],
                "documents": {{"doc1": "Financial report Q1 2025 for Apple...", "doc2": "Annual report 2024 for Apple..."}},
                "summary": "Apple's Q1 2025 financial report shows a revenue of $100B and profit of $20B."
            }}
        }}
        Example error response:
        {{
            "status": "error",
            "message": "Unable to process RAG query: invalid query"
        }}
        """
        
        logger.info("Sending analysis prompt to Grok model")
        analysis_response = self.model.response(messages=[{"role": "user", "content": prompt}])
        logger.info(f"Received analysis response: {analysis_response['content']}")
        
        # Trả về JSON trực tiếp
        return analysis_response['content']