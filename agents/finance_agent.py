from phi.model.groq import Groq
from config.env import Groq_API_KEY, GROQ_MODEL
from utils.logging import setup_logging

logger = setup_logging()

class FinanceAgent:
    def __init__(self):
        self.model = Groq(
            id=GROQ_MODEL,
            api_key=Groq_API_KEY,
            client_params={"timeout": 30, "max_retries": 3}
        )

    def run(self, query: str) -> str:
        """Generate financial analysis or insights and return the result in JSON format."""
        logger.info(f"Finance Agent processing query: {query}")

        # Prompt để yêu cầu Grok trả về JSON
        prompt = f"""
        You are a financial analysis agent that provides insights based on financial queries.
        Analyze the following query: "{query}"
        - If the query asks for financial analysis or insights (e.g., "phân tích tài chính của Apple", "insights on Apple's stock"), provide a brief analysis or insight.
        - If the query cannot be handled, return an error message.
        Return a JSON object with the following structure:
        {{
            "status": "success" or "error",
            "message": "A brief message about the response",
            "data": {{
                "analysis": "The financial analysis or insight text here"
            }}
        }}
        Example success response:
        {{
            "status": "success",
            "message": "Financial analysis generated",
            "data": {{
                "analysis": "Apple's stock has shown strong performance with a P/E ratio of 30.81, indicating high investor confidence."
            }}
        }}
        Example error response:
        {{
            "status": "error",
            "message": "Unable to generate financial analysis: invalid query"
        }}
        """
        
        logger.info("Sending analysis prompt to Grok model")
        analysis_response = self.model.response(messages=[{"role": "user", "content": prompt}])
        logger.info(f"Received analysis response: {analysis_response['content']}")
        
        # Trả về JSON trực tiếp
        return analysis_response['content']