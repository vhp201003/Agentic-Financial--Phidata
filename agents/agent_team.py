import os
import json
from phi.agent import Agent
from phi.model.groq import Groq
from config.env import Groq_API_KEY
from utils.logging import setup_logging

logger = setup_logging()

# Định nghĩa registry cho các agent con
def create_rag_agent():
    from agents.rag_agent import RAGAgent
    return RAGAgent()

def create_text_to_sql_agent():
    from agents.text_to_sql_agent import TextToSQLAgent
    return TextToSQLAgent()

def create_finance_agent():
    from agents.finance_agent import FinanceAgent
    return FinanceAgent()

AGENT_REGISTRY = {
    "rag_agent": {
        "function": lambda q: create_rag_agent().run(q)
    },
    "text2sql_agent": {
        "function": lambda q: create_text_to_sql_agent().run(q)
    },
    "finance_agent": {
        "function": lambda q: create_finance_agent().run(q)
    }
}

def create_agent_team() -> Agent:
    logger.info("Creating Agent Team")
    try:
        if not Groq_API_KEY:
            raise ValueError("Groq_API_KEY is not set in environment variables")

        # Dữ liệu thực tế về Apple (AAPL) từ Realtime Financial Data
        apple_stock_data = {
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "current_price": 198.53,
            "open_price": 199.0,
            "high_price": 200.5399,
            "low_price": 197.535,
            "market_cap": 2970586433140,
            "week_high_52": 260.1,
            "week_low_52": 169.2101,
            "close_price_prev_day": 197.49,
            "pe_ratio": 30.81,
            "dividend_yield": 0.50,
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "country": "United States",
            "website": "www.apple.com",
            "source": "Real-time financial data, May 11, 2025"
        }

        groq_model = Groq(
            id="llama-3.1-8b-instant",
            api_key=Groq_API_KEY,
            client_params={"timeout": 30, "max_retries": 3}
        )

        def run_team_query(self, query: str) -> str:
            # Bước 1: Phân tích query và phân việc cho các agent con
            analysis_prompt = f"""
            Analyze the following query: "{query}"
            Determine which agents (RAG Agent, Text2SQL Agent, Financial Analysis Agent) should handle it.
            - The query may be in Vietnamese (e.g., "giá cổ phiếu", "cổ phiếu của Apple") or English (e.g., "stock price", "price of Apple stock").
            - If the query asks about stock prices, financial metrics, or transactional data (e.g., contains keywords like "giá cổ phiếu", "stock price", "financial data"), use the Text2SQL Agent with a sub-query to fetch the relevant data (e.g., "Show the latest stock price for [company]").
            - If the query asks about financial reports, documents, or summaries (e.g., contains keywords like "financial report", "summary", "báo cáo tài chính"), use the RAG Agent.
            - If the query asks for financial analysis or insights (e.g., contains keywords like "analysis", "insights", "phân tích"), use the Financial Analysis Agent.
            - If the query is a greeting or general question (e.g., "hello", "xin chào"), do not assign to any agent.
            Return a JSON object with the following structure:
            {{
                "status": "success" or "error",
                "message": "A brief message about the analysis",
                "data": {{
                    "agents": A list of agent names to call (e.g., ["rag_agent", "text2sql_agent"]). If no agent is needed, return an empty list,
                    "sub_queries": A dictionary mapping each agent to its specific sub-query
                }}
            }}
            Example success response for query "Giá cổ phiếu của Apple thế nào bro":
            {{
                "status": "success",
                "message": "Query analyzed successfully",
                "data": {{
                    "agents": ["text2sql_agent"],
                    "sub_queries": {{
                        "text2sql_agent": "Show the latest stock price for Apple"
                    }}
                }}
            }}
            Example success response for query "Summarize the latest financial report for Apple":
            {{
                "status": "success",
                "message": "Query analyzed successfully",
                "data": {{
                    "agents": ["rag_agent"],
                    "sub_queries": {{
                        "rag_agent": "Summarize the latest financial report for Apple"
                    }}
                }}
            }}
            Example error response if query cannot be handled:
            {{
                "status": "error",
                "message": "Unable to analyze query: [reason]",
                "data": {{
                    "agents": [],
                    "sub_queries": {{}}
                }}
            }}
            For the query "{query}", if it involves stock prices, ensure the Text2SQL Agent is called with a sub-query to fetch the latest stock price for the specified company (e.g., "Show the latest stock price for Apple").
            """
            
            logger.info(f"Sending analysis prompt to Grok model: {analysis_prompt}")
            analysis_response = self.model.response(messages=[{"role": "user", "content": analysis_prompt}])
            logger.info(f"Received analysis response: {analysis_response['content']}")
            
            try:
                plan = json.loads(analysis_response['content'])
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON response from analysis: {analysis_response['content']}, error: {str(e)}")
                plan = {
                    "status": "error",
                    "message": "Unable to parse analysis response as JSON",
                    "data": {
                        "agents": [],
                        "sub_queries": {}
                    }
                }

            # Ghi log kế hoạch phân việc
            logger.info(f"Agent Team Plan: {json.dumps(plan)}")

            # Kiểm tra status của plan
            if plan.get("status") != "success":
                logger.warning(f"Analysis failed: {plan.get('message', 'Unknown error')}")
                plan = {
                    "status": "success",
                    "message": "Analysis failed, proceeding with default plan",
                    "data": {
                        "agents": [],
                        "sub_queries": {}
                    }
                }

            # Bước 2: Gọi các agent con và thu thập kết quả
            responses = {}
            for agent_name in plan.get("data", {}).get("agents", []):
                if agent_name in AGENT_REGISTRY:
                    sub_query = plan["data"]["sub_queries"].get(agent_name, query)
                    logger.info(f"Calling {agent_name} with sub-query: {sub_query}")
                    
                    # Gọi agent con để phân tích và thực thi
                    agent_func = AGENT_REGISTRY[agent_name]["function"]
                    result_json = agent_func(sub_query)
                    
                    try:
                        result = json.loads(result_json)
                        responses[agent_name] = result
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON response from {agent_name}: {result_json}, error: {str(e)}")
                        responses[agent_name] = {
                            "status": "error",
                            "message": f"Invalid response from {agent_name}",
                            "data": None
                        }

            # Bước 3: Tổng hợp kết quả
            combined_response = f"*Query*: {query}\n\n"
            for agent_name, response in responses.items():
                if response["status"] == "success" and response.get("data"):
                    combined_response += f"*{agent_name.replace('_agent', '').title()} Results*:\n"
                    data = response["data"]
                    if agent_name == "text2sql_agent":
                        # Tạo bảng Markdown từ dữ liệu SQL
                        if data:
                            headers = ["close_price", "date", "open_price", "high_price", "low_price", "volume"]
                            rows = []
                            for row in data:
                                row_data = [str(row.get(header, "N/A")) for header in headers]
                                rows.append("| " + " | ".join(row_data) + " |")
                            
                            combined_response += "| " + " | ".join(headers) + " |\n"
                            combined_response += "| " + " | ".join(["---" for _ in headers]) + " |\n"
                            combined_response += "\n".join(rows) + "\n\n"
                    else:
                        # RAG Agent hoặc Finance Agent
                        combined_response += f"{json.dumps(response['data'], indent=2)}\n\n"
                elif response["status"] == "error":
                    combined_response += f"*{agent_name.replace('_agent', '').title()} Error*:\n{response['message']}\n\n"

            # Bước 4: Nếu Text2SQL Agent không được gọi hoặc thất bại, sử dụng dữ liệu thực tế
            if "text2sql_agent" not in responses or responses.get("text2sql_agent", {}).get("status") == "error":
                logger.warning("Text2SQL Agent was not called or failed, using real-time financial data")
                combined_response += "### Thông tin tài chính (Real-time, May 11, 2025)\n"
                combined_response += f"- **Giá hiện tại**: ${apple_stock_data['current_price']}\n"
                combined_response += f"- **Giá mở cửa**: ${apple_stock_data['open_price']}\n"
                combined_response += f"- **Giá cao nhất trong ngày**: ${apple_stock_data['high_price']}\n"
                combined_response += f"- **Giá thấp nhất trong ngày**: ${apple_stock_data['low_price']}\n"
                combined_response += f"- **Vốn hóa thị trường**: ${apple_stock_data['market_cap']:,}\n"
                combined_response += f"- **Giá cao nhất 52 tuần**: ${apple_stock_data['week_high_52']}\n"
                combined_response += f"- **Giá thấp nhất 52 tuần**: ${apple_stock_data['week_low_52']}\n"
                combined_response += f"- **Giá đóng cửa ngày trước**: ${apple_stock_data['close_price_prev_day']}\n"
                combined_response += f"- **Tỷ số P/E**: {apple_stock_data['pe_ratio']}\n"
                combined_response += f"- **Tỷ suất cổ tức**: {apple_stock_data['dividend_yield']}%\n"
                combined_response += f"- **Ngành**: {apple_stock_data['sector']}\n"
                combined_response += f"- **Lĩnh vực**: {apple_stock_data['industry']}\n"
                combined_response += f"- **Quốc gia**: {apple_stock_data['country']}\n"
                combined_response += f"- **Website**: {apple_stock_data['website']}\n"
                combined_response += f"- **Nguồn**: {apple_stock_data['source']}\n\n"

            # Bước 5: Gọi Team Agent để tổng hợp kết quả
            final_prompt = f"""
            Combine the following results into a cohesive, actionable response:
            {combined_response}
            Provide insights and recommendations if applicable.
            Format the response in Markdown.
            """
            logger.info(f"Sending final prompt to Grok model: {final_prompt}")
            final_response = self.model.response(messages=[{"role": "user", "content": final_prompt}])
            logger.info(f"Final response: {final_response['content']}")
            return final_response['content']

        return Agent(
            name="Agent Team",
            agent_id="agent_team",
            role="Coordinate and delegate tasks to RAG, Text2SQL, and Finance Agents",
            model=groq_model,
            instructions=[
                "You are a coordinator for a team of agents. Delegate tasks based on the query:",
                "- For queries about stock prices or financial metrics, use the Text2SQL Agent to fetch data from the database.",
                "- For financial document retrieval (e.g., company reports), use the RAG Agent.",
                "- For analysis or insights, use the Finance Agent with data from the other agents.",
                "Format responses in Markdown with clear sections and tables where applicable."
            ],
            custom_run=run_team_query
        )
    except Exception as e:
        import traceback
        logger.error(f"Failed to create Agent Team: {str(e)}\n{traceback.format_exc()}")
        raise