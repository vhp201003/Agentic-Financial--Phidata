import json
import sys
import re
from pathlib import Path
from phi.agent import RunResponse

# Thêm thư mục gốc dự án vào sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from agents.orchestrator import create_orchestrator
from agents.text_to_sql_agent import create_text_to_sql_agent
from agents.rag_agent import create_rag_agent
from tools.sql_tool import CustomSQLTool
from tools.rag_tool import CustomRAGTool
from flow.sql_flow import sql_flow
from flow.rag_flow import rag_flow
from utils.logging import setup_logging
from utils.response import standardize_response

logger = setup_logging()

def log_token_metrics(response: RunResponse, context: str):
    """Log token metrics từ RunResponse với context cụ thể."""
    try:
        metrics = getattr(response, 'metrics', {})
        input_tokens = metrics.get('input_tokens', 0)
        output_tokens = metrics.get('output_tokens', 0)
        total_tokens = metrics.get('total_tokens', input_tokens + output_tokens)
        logger.info(f"[{context}] Token metrics: Input tokens={input_tokens}, Output tokens={output_tokens}, Total tokens={total_tokens}")
    except Exception as e:
        logger.warning(f"Failed to log token metrics for {context}: {str(e)}")

def main():
    """Chạy vòng lặp tương tác để test hệ thống orchestrator."""
    logger.info("Starting orchestrator test")
    try:
        orchestrator = create_orchestrator()

        # sql
        sql_agent = create_text_to_sql_agent()
        sql_tool = CustomSQLTool()

        # rag
        rag_agent = create_rag_agent()
        rag_tool = CustomRAGTool()
    except Exception as e:
        logger.error(f"Failed to create orchestrator or agents: {str(e)}")
        print(json.dumps(
            standardize_response("error", f"Không thể khởi tạo orchestrator hoặc agents: {str(e)}", {}),
            indent=2, ensure_ascii=False
        ))
        return

    while True:
        query = input("Enter your query (or 'exit' to quit): ").strip()
        if query.lower() == "exit":
            logger.info("Exiting orchestrator test")
            break
        if not query:
            print("Please enter a valid query.")
            continue

        try:
            result = orchestrator.run(query)
            # Log token metrics cho orchestrator
            if isinstance(result, RunResponse):
                log_token_metrics(result, "Orchestrator")
                result_content = result.content
            else:
                result_content = result

            if not isinstance(result_content, str):
                logger.warning(f"Unexpected result type: {type(result_content)}")
                result_content = json.dumps(standardize_response("error", "Kết quả không phải chuỗi", {}), ensure_ascii=False)

            result_content = re.sub(r'```json\n|\n```', '', result_content).strip()
            try:
                result_dict = json.loads(result_content)
                print("\nResult:")
                print(json.dumps(result_dict, indent=2, ensure_ascii=False))
                
                # Xử lý sub-queries với sql_flow hoặc rag_flow
                data = result_dict.get("data", {})
                for agent_name in data.get("agents", []):
                    sub_query = data.get("sub_queries", {}).get(agent_name)
                    if not sub_query:
                        logger.error(f"No sub-query provided for {agent_name}")
                        print(json.dumps(
                            standardize_response("error", f"Không có sub-query cho {agent_name}", {}),
                            indent=2, ensure_ascii=False
                        ))
                        continue
                    if agent_name == "text2sql_agent":
                        final_response = sql_flow(sub_query, sql_agent, sql_tool)
                        # Log token metrics cho sql_agent
                        if isinstance(final_response, RunResponse):
                            log_token_metrics(final_response, "Text2SQL Agent")
                            final_response = final_response.content
                        print("\nSQL Response:")
                        print(json.dumps(final_response, indent=2, ensure_ascii=False))
                    elif agent_name == "rag_agent":
                        final_response = rag_flow(sub_query, rag_agent, rag_tool)
                        # Log token metrics cho rag_agent
                        if isinstance(final_response, RunResponse):
                            log_token_metrics(final_response, "RAG Agent")
                            final_response = final_response.content
                        print("\nRAG Response:")
                        print(json.dumps(final_response, indent=2, ensure_ascii=False))
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {str(e)}, raw result: {result_content}")
                print(json.dumps(
                    standardize_response("error", f"Lỗi JSON: {str(e)}", {}),
                    indent=2, ensure_ascii=False
                ))
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            print(json.dumps(
                standardize_response("error", f"Lỗi xử lý truy vấn: {str(e)}", {}),
                indent=2, ensure_ascii=False
            ))

if __name__ == "__main__":
    main()