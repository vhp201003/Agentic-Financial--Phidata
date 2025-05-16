# main.py
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
from utils.response_validator import validate_response

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
        sql_agent = create_text_to_sql_agent()
        sql_tool = CustomSQLTool()
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

            # Validate orchestrator response
            validated_orchestrator = validate_response(result_content, "orchestrator")
            if validated_orchestrator:
                result_dict = validated_orchestrator.dict()
                logger.info(f"Validated orchestrator response: {json.dumps(result_dict, ensure_ascii=False)}")
                print("\nOrchestrator Response:")
                print(json.dumps(result_dict, indent=2, ensure_ascii=False))
            else:
                logger.error(f"Invalid orchestrator response: {result_content[:100]}...")
                print(json.dumps(
                    standardize_response("error", "Phản hồi orchestrator không phải JSON hợp lệ", {}),
                    indent=2, ensure_ascii=False
                ))
                continue

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
                    if isinstance(final_response, RunResponse):
                        log_token_metrics(final_response, "Text2SQL Agent")
                        final_response_content = final_response.content
                    else:
                        final_response_content = final_response

                    # Validate Text2SQL response
                    validated_response = validate_response(final_response_content, "text2sql")
                    if validated_response:
                        final_response_dict = validated_response.dict()
                        logger.info(f"Validated Text2SQL response: {json.dumps(final_response_dict, ensure_ascii=False)}")
                        print("\nSQL Response:")
                        print(json.dumps(final_response_dict, indent=2, ensure_ascii=False))
                    else:
                        logger.error(f"Invalid Text2SQL response: {final_response_content[:100]}...")
                        print(json.dumps(
                            standardize_response("error", "Phản hồi Text2SQL không phải JSON hợp lệ", {}),
                            indent=2, ensure_ascii=False
                        ))
                elif agent_name == "rag_agent":
                    final_response = rag_flow(sub_query, rag_agent, rag_tool)
                    if isinstance(final_response, RunResponse):
                        log_token_metrics(final_response, "RAG Agent")
                        final_response_content = final_response.content
                    else:
                        final_response_content = final_response

                    # Validate RAG response
                    validated_response = validate_response(final_response_content, "rag")
                    if validated_response:
                        final_response_dict = validated_response.dict()
                        logger.info(f"Validated RAG response: {json.dumps(final_response_dict, ensure_ascii=False)}")
                        print("\nRAG Response:")
                        print(json.dumps(final_response_dict, indent=2, ensure_ascii=False))
                    else:
                        logger.error(f"Invalid RAG response: {final_response_content[:100]}...")
                        print(json.dumps(
                            standardize_response("error", "Phản hồi RAG không phải JSON hợp lệ", {}),
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