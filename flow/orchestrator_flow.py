import json
from phi.agent import Agent, RunResponse
from utils.logging import setup_logging
from utils.response import standardize_response
from utils.response_parser import parse_response_to_json
from flow.sql_flow import sql_flow
from flow.rag_flow import rag_flow

logger = setup_logging()

def process_response(response: any, context: str) -> dict:
    """Process response, log token metrics, and return JSON dict."""
    try:
        if isinstance(response, RunResponse):
            metrics = getattr(response, 'metrics', {})
            input_tokens = metrics.get('input_tokens', 0)
            output_tokens = metrics.get('output_tokens', 0)
            total_tokens = metrics.get('total_tokens', input_tokens + output_tokens)
            logger.info(f"[{context}] Token metrics: Input tokens={input_tokens}, Output tokens={output_tokens}, Total tokens={total_tokens}")
            response_content = response.content
        else:
            response_content = response

        if isinstance(response_content, dict):
            logger.info(f"[{context}] Response is already JSON: {json.dumps(response_content, ensure_ascii=False)}")
            return response_content
        elif not isinstance(response_content, str):
            logger.warning(f"[{context}] Unexpected result type: {type(response_content)}")
            return standardize_response("error", f"Kết quả không phải chuỗi hoặc JSON ở {context}", {})

        return parse_response_to_json(response_content, context)

    except Exception as e:
        logger.error(f"[{context}] Error processing response: {str(e)}")
        return standardize_response("error", f"Lỗi xử lý phản hồi {context}: {str(e)}", {})

def orchestrator_flow(query: str, orchestrator: Agent, sql_agent, sql_tool, rag_agent, rag_tool, chat_completion_agent) -> dict:
    """Xử lý flow của Agent Team: phân việc, gọi agent con, và tổng hợp kết quả."""
    try:
        # Process orchestrator response
        result = orchestrator.run(query)
        result_dict = process_response(result, "Orchestrator")
        logger.info(f"Orchestrator Response: {json.dumps(result_dict, indent=2, ensure_ascii=False)}")

        # If orchestrator response is invalid, return error
        if result_dict.get("status") == "error":
            return {
                "status": "error",
                "message": "Có lỗi xảy ra khi phân tích truy vấn: " + result_dict.get("message", "Không xác định"),
                "data": {}
            }

        # Process sub-queries
        data = result_dict.get("data", {})
        responses = []
        actual_results = []  # Lưu trữ dữ liệu thực tế để vẽ dashboard
        for agent_name in data.get("agents", []):
            sub_query = data.get("sub_queries", {}).get(agent_name)
            if not sub_query:
                logger.error(f"No sub-query provided for {agent_name}")
                return {
                    "status": "error",
                    "message": "Không có truy vấn phụ cho " + agent_name,
                    "data": {}
                }
            if agent_name == "text2sql_agent":
                final_response = sql_flow(sub_query, sql_agent, sql_tool)
                # final_response giờ chứa cả response_for_chat và actual_result
                if isinstance(final_response, dict) and "response_for_chat" in final_response:
                    response_for_chat = final_response["response_for_chat"]
                    actual_result = final_response["actual_result"]
                    final_response_dict = process_response(response_for_chat, "Text2SQL Agent")
                    logger.info(f"SQL Response: {json.dumps(final_response_dict, indent=2, ensure_ascii=False)}")
                    responses.append(final_response_dict)
                    actual_results.append(actual_result)
                else:
                    final_response_dict = process_response(final_response, "Text2SQL Agent")
                    responses.append(final_response_dict)
            elif agent_name == "rag_agent":
                final_response = rag_flow(sub_query, rag_agent, rag_tool)
                final_response_dict = process_response(final_response, "RAG Agent")
                logger.info(f"RAG Response: {json.dumps(final_response_dict, indent=2, ensure_ascii=False)}")
                responses.append(final_response_dict)

        # Gửi responses và metadata từ Orchestrator đến Chat Completion Agent
        if responses:
            chat_input = {
                "responses": responses,
                "dashboard_info": {
                    "Dashboard": data.get("Dashboard", False),
                    "visualization": data.get("visualization", {"type": "none"})
                }
            }
            chat_response = chat_completion_agent.run(json.dumps(chat_input, ensure_ascii=False))
            chat_response_dict = process_response(chat_response, "Chat Completion Agent")
            logger.info(f"Chat Completion Response: {json.dumps(chat_response_dict, indent=2, ensure_ascii=False)}")

            # Tạo phản hồi cuối cùng, trả về dữ liệu để frontend vẽ dashboard
            final_response = {
                "status": "success",
                "message": chat_response_dict.get("message", "Không có câu trả lời."),
                "data": {
                    "result": chat_response_dict.get("message", "Không có câu trả lời."),
                    "dashboard": {
                        "enabled": chat_response_dict.get("Dashboard", False),
                        "data": actual_results[0] if actual_results else [],
                        "visualization": data.get("visualization", {"type": "none"})
                    }
                }
            }

            return final_response
        else:
            return {
                "status": "error",
                "message": "Không có phản hồi từ các agent để xử lý.",
                "data": {}
            }

    except Exception as e:
        logger.error(f"Error in orchestrator flow: {str(e)}")
        return {
            "status": "error",
            "message": f"Có lỗi xảy ra khi xử lý truy vấn: {str(e)}",
            "data": {}
        }