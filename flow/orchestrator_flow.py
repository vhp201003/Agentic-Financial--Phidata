# orchestrator_flow.py
import json
from phi.agent import Agent, RunResponse
from utils.logging import setup_logging, get_collected_logs
from utils.response import standardize_response
from utils.response_parser import parse_response_to_json
from flow.sql_flow import sql_flow
from flow.rag_flow import rag_flow
import re

logger = setup_logging()

def process_response(response: any, context: str) -> dict:
    """Process response, log token metrics, and return JSON dict (only for Orchestrator)."""
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
            return standardize_response("error", "Xin lỗi, hệ thống không thể phân tích truy vấn của bạn. Vui lòng thử lại với một truy vấn khác.", {})

        return parse_response_to_json(response_content, context)

    except Exception as e:
        logger.error(f"[{context}] Error processing response: {str(e)}")
        return standardize_response("error", "Xin lỗi, hệ thống không thể phân tích truy vấn của bạn. Vui lòng thử lại với một truy vấn khác.", {})

def limit_lines(text: str, max_lines: int = 5) -> str:
    """Giới hạn số dòng của text, tối đa max_lines dòng."""
    lines = text.split("\n")
    if len(lines) > max_lines:
        limited_lines = lines[:max_lines]
        limited_lines.append("... (đã lược bớt nội dung để tránh vượt giới hạn)")
        return "\n".join(limited_lines)
    return text

def limit_sql_records(sql_response: str, max_records: int = 5) -> str:
    """Giới hạn số record trong SQL response, thêm '...' nếu vượt quá max_records."""
    try:
        match = re.search(r'\[(.*)\]', sql_response, re.DOTALL)
        if not match:
            return sql_response

        records_str = match.group(1)
        records = []
        current_record = ""
        brace_count = 0
        for char in records_str:
            current_record += char
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and current_record.strip():
                    records.append(current_record.strip())
                    current_record = ""
                    if records and records[-1][-1] == ',':
                        records[-1] = records[-1][:-1]
        if current_record.strip() and brace_count == 0:
            records.append(current_record.strip().rstrip(','))

        if len(records) <= max_records:
            return sql_response

        limited_records = records[:max_records]
        limited_records_str = ", ".join(limited_records)
        truncated_response = sql_response[:match.start()] + f'[{limited_records_str}, ...]' + sql_response[match.end():]
        return truncated_response
    except Exception as e:
        logger.error(f"Error limiting SQL records: {str(e)}")
        return sql_response

def orchestrator_flow(query: str, orchestrator: Agent, sql_agent, sql_tool, rag_agent, rag_tool, chat_completion_agent) -> dict:
    """Xử lý flow của Agent Team: phân việc, gọi agent con, và tổng hợp kết quả."""
    try:
        # Process orchestrator response (JSON)
        result = orchestrator.run(query)
        result_dict = process_response(result, "Orchestrator")
        logger.info(f"Orchestrator Response: {json.dumps(result_dict, indent=2, ensure_ascii=False)}")

        # If orchestrator response is invalid, return error
        if result_dict.get("status") == "error":
            return {
                "status": "error",
                "message": "Xin lỗi, hệ thống không thể phân tích truy vấn của bạn. Vui lòng thử lại với một truy vấn khác, ví dụ: 'Giá cổ phiếu của Apple vào ngày 01/01/2025'.",
                "data": {},
                "logs": get_collected_logs()
            }

        # Process sub-queries
        data = result_dict.get("data", {})
        rag_response = "Không có phản hồi từ RAG."
        sql_response = "Không có phản hồi từ SQL."
        actual_results = []  # Lưu trữ dữ liệu thực tế để vẽ dashboard

        for agent_name in data.get("agents", []):
            sub_query = data.get("sub_queries", {}).get(agent_name)
            if not sub_query:
                logger.error(f"No sub-query provided for {agent_name}")
                return {
                    "status": "error",
                    "message": "Xin lỗi, hệ thống không thể xử lý yêu cầu của bạn. Vui lòng thử lại với một yêu cầu khác.",
                    "data": {},
                    "logs": get_collected_logs()
                }
            if agent_name == "text2sql_agent":
                final_response = sql_flow(sub_query, sql_agent, sql_tool)
                response_for_chat = final_response["response_for_chat"]
                actual_result = final_response["actual_result"]
                sql_response = limit_sql_records(response_for_chat, max_records=5)
                logger.info(f"SQL Response (limited): {sql_response}")
                actual_results.append(actual_result)
            elif agent_name == "rag_agent":
                final_response = rag_flow(sub_query, rag_agent, rag_tool)
                rag_response = limit_lines(final_response, max_lines=5)
                logger.info(f"RAG Response: {rag_response}")

        # Kiểm tra actual_results để quyết định Dashboard
        dashboard_enabled = data.get("Dashboard", False) and bool(actual_results and actual_results[0])  # Chỉ bật nếu có dữ liệu
        dashboard_info = {
            "Dashboard": dashboard_enabled,
            "visualization": data.get("visualization", {"type": "none"})
        }
        chat_input = (
            f"RAG response: {rag_response}\n"
            f"SQL response: {sql_response}\n\n"
            f"Dashboard info: {json.dumps(dashboard_info, ensure_ascii=False)}"
        )
        logger.info(f"Chat input: {chat_input}")

        # Chat Completion Agent trả về text markdown
        chat_response = chat_completion_agent.run(chat_input)
        if isinstance(chat_response, RunResponse):
            chat_response = chat_response.content
        logger.info(f"Chat Completion Response: {chat_response}")

        # Tạo phản hồi cuối cùng
        final_response = {
            "status": "success",
            "message": chat_response if isinstance(chat_response, str) else "Không có câu trả lời.",
            "data": {
                "result": chat_response if isinstance(chat_response, str) else "Không có câu trả lời.",
                "dashboard": {
                    "enabled": dashboard_enabled,
                    "data": actual_results[0] if actual_results else [],
                    "visualization": data.get("visualization", {"type": "none"})
                }
            },
            "logs": get_collected_logs()
        }
        return final_response

    except Exception as e:
        logger.error(f"Error in orchestrator flow: {str(e)}")
        return {
            "status": "error",
            "message": "Xin lỗi, hệ thống không thể xử lý yêu cầu của bạn. Vui lòng thử lại với một yêu cầu khác.",
            "data": {},
            "logs": get_collected_logs()
        }