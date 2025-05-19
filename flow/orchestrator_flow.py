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
            return standardize_response("error", "Sorry, the system cannot parse your query. Please try again with a different query.", {})

        return parse_response_to_json(response_content, context)

    except Exception as e:
        logger.error(f"[{context}] Error processing response: {str(e)}")
        return standardize_response("error", "Sorry, the system cannot parse your query. Please try again with a different query.", {})

def limit_lines(text: str, max_lines: int = 5) -> str:
    """Limit the number of lines in text to max_lines."""
    lines = text.split("\n")
    if len(lines) > max_lines:
        limited_lines = lines[:max_lines]
        limited_lines.append("... (content truncated to avoid exceeding limit)")
        return "\n".join(limited_lines)
    return text

def limit_sql_records(sql_response: str, max_records: int = 5) -> str:
    """Limit the number of records in SQL response, append '...' if exceeds max_records."""
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

def limit_records(data: list, max_records: int = 5, for_dashboard: bool = False, for_chat_input: bool = False) -> list:
    """Limit the number of records in the data list, depending on the context."""
    if not isinstance(data, list):
        logger.error(f"Expected list for limiting records, got {type(data)}")
        return []
    if for_dashboard:
        # Do not limit data for dashboard, but set a safe max if needed
        max_dashboard_records = 1000
        if len(data) > max_dashboard_records:
            logger.info(f"Limiting {len(data)} dashboard records to {max_dashboard_records}")
            return data[:max_dashboard_records]
        return data
    # Limit for Chat Completion log or chat input
    if len(data) <= max_records:
        return data
    if for_chat_input:
        logger.info(f"Limiting {len(data)} records to {max_records} for chat input to avoid token limit")
    else:
        logger.info(f"Limiting {len(data)} records to {max_records} for Chat Completion log")
    return data[:max_records]

def orchestrator_flow(query: str, orchestrator: Agent, sql_agent, sql_tool, rag_tool, chat_completion_agent) -> dict:
    """Execute the Agent Team flow: delegate tasks, call sub-agents, and combine results."""
    try:
        # Process orchestrator response (JSON)
        result = orchestrator.run(query)
        result_dict = process_response(result, "Orchestrator")
        logger.info(f"Orchestrator Response: {json.dumps(result_dict, indent=2, ensure_ascii=False)}")

        # If orchestrator response is invalid, return error
        if result_dict.get("status") == "error":
            return {
                "status": "error",
                "message": "Sorry, the system cannot parse your query. Please try again with a different query, e.g., 'Stock price of Apple on 01/01/2025'.",
                "data": {},
                "logs": get_collected_logs()
            }

        # Process sub-queries
        data = result_dict.get("data", {})
        rag_documents = []  # Store the original documents from RAG
        sql_response = "No response from SQL."
        actual_results = []  # Store actual data for dashboard

        # Lấy required_columns từ visualization để truyền vào sql_flow
        required_columns = data.get("visualization", {}).get("required_columns", [])

        for agent_name in data.get("agents", []):
            sub_query = data.get("sub_queries", {}).get(agent_name)
            if not sub_query:
                logger.error(f"No sub-query provided for {agent_name}")
                return {
                    "status": "error",
                    "message": "Sorry, the system cannot process your request. Please try again with a different request.",
                    "data": {},
                    "logs": get_collected_logs()
                }
            if agent_name == "text2sql_agent":
                # Truyền required_columns vào sql_flow
                final_response = sql_flow(sub_query, sql_agent, sql_tool, required_columns=required_columns)
                response_for_chat = final_response["response_for_chat"]
                actual_result = final_response["actual_result"]
                sql_response = limit_sql_records(response_for_chat, max_records=5)
                # Keep full actual_result for dashboard
                dashboard_result = limit_records(actual_result, for_dashboard=True)
                # Limit actual_result for Chat Completion log
                limited_result = limit_records(actual_result, max_records=5, for_dashboard=False)
                logger.info(f"SQL Response (limited for log): {sql_response}")
                logger.info(f"Dashboard records: {len(dashboard_result)}, Limited log records: {len(limited_result)}")
                actual_results.append(dashboard_result)
            elif agent_name == "rag_agent":
                rag_documents = rag_flow(sub_query, rag_tool)
                logger.info(f"RAG Documents: {rag_documents}")

        # Prepare dashboard info
        dashboard_enabled = data.get("Dashboard", False) and bool(actual_results and actual_results[0])
        dashboard_data = actual_results[0] if actual_results else []
        
        # Giới hạn dữ liệu trong dashboard_info khi gửi vào chat_input để tránh vượt giới hạn token
        limited_dashboard_data = limit_records(dashboard_data, max_records=5, for_chat_input=True)
        dashboard_info = {
            "enabled": dashboard_enabled,
            "data": limited_dashboard_data,  # Dữ liệu giới hạn cho chat_input
            "visualization": data.get("visualization", {"type": "none", "required_columns": [], "aggregation": None})
        }

        # Combine RAG documents, SQL response, and limited dashboard info for Chat Completion
        chat_input = (
            f"RAG documents:\n{json.dumps(rag_documents, ensure_ascii=False)}\n\n"
            f"SQL response:\n{sql_response}\n\n"
            f"Dashboard info:\n{json.dumps(dashboard_info, ensure_ascii=False)}"
        )
        logger.info(f"Chat input: {chat_input}")

        # Chat Completion Agent generates the final response
        chat_response = chat_completion_agent.run(chat_input)
        if isinstance(chat_response, RunResponse):
            chat_response = chat_response.content
        logger.info(f"Chat Completion Response: {chat_response}")

        # Create the final response, sử dụng dữ liệu đầy đủ cho dashboard
        final_dashboard_info = {
            "enabled": dashboard_enabled,
            "data": dashboard_data,  # Dữ liệu đầy đủ cho dashboard
            "visualization": data.get("visualization", {"type": "none", "required_columns": [], "aggregation": None})
        }
        final_response = {
            "status": "success",
            "message": chat_response if isinstance(chat_response, str) else "No response available.",
            "data": {
                "result": chat_response if isinstance(chat_response, str) else "No response available.",
                "dashboard": final_dashboard_info
            },
            "logs": get_collected_logs()
        }
        return final_response

    except Exception as e:
        logger.error(f"Error in orchestrator flow: {str(e)}")
        return {
            "status": "error",
            "message": "Sorry, the system cannot process your request. Please try again with a different request.",
            "data": {},
            "logs": get_collected_logs()
        }