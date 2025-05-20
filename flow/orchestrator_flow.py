import json
from pathlib import Path
from phi.agent import Agent, RunResponse
from utils.logging import setup_logging, get_collected_logs
from utils.response import standardize_response
from utils.response_parser import parse_response_to_json
from flow.sql_flow import sql_flow
from flow.rag_flow import rag_flow
from flow.chat_completion_flow import chat_completion_flow
import re
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
logger = setup_logging()

def load_metadata() -> dict:
    """Đọc visualization metadata."""
    metadata = {"template_query": []}
    vis_metadata_file = BASE_DIR / "config" / "visualization_metadata.yml"
    try:
        with open(vis_metadata_file, "r") as file:
            vis_metadata = yaml.safe_load(file)
        logger.info("Successfully loaded visualization_metadata.yml")
        metadata["visualization_metadata"] = vis_metadata["visualization_metadata"]
    except FileNotFoundError:
        logger.error("visualization_metadata.yml not found")
        metadata["visualization_metadata"] = []
    except Exception as e:
        logger.error(f"Error loading visualization metadata: {str(e)}")
        metadata["visualization_metadata"] = []
    return metadata

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

        cleaned_content = re.sub(r'```(?:json|sql)?|```|\n|\t', '', response_content).strip()
        logger.debug(f"[{context}] Cleaned response content: {cleaned_content}")
        return parse_response_to_json(cleaned_content, context)

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
    try:
        match = re.search(r'\[(.*)\]', sql_response, re.DOTALL)
        if not match:
            return sql_response

        records_str = match.group(1)
        try:
            records = json.loads(f"[{records_str}]")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in SQL response: {str(e)}")
            return sql_response

        if len(records) <= max_records:
            return sql_response

        limited_records = records[:max_records]
        limited_records_str = json.dumps(limited_records, ensure_ascii=False)
        truncated_response = sql_response[:match.start()] + f'{limited_records_str[:-1]}, ...]' + sql_response[match.end():]
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
        max_dashboard_records = 1000
        if len(data) > max_dashboard_records:
            logger.info(f"Limiting {len(data)} dashboard records to {max_dashboard_records}")
            return data[:max_dashboard_records]
        return data
    if len(data) <= max_records:
        return data
    if for_chat_input:
        logger.info(f"Limiting {len(data)} records to {max_records} for chat input to avoid token limit")
    else:
        logger.info(f"Limiting {len(data)} records to {max_records} for Chat Completion log")
    return data[:max_records]

def orchestrator_flow(query: str, orchestrator: Agent, sql_agent, sql_tool, rag_tool, chat_completion_agent) -> dict:
    metadata = load_metadata()
    try:
        result = orchestrator.run(query)
        result_dict = process_response(result, "Orchestrator")
        logger.info(f"Orchestrator Response: {json.dumps(result_dict, indent=2, ensure_ascii=False)}")

        if result_dict.get("status") == "error":
            return {
                "status": "error",
                "message": "Sorry, the system cannot parse your query. Please try again with a different query, e.g., 'Stock price of Apple on 01/01/2025'.",
                "data": {},
                "logs": get_collected_logs()
            }

        data = result_dict.get("data", {})
        rag_documents = []
        sql_response = "No response from SQL."
        actual_results = []

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
                metadata_with_columns = data.copy()
                metadata_with_columns["visualization_metadata"] = metadata["visualization_metadata"]
                final_response = sql_flow(sub_query, sql_agent, sql_tool, metadata=metadata_with_columns)
                response_for_chat = final_response["response_for_chat"]
                actual_result = final_response["actual_result"]
                sql_response = limit_sql_records(response_for_chat, max_records=5)
                dashboard_result = limit_records(actual_result, for_dashboard=True)
                limited_result = limit_records(actual_result, max_records=5, for_dashboard=False)
                logger.info(f"SQL Response (limited for log): {sql_response}")
                logger.info(f"Dashboard records: {len(dashboard_result)}, Limited log records: {len(limited_result)}")
                actual_results.append(dashboard_result)
            elif agent_name == "rag_agent":
                rag_documents = rag_flow(sub_query, rag_tool)
                logger.info(f"RAG Documents: {rag_documents}")

        dashboard_enabled = data.get("Dashboard", False) and bool(actual_results and actual_results[0])
        dashboard_data = actual_results[0] if actual_results else []
        limited_dashboard_data = limit_records(dashboard_data, max_records=5, for_chat_input=True)
        dashboard_info = {
            "enabled": dashboard_enabled,
            "data": limited_dashboard_data,
            "visualization": {
                "type": data.get("vis_type", "none"),
                "required_columns": data.get("required_columns", []),
                "aggregation": data.get("aggregation", None),
                "ui_requirements": data.get("ui_requirements", {})
            }
        }

        # Truyền dashboard_enabled vào chat_completion_flow
        final_response_message = chat_completion_flow(query, rag_documents, sql_response, dashboard_info, chat_completion_agent, tickers=data.get("tickers", []))
        logger.info(f"Final response message: {final_response_message}")

        final_dashboard_info = {
            "enabled": dashboard_enabled,
            "data": dashboard_data,
            "visualization": {
                "type": data.get("vis_type", "none"),
                "required_columns": data.get("required_columns", []),
                "aggregation": data.get("aggregation", None),
                "ui_requirements": data.get("ui_requirements", {})
            }
        }
        final_response = {
            "status": "success",
            "message": final_response_message if isinstance(final_response_message, str) else "No response available.",
            "data": {
                "result": final_response_message if isinstance(final_response_message, str) else "No response available.",
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