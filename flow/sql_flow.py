# sql_flow.py
import json
import re
from phi.agent import RunResponse
from utils.logging import setup_logging
from utils.response import standardize_response

logger = setup_logging()

def sql_flow(sub_query: str, sql_agent, sql_tool) -> dict:
    """Tích hợp sql_agent và sql_tool để tạo và thực thi SQL query."""
    try:
        # Gọi sql_agent để tạo SQL query
        logger.info(f"Calling sql_agent with sub_query: {sub_query}")
        sql_response = sql_agent.run(sub_query)
        if isinstance(sql_response, RunResponse):
            sql_response = sql_response.content
        
        logger.debug(f"Raw SQL response from sql_agent: {sql_response}")

        # Loại bỏ markdown code fences
        sql_response = re.sub(r'```(?:json|python|sql)?\n|\n```', '', sql_response).strip()
        
        # Parse JSON response
        try:
            sql_response_dict = json.loads(sql_response)
            logger.info(f"Parsed SQL response: {json.dumps(sql_response_dict, ensure_ascii=False)}")
        except json.JSONDecodeError:
            logger.error(f"SQL response is not valid JSON: {sql_response}")
            return standardize_response("error", "Phản hồi SQL không phải JSON hợp lệ", {})
        
        # Kiểm tra cấu trúc JSON
        if not isinstance(sql_response_dict, dict) or "status" not in sql_response_dict:
            logger.error(f"Invalid SQL response structure: {sql_response_dict}")
            return standardize_response("error", "Cấu trúc phản hồi SQL không hợp lệ", {})
        
        if sql_response_dict["status"] != "success":
            logger.warning(f"sql_agent failed: {sql_response_dict.get('message', 'Unknown error')}")
            return sql_response_dict
        
        # Lấy sql_query
        sql_query = sql_response_dict["data"].get("sql_query", "")
        if not sql_query:
            logger.error("No SQL query generated")
            return standardize_response("error", "Không tạo được SQL query", {})
        
        # Thực thi query với sql_tool
        logger.info(f"Executing SQL query with sql_tool: {sql_query}")
        try:
            tool_response = sql_tool.run(sql_query)
            logger.debug(f"Raw tool response: {tool_response}")
            tool_response_dict = json.loads(tool_response)
            logger.info(f"Parsed tool response: {json.dumps(tool_response_dict, ensure_ascii=False)}")
        except Exception as e:
            logger.error(f"Error executing query with sql_tool: {str(e)}")
            return standardize_response("error", f"Lỗi thực thi query: {str(e)}", {})
        
        # Kiểm tra trạng thái dữ liệu
        result_data = tool_response_dict["data"].get("result", [])
        result_status = "not_empty" if result_data else "empty"
        logger.info(f"Result data status: {result_status}, number of records: {len(result_data)}")
        if not result_data:
            logger.warning(f"No data returned for query: {sql_query}")

        # Chuẩn bị response cho Chat Completion Agent (bao gồm result_data)
        response_for_chat = {
            "status": sql_response_dict["status"],
            "message": sql_response_dict["message"],
            "data": {
                "tables": sql_response_dict["data"].get("tables", []),
                "sql_query": sql_query,
                "result": result_status,  # Trạng thái dữ liệu
                "result_data": result_data  # Dữ liệu thực tế
            }
        }

        # Trả về dữ liệu đầy đủ (bao gồm "result") để sử dụng sau này (e.g., vẽ dashboard)
        return {
            "response_for_chat": response_for_chat,
            "actual_result": result_data  # Dữ liệu thực tế để dùng cho dashboard
        }
    except Exception as e:
        logger.error(f"Error in sql_flow: {str(e)}")
        return standardize_response("error", f"Lỗi trong sql_flow: {str(e)}", {})