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
        sql_response = sql_agent.run(sub_query)
        if isinstance(sql_response, RunResponse):
            sql_response = sql_response.content
        
        logger.debug(f"Raw SQL response: {sql_response}")
        # Loại bỏ markdown code fences
        sql_response = re.sub(r'```(?:json|python|sql)?\n|\n```', '', sql_response).strip()
        
        # Parse JSON response
        try:
            sql_response_dict = json.loads(sql_response)
        except json.JSONDecodeError:
            logger.error(f"SQL response is not valid JSON: {sql_response}")
            return standardize_response("error", "Phản hồi SQL không phải JSON hợp lệ", {})
        
        # Kiểm tra cấu trúc JSON
        if not isinstance(sql_response_dict, dict) or "status" not in sql_response_dict:
            logger.error(f"Invalid SQL response structure: {sql_response_dict}")
            return standardize_response("error", "Cấu trúc phản hồi SQL không hợp lệ", {})
        
        if sql_response_dict["status"] != "success":
            return sql_response_dict
        
        # Lấy sql_query
        sql_query = sql_response_dict["data"].get("sql_query", "")
        if not sql_query:
            logger.error("No SQL query generated")
            return standardize_response("error", "Không tạo được SQL query", {})
        
        # Thực thi query với sql_tool
        logger.info(f"Executing SQL query: {sql_query}")
        try:
            tool_response = sql_tool.run(sql_query)
            tool_response_dict = json.loads(tool_response)
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return standardize_response("error", f"Lỗi thực thi query: {str(e)}", {})
        
        # Kết hợp kết quả
        return {
            "status": sql_response_dict["status"],
            "message": sql_response_dict["message"],
            "data": {
                "tables": sql_response_dict["data"].get("tables", []),
                "sql_query": sql_query,
                "result": tool_response_dict["data"].get("result", [])
            }
        }
    except Exception as e:
        logger.error(f"Error in sql_flow: {str(e)}")
        return standardize_response("error", f"Lỗi trong sql_flow: {str(e)}", {})