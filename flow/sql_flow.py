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

        # Kiểm tra nếu sql_response là một thông báo lỗi
        if sql_response.startswith("Không tạo được câu SQL"):
            logger.error(f"Failed to generate SQL query: {sql_response}")
            return {
                "response_for_chat": sql_response,
                "actual_result": []
            }
        
        # sql_response là câu SQL, thực thi query với sql_tool
        sql_query = sql_response
        logger.info(f"Executing SQL query with sql_tool: {sql_query}")
        try:
            tool_response = sql_tool.run(sql_query)
            logger.debug(f"Raw tool response: {tool_response}")
            tool_response_dict = json.loads(tool_response)
            logger.info(f"Parsed tool response: {json.dumps(tool_response_dict, ensure_ascii=False)}")
        except Exception as e:
            logger.error(f"Error executing query with sql_tool: {str(e)}")
            return {
                "response_for_chat": f"Lỗi thực thi query: {str(e)}",
                "actual_result": []
            }
        
        # Kiểm tra trạng thái dữ liệu
        result_data = tool_response_dict["data"].get("result", [])
        result_status = "not_empty" if result_data else "empty"
        logger.info(f"Result data status: {result_status}, number of records: {len(result_data)}")
        if not result_data:
            logger.warning(f"No data returned for query: {sql_query}")

        # Chuẩn bị response cho Chat Completion Agent (dạng text)
        if result_data:
            response_for_chat = f"Dữ liệu từ cơ sở dữ liệu cho truy vấn '{sub_query}': {json.dumps(result_data, ensure_ascii=False)}"
        else:
            response_for_chat = f"Không tìm thấy dữ liệu trong cơ sở dữ liệu cho truy vấn '{sub_query}'."

        # Trả về dữ liệu đầy đủ (bao gồm result_data để vẽ dashboard)
        return {
            "response_for_chat": response_for_chat,
            "actual_result": result_data
        }
    except Exception as e:
        logger.error(f"Error in sql_flow: {str(e)}")
        return {
            "response_for_chat": f"Lỗi trong sql_flow: {str(e)}",
            "actual_result": []
        }