import json
import re
from phi.agent import RunResponse
from utils.logging import setup_logging
from utils.response import standardize_response
import pandas as pd

logger = setup_logging()

def sql_flow(sub_query: str, sql_agent, sql_tool, metadata: dict = None) -> dict:
    try:
        logger.info(f"Calling sql_agent with sub_query: {sub_query}")
        sql_response = sql_agent.run(sub_query, metadata=metadata or {})
        token_metrics = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        if isinstance(sql_response, RunResponse):
            metrics = getattr(sql_response, 'metrics', {})
            input_tokens = metrics.get('input_tokens', 0)
            output_tokens = metrics.get('output_tokens', 0)
            token_metrics["input_tokens"] = input_tokens[0] if isinstance(input_tokens, list) and input_tokens else input_tokens
            token_metrics["output_tokens"] = output_tokens[0] if isinstance(output_tokens, list) and output_tokens else output_tokens
            token_metrics["total_tokens"] = metrics.get('total_tokens', token_metrics["input_tokens"] + token_metrics["output_tokens"])
            if isinstance(token_metrics["total_tokens"], list):
                token_metrics["total_tokens"] = token_metrics["total_tokens"][0] if token_metrics["total_tokens"] else 0
            logger.info(f"[Text2SQL] Token metrics: Input tokens={token_metrics['input_tokens']}, Output tokens={token_metrics['output_tokens']}, Total tokens={token_metrics['total_tokens']}")
            sql_response = sql_response.content
        logger.debug(f"Raw SQL response from sql_agent: {sql_response}")

        if sql_response.startswith("Không tạo được câu SQL"):
            logger.error(f"Failed to generate SQL query: {sql_response}")
            return {
                "response_for_chat": sql_response,
                "actual_result": [],
                "token_metrics": token_metrics,
                "sql_query": "Không tạo được câu SQL"
            }
        
        sql_query = re.sub(r'```(?:sql|json)?|```|\n|\t', '', sql_response).strip()
        if not sql_query.endswith(';'):
            sql_query += ';'
        logger.info(f"Executing SQL query with sql_tool: {sql_query}")
        try:
            tool_response = sql_tool.run(sql_query)
            tool_response_dict = json.loads(tool_response)
            logger.info(f"Parsed tool response: {json.dumps(tool_response_dict, ensure_ascii=False)}")
        except Exception as e:
            logger.error(f"Error executing query with sql_tool: {str(e)}")
            return {
                "response_for_chat": f"Lỗi thực thi query: {str(e)}",
                "actual_result": [],
                "token_metrics": token_metrics,
                "sql_query": sql_query
            }
        
        result_data = tool_response_dict["data"].get("result", [])
        if isinstance(result_data, pd.DataFrame):
            result_data = result_data.to_dict('records')
        elif not isinstance(result_data, list):
            logger.error(f"Invalid result data format: {type(result_data)}")
            return {
                "response_for_chat": "Dữ liệu trả về không hợp lệ từ cơ sở dữ liệu.",
                "actual_result": [],
                "token_metrics": token_metrics,
                "sql_query": sql_query
            }

        response_for_chat = (
            f"Dữ liệu từ cơ sở dữ liệu cho truy vấn '{sub_query}': {json.dumps(result_data, ensure_ascii=False)}"
            if result_data
            else f"Không tìm thấy dữ liệu trong cơ sở dữ liệu cho truy vấn '{sub_query}'."
        )

        return {
            "response_for_chat": response_for_chat,
            "actual_result": result_data,
            "token_metrics": token_metrics,
            "sql_query": sql_query  # Thêm câu SQL vào final_response
        }
    except Exception as e:
        logger.error(f"Error in sql_flow: {str(e)}")
        return {
            "response_for_chat": f"Lỗi trong sql_flow: {str(e)}",
            "actual_result": [],
            "token_metrics": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "sql_query": "Lỗi trong sql_flow"
        }