import json
import re
from phi.agent import RunResponse
from utils.logging import setup_logging
from utils.response import standardize_response

logger = setup_logging()

def rag_flow(sub_query: str, rag_agent, rag_tool) -> dict:
    """Tích hợp rag_agent và rag_tool để xử lý truy vấn RAG với Qdrant."""
    try:
        # Gọi rag_agent để tạo RAG query
        rag_response = rag_agent.run(sub_query)
        if isinstance(rag_response, RunResponse):
            rag_response = rag_response.content
        
        logger.debug(f"Raw RAG response: {rag_response}")
        # Loại bỏ markdown code fences
        rag_response = re.sub(r'```(?:json|python|sql)?\n|\n```', '', rag_response).strip()
        
        # Parse JSON response
        try:
            rag_response_dict = json.loads(rag_response)
        except json.JSONDecodeError:
            logger.error(f"RAG response is not valid JSON: {rag_response}")
            return standardize_response("error", "Phản hồi RAG không phải JSON hợp lệ", {})
        
        # Kiểm tra cấu trúc JSON
        if not isinstance(rag_response_dict, dict) or "status" not in rag_response_dict:
            logger.error(f"Invalid RAG response structure: {rag_response_dict}")
            return standardize_response("error", "Cấu trúc phản hồi RAG không hợp lệ", {})
        
        if rag_response_dict["status"] != "success":
            return rag_response_dict
        
        # Lấy thông tin từ rag_agent
        rag_query = rag_response_dict["data"].get("rag_query", "")
        company = rag_response_dict["data"].get("company", None)
        description = rag_response_dict["data"].get("description", None)
        
        if not rag_query:
            logger.error("No RAG query generated")
            return standardize_response("error", "Không tạo được RAG query", {})
        
        # Thực thi query với rag_tool (truy vấn Qdrant)
        logger.info(f"Executing RAG query: {rag_query} (company: {company}, description: {description})")
        try:
            # Truyền cả rag_query, company và description vào rag_tool
            tool_response = rag_tool.run(rag_query, company=company, description=description)
            tool_response_dict = json.loads(tool_response)
        except Exception as e:
            logger.error(f"Error executing RAG query: {str(e)}")
            return standardize_response("error", f"Lỗi thực thi RAG query: {str(e)}", {})
        
        # Kiểm tra cấu trúc tool_response
        if not isinstance(tool_response_dict, dict) or "status" not in tool_response_dict:
            logger.error(f"Invalid tool response structure: {tool_response_dict}")
            return standardize_response("error", "Cấu trúc phản hồi từ rag_tool không hợp lệ", {})
        
        if tool_response_dict["status"] != "success":
            return tool_response_dict

        # Kết hợp kết quả từ rag_agent và rag_tool
        return {
            "status": tool_response_dict["status"],
            "message": tool_response_dict["message"],
            "data": {
                "sources": tool_response_dict["data"].get("sources", []),
                "documents": tool_response_dict["data"].get("documents", {}),
                "metadata": tool_response_dict["data"].get("metadata", []),
                "rag_query": rag_query,
                "summary": tool_response_dict["data"].get("summary", "")
            }
        }
    except Exception as e:
        logger.error(f"Error in rag_flow: {str(e)}")
        return standardize_response("error", f"Lỗi trong rag_flow: {str(e)}", {})