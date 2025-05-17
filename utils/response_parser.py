# utils/response_parser.py
import json
import re
from typing import Dict, Any
from pydantic import BaseModel, ValidationError
from utils.logging import setup_logging

logger = setup_logging()

# Định nghĩa Pydantic model cho response
class AgentResponse(BaseModel):
    status: str
    message: str
    data: Dict[str, Any]

def parse_response_to_json(response_content: str, context: str) -> Dict[str, Any]:
    """
    Parse response content thành JSON, trích xuất JSON nếu cần và ép kiểu bằng Pydantic.

    Args:
        response_content (str): Response content cần parse.
        context (str): Context để log (e.g., 'Text2SQL Agent').

    Returns:
        Dict[str, Any]: JSON dict đã parse và validate.
    """
    try:
        # Loại bỏ markdown code block nếu có (e.g., ```json ... ```)
        cleaned_content = re.sub(r'^```json\s*|\s*```$', '', response_content.strip())

        # Thử parse JSON trực tiếp
        try:
            response_dict = json.loads(cleaned_content)
        except json.JSONDecodeError as e:
            logger.warning(f"[{context}] Failed to parse JSON directly: {str(e)}. Attempting to extract JSON with regex.")
            # Trích xuất JSON bằng regex
            json_match = re.search(r'\{.*\}', cleaned_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    response_dict = json.loads(json_str)
                except json.JSONDecodeError as e2:
                    logger.error(f"[{context}] Failed to parse extracted JSON: {str(e2)}. Raw content: {cleaned_content[:200]}...")
                    return {"status": "error", "message": f"Phản hồi {context} không phải JSON hợp lệ", "data": {}}
            else:
                logger.error(f"[{context}] No JSON found in response: {cleaned_content[:200]}...")
                return {"status": "error", "message": f"Không tìm thấy JSON trong phản hồi {context}", "data": {}}

        # Validate và ép kiểu bằng Pydantic
        try:
            agent_response = AgentResponse(**response_dict)
            logger.info(f"[{context}] Successfully parsed and validated response: {json.dumps(response_dict, ensure_ascii=False)}")
            return response_dict
        except ValidationError as e:
            logger.error(f"[{context}] Response validation failed: {str(e)}. Raw response: {json.dumps(response_dict, ensure_ascii=False)}")
            return {"status": "error", "message": f"Phản hồi {context} không đúng cấu trúc mong muốn: {str(e)}", "data": {}}

    except Exception as e:
        logger.error(f"[{context}] Unexpected error while parsing response: {str(e)}. Raw content: {response_content[:200]}...")
        return {"status": "error", "message": f"Lỗi không xác định khi xử lý phản hồi {context}: {str(e)}", "data": {}}