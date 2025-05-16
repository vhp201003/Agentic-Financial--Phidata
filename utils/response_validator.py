# utils/response_validator.py
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict
import re
import json
from utils.logging import setup_logging

logger = setup_logging()

class OrchestratorData(BaseModel):
    agents: List[str]
    sub_queries: Dict[str, str]
    general_response: Optional[str] = None

class OrchestratorResponse(BaseModel):
    status: str = Field(..., pattern="^(success|error)$")
    message: str = Field(..., min_length=1)
    data: OrchestratorData

class RagData(BaseModel):
    rag_query: str = Field(..., min_length=1)
    company: str = Field(..., min_length=1)
    description: Optional[str] = None
    result: List = Field(default_factory=list)
    suggestion: Optional[str] = None

class RagResponse(BaseModel):
    status: str = Field(..., pattern="^(success|error)$")
    message: str = Field(..., min_length=1)
    data: RagData

class Text2SQLData(BaseModel):
    tables: List[str]
    sql_query: str = Field(..., min_length=1)
    result: List = Field(default_factory=list)

class Text2SQLResponse(BaseModel):
    status: str = Field(..., pattern="^(success|error)$")
    message: str = Field(..., min_length=1)
    data: Text2SQLData

def clean_and_extract_json(response: str) -> Optional[dict]:
    """Trích xuất JSON từ response, làm sạch markdown/code."""
    try:
        # Loại bỏ markdown hoặc code fences
        cleaned = re.sub(r'```(python|json)?\n|\n```', '', response).strip()
        
        # Tìm khối JSON bằng regex
        json_match = re.search(r'\{[\s\S]*\}', cleaned)
        if not json_match:
            logger.error(f"No JSON found in response: {cleaned[:100]}...")
            return None
        
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {str(e)}, json_str: {json_str[:100]}...")
            return None
    except Exception as e:
        logger.error(f"Error cleaning response: {str(e)}, response: {response[:100]}...")
        return None

def validate_response(response: str, response_type: str) -> Optional[BaseModel]:
    """Validate response và trả về Pydantic model hoặc None nếu sai."""
    json_data = clean_and_extract_json(response)
    if not json_data:
        return None
    
    try:
        if response_type == "orchestrator":
            return OrchestratorResponse(**json_data)
        elif response_type == "rag":
            return RagResponse(**json_data)
        elif response_type == "text2sql":
            return Text2SQLResponse(**json_data)
        else:
            logger.error(f"Unknown response type: {response_type}")
            return None
    except ValidationError as e:
        logger.error(f"Pydantic validation failed for {response_type}: {str(e)}, json_data: {json_data}")
        return None