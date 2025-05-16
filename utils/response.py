import json
from typing import Any

def standardize_response(status: str, message: str, result: Any) -> dict:
    """
    Chuẩn hóa phản hồi JSON với cấu trúc thống nhất.

    Args:
        status (str): Trạng thái của phản hồi ('success' hoặc 'error').
        message (str): Thông báo ngắn mô tả kết quả.
        result (Any): Dữ liệu kết quả, có thể là dict, list, str, hoặc None.

    Returns:
        dict: Phản hồi JSON với cấu trúc { "status": str, "message": str, "data": { "result": Any } }
    """
    if status not in ["success", "error"]:
        raise ValueError("Status phải là 'success' hoặc 'error'")

    return {
        "status": status,
        "message": message,
        "data": {"result": result}
    }