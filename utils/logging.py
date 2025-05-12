import logging
import sys

def setup_logging():
    # Tạo logger
    logger = logging.getLogger('financial_agent_system')
    logger.setLevel(logging.DEBUG)  # Đặt mức độ log là DEBUG để hiển thị tất cả log

    # Tạo handler để xuất log ra console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    # Định dạng log
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # Xóa các handler cũ (tránh trùng lặp)
    logger.handlers = []
    logger.addHandler(console_handler)

    # Đảm bảo không truyền log lên parent (Uvicorn/FastAPI)
    logger.propagate = False

    return logger