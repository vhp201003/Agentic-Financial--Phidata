import os
import sys
from pathlib import Path

# Thêm thư mục gốc dự án vào sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from tools.rag_tool import CustomRAGTool
from utils.logging import setup_logging

logger = setup_logging()

def populate_rag():
    try:
        # Khởi tạo RAG tool để tải tài liệu
        rag_tool = CustomRAGTool()
        logger.info("RAG documents populated successfully")
    except Exception as e:
        logger.error(f"Failed to populate RAG: {str(e)}")
        raise

if __name__ == "__main__":
    populate_rag()