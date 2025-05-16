# scripts/populate_rag.py
import os
import sys
from pathlib import Path

# Thêm thư mục gốc dự án vào sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))  # Thêm vào đầu sys.path để ưu tiên

from tools.rag_tool import CustomRAGTool
from utils.logging import setup_logging

logger = setup_logging()
logger.debug(f"sys.path: {sys.path}")
logger.debug(f"BASE_DIR: {BASE_DIR}")


def populate_rag():
    """Vector hóa và upsert tài liệu từ RAG_DATA_DIR vào Qdrant."""
    try:
        rag_tool = CustomRAGTool()
        rag_tool._load_documents()
        logger.info("RAG documents populated successfully")
    except Exception as e:
        logger.error(f"Failed to populate RAG documents: {str(e)}")
        raise

if __name__ == "__main__":
    populate_rag()