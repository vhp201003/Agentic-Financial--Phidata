# scripts/clean_qdrant_collection.py
import sys
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Thêm thư mục gốc dự án vào sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from utils.logging import setup_logging
from config.env import QDRANT_HOST, QDRANT_PORT

logger = setup_logging()

def clean_qdrant_collection(collection_name="financial_docs"):
    """Xóa dữ liệu trong collection của Qdrant."""
    try:
        # Khởi tạo Qdrant client
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        # Kiểm tra collection tồn tại
        collections = client.get_collections()
        if collection_name not in [col.name for col in collections.collections]:
            logger.warning(f"Collection {collection_name} không tồn tại trong Qdrant")
            return
        
        # Xóa collection
        client.delete_collection(collection_name=collection_name)
        logger.info(f"Đã xóa collection {collection_name} trong Qdrant")

        # Tạo lại collection trống
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=384,  # Kích thước vector của all-MiniLM-L6-v2
                distance=models.Distance.COSINE
            )
        )
        logger.info(f"Đã tạo lại collection {collection_name} trống trong Qdrant")
    
    except Exception as e:
        logger.error(f"Lỗi khi xóa collection {collection_name}: {str(e)}")
        raise

if __name__ == "__main__":
    clean_qdrant_collection()