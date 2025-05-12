# import os
# import sys
# from pathlib import Path

# # Thêm thư mục gốc dự án vào sys.path
# BASE_DIR = Path(__file__).resolve().parent.parent
# sys.path.append(str(BASE_DIR))

# from phi.tools import Toolkit
# from qdrant_client import QdrantClient
# from qdrant_client.http import models
# from sentence_transformers import SentenceTransformer
# import PyPDF2
# import httpx
# from config.env import QDRANT_HOST, QDRANT_PORT, RAG_DATA_DIR
# from utils.logging import setup_logging
# from utils.validators import validate_rag_dir

# logger = setup_logging()

# class CustomRAGTool(Toolkit):
#     def __init__(self, proxies: dict = None):
#         super().__init__(name="rag_tool")
#         try:
#             # Kiểm tra thư mục RAG
#             validate_rag_dir(RAG_DATA_DIR)
            
#             # Khởi tạo HTTP client với proxy (nếu có)
            
#             # Khởi tạo Qdrant client
#             self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
#             self.collection_name = "financial_docs"
            
#             # Khởi tạo mô hình embedding
#             self.model = SentenceTransformer("all-MiniLM-L6-v2")
            
#             # Tạo collection nếu chưa tồn tại
#             self._create_collection()
            
#             # Tải và xử lý tài liệu
#             self._load_documents()
            
#             # Đăng ký hàm run
#             self.register(self.run)
            
#             logger.info("RAG tool initialized successfully")
#         except Exception as e:
#             logger.error(f"Failed to initialize RAG tool: {str(e)}")
#             raise

#     def _create_collection(self):
#         try:
#             collections = self.client.get_collections()
#             if self.collection_name not in [col.name for col in collections.collections]:
#                 self.client.create_collection(
#                     collection_name=self.collection_name,
#                     vectors_config=models.VectorParams(
#                         size=384,  # Kích thước vector của all-MiniLM-L6-v2
#                         distance=models.Distance.COSINE
#                     )
#                 )
#                 logger.info(f"Created Qdrant collection: {self.collection_name}")
#         except Exception as e:
#             logger.error(f"Failed to create Qdrant collection: {str(e)}")
#             raise

#     def _load_documents(self):
#         try:
#             # Đọc tất cả file PDF trong thư mục RAG_DATA_DIR
#             documents = []
#             doc_ids = []
#             for idx, filename in enumerate(os.listdir(RAG_DATA_DIR)):
#                 if filename.endswith(".pdf"):
#                     filepath = os.path.join(RAG_DATA_DIR, filename)
#                     with open(filepath, "rb") as file:
#                         pdf_reader = PyPDF2.PdfReader(file)
#                         text = ""
#                         for page in pdf_reader.pages:
#                             text += page.extract_text() + "\n"
#                         documents.append(text)
#                         doc_ids.append(idx)
            
#             if not documents:
#                 logger.warning("No PDF documents found in RAG_DATA_DIR")
#                 return
            
#             # Tạo embedding cho các tài liệu
#             embeddings = self.model.encode(documents, show_progress_bar=True)
            
#             # Lưu tài liệu và vector vào Qdrant
#             points = [
#                 models.PointStruct(
#                     id=doc_id,
#                     vector=embedding.tolist(),
#                     payload={"text": doc_text}
#                 )
#                 for doc_id, embedding, doc_text in zip(doc_ids, embeddings, documents)
#             ]
#             self.client.upsert(
#                 collection_name=self.collection_name,
#                 points=points
#             )
#             logger.info(f"Loaded {len(documents)} documents into Qdrant")
#         except Exception as e:
#             logger.error(f"Failed to load documents into Qdrant: {str(e)}")
#             raise

#     def run(self, query: str) -> str:
#         """Retrieve information from financial PDFs using Qdrant.

#         Args:
#             query (str): The query to search for.

#         Returns:
#             str: Retrieved information from the PDFs.
#         """
#         try:
#             # Tạo embedding cho truy vấn
#             query_embedding = self.model.encode(query).tolist()
            
#             # Tìm kiếm các vector gần nhất
#             search_result = self.client.search(
#                 collection_name=self.collection_name,
#                 query_vector=query_embedding,
#                 limit=3  # Lấy 3 kết quả gần nhất
#             )
            
#             # Trích xuất nội dung từ kết quả
#             results = [hit.payload["text"] for hit in search_result]
#             return "\n".join(results) if results else "No relevant documents found."
#         except Exception as e:
#             logger.error(f"Error retrieving data: {str(e)}")
#             return f"Error retrieving data: {str(e)}"

import os
import sys
from pathlib import Path
from phi.tools import Toolkit
from utils.logging import setup_logging
import json

logger = setup_logging()

class CustomRAGTool(Toolkit):
    def __init__(self):
        super().__init__(name="rag_tool")
        logger.info("RAG tool initialized successfully")

    def run(self, query: str) -> str:
        """Run a RAG query and return JSON with source and document details."""
        try:
            # Bước 1: Phân tích query và xác định tài liệu cần truy vấn
            # (Giả lập logic phân tích)
            sources = ["doc1", "doc2"]
            documents = {
                "doc1": "Financial report Q1 2025 for Apple: Revenue $100B, Profit $20B.",
                "doc2": "Annual report 2024 for Apple: Revenue $391B, Profit $93.74B."
            }
            logger.info(f"RAG Plan: {{ 'sources': {sources}, 'documents': {documents} }}")

            # Bước 2: Thực thi truy vấn RAG (giả lập)
            summary = "Apple's Q1 2025 financial report shows a revenue of $100B and profit of $20B, while the 2024 annual report indicates a revenue of $391B and profit of $93.74B."
            
            return json.dumps({
                "status": "success",
                "message": "RAG query executed successfully",
                "data": {
                    "sources": sources,
                    "documents": documents,
                    "summary": summary
                },
                "source": "rag"
            })
        except Exception as e:
            logger.error(f"Error executing RAG query: {str(e)}")
            return json.dumps({
                "status": "error",
                "message": f"Error executing RAG query: {str(e)}",
                "data": None
            })