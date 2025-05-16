# tools/rag_tool.py
import os
import sys
from pathlib import Path
from phi.tools import Toolkit
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import re
import pdfplumber
from config.env import QDRANT_HOST, QDRANT_PORT, RAG_DATA_DIR
from utils.logging import setup_logging
from utils.validators import validate_rag_dir
from utils.company_mapping import build_company_mapping, map_company_name, normalize_company_name
import json
from sklearn.metrics.pairwise import cosine_similarity

# Thêm thư mục gốc dự án vào sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

logger = setup_logging()

class CustomRAGTool(Toolkit):
    def __init__(self):
        super().__init__(name="rag_tool")
        try:
            # Kiểm tra thư mục RAG
            validate_rag_dir(RAG_DATA_DIR)
            
            # Khởi tạo Qdrant client
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            self.collection_name = "financial_docs"
            
            # Khởi tạo mô hình embedding
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            
            # Tạo collection nếu chưa tồn tại
            self._create_collection()
            
            # # Tải và xử lý tài liệu
            # self._load_documents()
            
            logger.info("RAG tool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG tool: {str(e)}")
            raise

    def _create_collection(self):
        try:
            collections = self.client.get_collections()
            if self.collection_name not in [col.name for col in collections.collections]:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=384,  # Kích thước vector của all-MiniLM-L6-v2
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to create Qdrant collection: {str(e)}")
            raise

    def _load_documents(self):
        """Load and process PDF documents from RAG_DATA_DIR, extract text and tables, create embeddings, and upsert into Qdrant with detailed metadata."""
        try:
            # Tạo ánh xạ công ty từ thư mục PDF
            company_mapping = build_company_mapping()
            logger.info(f"Found {len(company_mapping)} companies in RAG_DATA_DIR: {list(company_mapping.values())}")
            
            documents = []
            doc_ids = []
            doc_names = []
            chunk_id = 0
            processed_files = []
            failed_files = []
            CHUNK_SIZE = 1000  # Tăng kích thước chunk
            BATCH_SIZE = 100  # Số points mỗi lần upsert

            def clean_text(text):
                """Làm sạch text, giữ tên riêng, loại khoảng trắng thừa."""
                text = re.sub(r'\s+', ' ', text)
                text = re.sub(r'(?<=\w)- (?=\w)', '', text)  # Loại gạch nối giữa từ
                return text.strip()

            def chunk_text(text, chunk_size=CHUNK_SIZE):
                """Chia text thành chunk, giữ ngữ nghĩa."""
                chunks = []
                while len(text) > chunk_size:
                    # Tìm điểm cắt tự nhiên (dấu câu hoặc xuống dòng)
                    match = re.search(r'([.!?\n])\s', text[:chunk_size][::-1])
                    if match:
                        last_period_index = chunk_size - match.start() - 1
                    else:
                        space_index = text[:chunk_size].rfind(' ')
                        last_period_index = space_index if space_index != -1 else chunk_size
                    
                    chunk = text[:last_period_index].strip()
                    if len(chunk) >= 50:  # Loại chunk quá ngắn
                        chunks.append(chunk)
                    text = text[last_period_index+1:].lstrip()
                
                if len(text.strip()) >= 50:
                    chunks.append(text.strip())
                return chunks

            for filename in os.listdir(RAG_DATA_DIR):
                if not filename.endswith(".pdf"):
                    logger.debug(f"Skipping non-PDF file: {filename}")
                    continue
                filepath = os.path.join(RAG_DATA_DIR, filename)
                try:
                    # Kiểm tra file hợp lệ
                    with pdfplumber.open(filepath) as pdf:
                        if not pdf.pages:
                            logger.warning(f"Empty PDF: {filename}")
                            failed_files.append(filename)
                            continue
                        text = ""
                        has_content = False
                        for page in pdf.pages:
                            # Extract tables
                            tables = page.extract_tables()
                            for table in tables:
                                cleaned_table = [[str(cell) if cell is not None else "" for cell in row] for row in table]
                                table_text = "\n".join([" | ".join(row) for row in cleaned_table]) + "\n"
                                text += table_text
                                if table_text.strip():
                                    has_content = True
                            # Extract free text
                            page_text = page.extract_text()
                            if page_text:
                                text += clean_text(page_text) + "\n"
                                has_content = True
                        
                        if not has_content:
                            logger.warning(f"No text or tables extracted from {filename}. Likely image-based PDF.")
                            failed_files.append(filename)
                            continue

                    # Extract metadata from content
                    raw_company = filename.replace(".pdf", "").split("_")[0]
                    company = map_company_name(raw_company, company_mapping)
                    year_match = re.search(r"\b(202[0-5])\b", text)
                    year = int(year_match.group(1)) if year_match else 2024
                    report_type = "annual_report" if "Annual" in filename.lower() else "financial_report"
                    logger.debug(f"Processed {filename}: company={company}, year={year}, report_type={report_type}, text_length={len(text)}")

                    # Chia text thành chunk
                    chunks = chunk_text(text, CHUNK_SIZE)
                    logger.debug(f"Generated {len(chunks)} chunks for {filename}: sample={chunks[0][:100] if chunks else ''}")

                    for chunk in chunks:
                        documents.append(chunk)
                        doc_ids.append(chunk_id)
                        doc_names.append(filename)
                        chunk_id += 1
                    processed_files.append(filename)

                except Exception as e:
                    logger.error(f"Failed to process PDF {filename}: {str(e)}")
                    failed_files.append(filename)
                    continue

            if not documents:
                logger.warning(f"No valid PDF documents found in RAG_DATA_DIR: {RAG_DATA_DIR}")
                return

            logger.info(f"Processed {len(processed_files)} files: {processed_files}")
            if failed_files:
                logger.warning(f"Failed to process {len(failed_files)} files: {failed_files}")

            # Create embeddings
            try:
                embeddings = self.model.encode(documents, show_progress_bar=True)
            except Exception as e:
                logger.error(f"Failed to create embeddings: {str(e)}")
                raise

            # Prepare points for Qdrant
            points = [
                models.PointStruct(
                    id=doc_id,
                    vector=embedding.tolist(),
                    payload={
                        "text": doc_text,
                        "filename": doc_name,
                        "company": map_company_name(doc_name.replace(".pdf", "").split("_")[0], company_mapping),
                        "year": year,
                        "report_type": report_type,
                        "keywords": ["revenue", "profit", "financial", "annual"],
                        "chunk_id": doc_id
                    }
                )
                for doc_id, embedding, doc_text, doc_name, year in zip(
                    doc_ids, embeddings, documents, doc_names,
                    [int(re.search(r"\b(202[0-5])\b", doc_text).group(1)) if re.search(r"\b(202[0-5])\b", doc_text) else 2024 for doc_text in documents]
                )
            ]

            # Upsert to Qdrant in batches
            try:
                for i in range(0, len(points), BATCH_SIZE):
                    batch = points[i:i + BATCH_SIZE]
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=batch
                    )
                    logger.info(f"Upserted batch {i//BATCH_SIZE + 1} with {len(batch)} points")
                logger.info(f"Loaded {len(documents)} document chunks into Qdrant")
            except Exception as e:
                logger.error(f"Failed to upsert into Qdrant: {str(e)}")
                raise

        except Exception as e:
            logger.error(f"Unexpected error in _load_documents: {str(e)}")
            raise
    
    def run(self, query: str, company: str = None, description: str = None) -> str:
        """Retrieve and summarize information from financial PDFs using Qdrant."""
        try:
            logger.info(f"Executing RAG query: {query}")
            # Kiểm tra Qdrant connection
            self.client.get_collections()
            
            # Tạo ánh xạ công ty từ thư mục PDF
            company_mapping = build_company_mapping()
            
            # Phân tích sub-query nếu không có company/description được truyền vào
            if not company and " for " in query:
                company_part = query.split(" for ")[-1]
                company = company_part.split()[0] if company_part else None
                if " in " in company_part:
                    company = company_part.split(" in ")[0].strip()
                elif " with " in company_part:
                    company = company_part.split(" with ")[0].strip()
            
            # Ánh xạ tên công ty
            if company:
                company = map_company_name(company, company_mapping)
                logger.debug(f"Mapped company for search: {company}")

            year = None
            year_match = re.search(r"in (\d{4})", query)
            year = int(year_match.group(1)) if year_match else None

            if not description and " with " in query:
                description = query.split(" with ")[1].split(" in ")[0].strip() if " in " in query else query.split(" with ")[1].strip()
            elif description == "report":  # Xử lý intent 'báo cáo', 'tài chính'
                description = None  # Bỏ description mặc định để tránh lọc quá nghiêm ngặt

            logger.debug(f"Query: {query}, company: {company}, description: {description}, year: {year}")
            
            # Tạo embedding cho truy vấn
            query_embedding = self.model.encode(query).tolist()
            
            # Kiểm tra tất cả công ty trong Qdrant để debug
            all_companies_check = self.client.scroll(
                collection_name=self.collection_name,
                limit=100
            )
            qdrant_companies = list(set([hit.payload.get("company", "") for hit in all_companies_check[0]]))
            logger.debug(f"Companies in Qdrant: {qdrant_companies}")

            # Kiểm tra xem công ty có trong Qdrant không
            if company:
                company_check = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=models.Filter(
                        must=[models.FieldCondition(key="company", match=models.MatchText(text=company))]
                    ),
                    limit=1
                )
                if not company_check[0]:
                    logger.warning(f"No documents found for company: {company}")
                    suggestion = f"Try full company name (e.g., 'Apple Inc.') or check if '{company}.pdf' exists in {RAG_DATA_DIR}. "
                    suggestion += f"Ensure documents are indexed by running _load_documents. Current companies in Qdrant: {', '.join(qdrant_companies)}"
                    return json.dumps({
                        "status": "success",
                        "message": f"No documents found for company: {company}",
                        "data": {
                            "sources": [],
                            "documents": {},
                            "metadata": [],
                            "summary": "",
                            "suggestion": suggestion
                        },
                        "source": "rag"
                    }, ensure_ascii=False)

            # Tìm kiếm với filter
            filter_conditions = []
            if company:
                company_variants = [company, normalize_company_name(company)]
                if company in company_mapping.values():
                    company_variants.append(company_mapping.get(normalize_company_name(company), company))
                company_variants.extend([f"{company} Inc.", f"{company} Corporation"])
                company_variants = list(set(company_variants))
                logger.debug(f"Company variants for search: {company_variants}")
                
                company_conditions = [
                    models.FieldCondition(
                        key="company",
                        match=models.MatchText(text=variant)
                    )
                    for variant in company_variants
                ]
                
                filter_conditions.append(
                    models.Filter(should=company_conditions)
                )
            if year:
                filter_conditions.append(
                    models.FieldCondition(
                        key="year",
                        match=models.MatchValue(value=year)
                    )
                )

            # Tìm kiếm trong Qdrant
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=models.Filter(must=filter_conditions) if filter_conditions else None,
                limit=5
            )

            # Lọc thêm dựa trên description nếu có
            filtered_results = search_result
            if description:
                description_embedding = self.model.encode(description).tolist()
                filtered_results = []
                for hit in search_result:
                    text_embedding = self.model.encode(hit.payload["text"]).tolist()
                    similarity = cosine_similarity([description_embedding], [text_embedding])[0][0]
                    if similarity > 0.7:  # Ngưỡng tương đồng
                        filtered_results.append(hit)

            logger.debug(f"Found {len(search_result)} results before filtering, {len(filtered_results)} after filtering")

            # Giới hạn lại số kết quả sau lọc
            filtered_results = filtered_results[:3]

            # Trích xuất nội dung và metadata
            sources = [hit.payload["filename"] for hit in filtered_results]
            documents = {hit.payload["filename"]: hit.payload["text"] for hit in filtered_results}
            metadata = [
                {
                    "filename": hit.payload["filename"],
                    "company": hit.payload["company"],
                    "year": hit.payload["year"],
                    "report_type": hit.payload["report_type"]
                } for hit in filtered_results
            ]
            summary = " ".join([hit.payload["text"][:200] for hit in filtered_results])
            
            if not filtered_results:
                logger.warning(f"No relevant documents found for query: {query}")
                response_data = {
                    "status": "success",
                    "message": "No relevant documents found",
                    "data": {
                        "sources": [],
                        "documents": {},
                        "metadata": [],
                        "summary": "",
                        "suggestion": f"Try broader keywords or check if documents are indexed in {RAG_DATA_DIR}. Ensure '{company}.pdf' is processed by _load_documents. Current companies in Qdrant: {', '.join(qdrant_companies)}"
                    },
                    "source": "rag"
                }
                if company:
                    response_data["data"]["suggestion"] += f" Searched variants: {', '.join(company_variants)}"
                return json.dumps(response_data, ensure_ascii=False)
            
            return json.dumps({
                "status": "success",
                "message": "RAG query executed successfully",
                "data": {
                    "sources": sources,
                    "documents": documents,
                    "metadata": metadata,
                    "summary": summary
                },
                "source": "rag"
            }, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error executing RAG query: {str(e)}")
            return json.dumps({
                "status": "error",
                "message": f"Error executing RAG query: {str(e)}",
                "data": None,
                "source": "rag"
            }, ensure_ascii=False)