# tools/rag_tool.py
import os
import sys
from pathlib import Path
from phi.tools import Toolkit
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import re
from pdf2image import convert_from_path
import pytesseract
from config.env import QDRANT_HOST, QDRANT_PORT, RAG_DATA_DIR
from utils.logging import setup_logging
from utils.validators import validate_rag_dir
from utils.company_mapping import build_company_mapping, map_company_name, normalize_company_name
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

logger = setup_logging()

class CustomRAGTool(Toolkit):
    def __init__(self):
        super().__init__(name="rag_tool")
        try:
            validate_rag_dir(RAG_DATA_DIR)
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            self.collection_name = "financial_docs"
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self._create_collection()
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
                        size=384,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to create Qdrant collection: {str(e)}")
            raise

    def _load_documents(self):
        """Load and process PDF documents from RAG_DATA_DIR, extract text via OCR, create embeddings, and upsert into Qdrant."""
        try:
            company_mapping = build_company_mapping()
            logger.info(f"Found {len(company_mapping)} companies in RAG_DATA_DIR: {list(company_mapping.values())}")
            logger.debug(f"Company mapping: {company_mapping}")
            
            documents = []
            doc_ids = []
            doc_names = []
            companies = []
            chunk_id = 0
            processed_files = []
            failed_files = []
            CHUNK_SIZE = 1000
            BATCH_SIZE = 100
            MAX_PAGES = 100

            def clean_text(text):
                text = re.sub(r'\s+', ' ', text)
                text = re.sub(r'(?<=\w)- (?=\w)', '', text)
                text = re.sub(r'[^\w\s.,!?&-]', '', text)
                return text.strip()

            def detect_table_lines(text):
                lines = text.split('\n')
                table_lines = []
                for line in lines:
                    if '|' in line or len(line.split()) > 3 and len(set(len(word) for word in line.split() if word)) < 3:
                        table_lines.append(line.strip())
                if table_lines:
                    return '\n'.join(table_lines) + '\n'
                return ''

            def chunk_text(text, chunk_size=CHUNK_SIZE):
                chunks = []
                while len(text) > chunk_size:
                    match = re.search(r'([.!?\n])\s', text[:chunk_size][::-1])
                    if match:
                        last_period_index = chunk_size - match.start() - 1
                    else:
                        space_index = text[:chunk_size].rfind(' ')
                        last_period_index = space_index if space_index != -1 else chunk_size
                    
                    chunk = text[:last_period_index].strip()
                    if len(chunk) >= 50:
                        chunks.append(chunk)
                    text = text[last_period_index+1:].lstrip()
                
                if len(text.strip()) >= 50:
                    chunks.append(text.strip())
                return chunks

            logger.info(f"Files in RAG_DATA_DIR: {os.listdir(RAG_DATA_DIR)}")
            for filename in os.listdir(RAG_DATA_DIR):
                if not filename.endswith(".pdf"):
                    logger.debug(f"Skipping non-PDF file: {filename}")
                    continue
                filepath = os.path.join(RAG_DATA_DIR, filename)
                try:
                    text = ""
                    has_content = False
                    logger.info(f"Processing {filename} with OCR")
                    try:
                        images = convert_from_path(filepath, first_page=1, last_page=MAX_PAGES)
                        for i, image in enumerate(images):
                            ocr_text = pytesseract.image_to_string(image, lang='eng')
                            if ocr_text.strip():
                                table_text = detect_table_lines(ocr_text)
                                non_table_text = clean_text(re.sub(r'\n\s*\n', '\n', ocr_text))
                                text += table_text + non_table_text + '\n'
                                has_content = True
                            logger.debug(f"OCR page {i+1}/{len(images)} of {filename}: {len(ocr_text)} characters, sample={ocr_text[:100]}")
                    except Exception as e:
                        logger.error(f"OCR failed for {filename}: {str(e)}, filepath={filepath}")
                        failed_files.append(filename)
                        continue

                    if not has_content:
                        logger.warning(f"No content extracted from {filename} via OCR.")
                        failed_files.append(filename)
                        continue

                    raw_company = filename.replace(".pdf", "").split("_")[0]
                    company = map_company_name(raw_company, company_mapping)
                    if not company:
                        logger.error(f"Failed to map company for {filename}: raw_company={raw_company}, company_mapping={company_mapping}")
                        failed_files.append(filename)
                        continue
                    year_match = re.search(r"\b(202[0-5])\b", text)
                    year = int(year_match.group(1)) if year_match else 2024
                    report_type = "annual_report" if "Annual" in filename.lower() else "financial_report"
                    logger.debug(f"Processed {filename}: raw_company={raw_company}, company={company}, year={year}, report_type={report_type}, text_length={len(text)}")

                    chunks = chunk_text(text, CHUNK_SIZE)
                    logger.debug(f"Generated {len(chunks)} chunks for {filename}: sample={chunks[0][:100] if chunks else ''}")

                    for chunk in chunks:
                        documents.append(chunk)
                        doc_ids.append(chunk_id)
                        doc_names.append(filename)
                        companies.append(company)
                        chunk_id += 1
                    processed_files.append(filename)

                except Exception as e:
                    logger.error(f"Failed to process PDF {filename}: {str(e)}, filepath={filepath}")
                    failed_files.append(filename)
                    continue

            if not documents:
                logger.warning(f"No valid PDF documents found in RAG_DATA_DIR: {RAG_DATA_DIR}")
                return

            logger.info(f"Processed {len(processed_files)} files: {processed_files}")
            if failed_files:
                logger.warning(f"Failed to process {len(failed_files)} files: {failed_files}")

            try:
                embeddings = self.model.encode(documents, show_progress_bar=True)
            except Exception as e:
                logger.error(f"Failed to create embeddings: {str(e)}")
                raise

            points = [
                models.PointStruct(
                    id=doc_id,
                    vector=embedding.tolist(),
                    payload={
                        "text": doc_text,
                        "filename": doc_name,
                        "company": company,
                        "year": year,
                        "report_type": report_type,
                        "keywords": ["revenue", "profit", "financial", "annual"],
                        "chunk_id": doc_id
                    }
                )
                for doc_id, embedding, doc_text, doc_name, year, company in zip(
                    doc_ids, embeddings, documents, doc_names,
                    [int(re.search(r"\b(202[0-5])\b", doc_text).group(1)) if re.search(r"\b(202[0-5])\b", doc_text) else 2024 for doc_text in documents],
                    companies
                )
            ]

            for point in points[:5]:
                logger.debug(f"Upserting point: id={point.id}, company={point.payload['company']}, filename={point.payload['filename']}")

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

    def run(self, query: str, company: str = None, tickers: list = None) -> list:
        """Retrieve top 5 closest documents from Qdrant based on query embedding."""
        try:
            logger.info(f"Executing RAG query: {query}")
            query_embedding = self.model.encode(query).tolist()

            # Kiểm tra xem Qdrant có tài liệu không
            collections = self.client.get_collections()
            if not any(col.name == self.collection_name for col in collections.collections):
                logger.error(f"Qdrant collection {self.collection_name} does not exist")
                return [{"error": "No documents loaded in Qdrant. Please upload financial reports to ./data/rag_documents and reload."}]

            # Tìm kiếm không dùng bộ lọc, lấy top 5 tài liệu
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=5  # Lấy 5 tài liệu gần nhất
            )

            if not search_result:
                logger.warning(f"No documents found for query: {query}")
                return [{"error": "No relevant financial reports found in Qdrant. Please ensure relevant documents are uploaded to ./data/rag_documents."}]

            # Trả về danh sách tài liệu với nội dung và metadata
            results = [
                {
                    "document": hit.payload["text"],
                    "filename": hit.payload["filename"],
                    "company": hit.payload["company"]
                }
                for hit in search_result
            ]
            logger.info(f"Returning {len(results)} documents: {[r['filename'] for r in results]}")
            return results

        except Exception as e:
            logger.error(f"Error executing RAG query: {str(e)}")
            return [{"error": f"Error retrieving documents: {str(e)}"}]