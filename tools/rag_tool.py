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

    def run(self, query: str, company: str = None, description: str = None) -> list:
        """Retrieve top 5 closest documents from Qdrant and return their content along with metadata."""
        THRESHOLD = 0.7  # Threshold for similarity score
        try:
            logger.info(f"Executing RAG query: {query}")
            self.client.get_collections()
            company_mapping = build_company_mapping()

            if not company and " for " in query:
                company_part = query.split(" for ")[-1]
                company = company_part.strip()
                if " in " in company_part:
                    company = company_part.split(" in ")[0].strip()
                elif " with " in company_part:
                    company = company_part.split(" with ")[0].strip()

            if company:
                company = map_company_name(company, company_mapping)
                if not company:
                    logger.warning(f"Failed to map company: {company}")
                    company = company_part.strip()
                logger.debug(f"Mapped company for search: {company}")

            year = None
            year_match = re.search(r"in (\d{4})", query)
            year = int(year_match.group(1)) if year_match else None

            if not description and " with " in query:
                description = query.split(" with ")[1].split(" in ")[0].strip() if " in " in query else query.split(" with ")[1].strip()
            elif description == "report":
                description = None

            logger.debug(f"Query: {query}, company: {company}, description: {description}, year: {year}")
            query_embedding = self.model.encode(query).tolist()

            all_companies_check = self.client.scroll(
                collection_name=self.collection_name,
                limit=100
            )
            qdrant_companies = list(set([hit.payload.get("company", "") for hit in all_companies_check[0]]))
            logger.debug(f"Companies in Qdrant: {qdrant_companies}")

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
                    return [{"error": f"No relevant documents found for {company} in the system. Ensure '{company}.pdf' is uploaded and processed."}]

            filter_conditions = []
            if company:
                company_variants = [company]
                if company in company_mapping.values():
                    company_variants.append(company_mapping.get(normalize_company_name(company), company))
                company_variants.extend([f"{company} Inc.", f"{company} Corporation", f"{company} Co.", f"The {company}"])
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

            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=models.Filter(must=filter_conditions) if filter_conditions else None,
                limit=5  # Lấy tối đa 5 tài liệu
            )

            filtered_results = search_result
            if description:
                description_embedding = self.model.encode(description).tolist()
                filtered_results = []
                for hit in search_result:
                    text_embedding = self.model.encode(hit.payload["text"]).tolist()
                    similarity = cosine_similarity([description_embedding], [text_embedding])[0][0]
                    if similarity > THRESHOLD:
                        filtered_results.append(hit)
                    logger.debug(f"Similarity for {hit.payload['filename']}: {similarity}")

            logger.debug(f"Found {len(search_result)} results before filtering, {len(filtered_results)} after filtering")
            filtered_results = filtered_results[:5]  # Đảm bảo tối đa 5 tài liệu

            if not filtered_results:
                logger.warning(f"No relevant documents found for query: {query}")
                return [{"error": f"No relevant documents found for {query} in the system."}]

            # Trả về danh sách các tài liệu với nội dung và metadata
            results = [
                {
                    "document": hit.payload["text"],  # Đổi key từ 'text' thành 'document'
                    "filename": hit.payload["filename"],
                    "company": hit.payload["company"]
                }
                for hit in filtered_results
            ]
            return results

        except Exception as e:
            logger.error(f"Error executing RAG query: {str(e)}")
            return [{"error": f"Error retrieving documents: {str(e)}"}]