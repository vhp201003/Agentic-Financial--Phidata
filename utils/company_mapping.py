import os
import re
from pathlib import Path
from config.env import RAG_DATA_DIR
from utils.logging import setup_logging

logger = setup_logging()

def normalize_company_name(name: str) -> str:
    """Chuẩn hóa tên công ty: chuyển về chữ thường, thay dấu câu, loại hậu tố."""
    if not name:
        return name
    name = name.lower().replace("-", " ").replace("&", "and").strip()
    suffixes = r"\b(inc\.|incorporated|corp\.|corporation|ltd\.|limited)\b"
    return re.sub(suffixes, "", name, flags=re.IGNORECASE).strip()

def build_company_mapping() -> dict:
    """Quét RAG_DATA_DIR để tạo ánh xạ từ tên ngắn sang tên đầy đủ."""
    mapping = {}
    if not os.path.exists(RAG_DATA_DIR):
        logger.error(f"RAG_DATA_DIR not found: {RAG_DATA_DIR}")
        return mapping

    for filename in os.listdir(RAG_DATA_DIR):
        if not filename.endswith(".pdf"):
            continue
        # Lấy tên công ty từ tên file (loại bỏ .pdf)
        company_name = filename.replace(".pdf", "")
        # Chuẩn hóa để tạo key ngắn
        normalized_name = normalize_company_name(company_name)
        # Lưu tên đầy đủ làm value
        mapping[normalized_name] = company_name
        # Thêm các biến thể khác (ví dụ: Apple Inc. → Apple)
        if " " not in normalized_name:  # Chỉ thêm biến thể cho tên ngắn
            mapping[company_name.lower()] = company_name
        logger.debug(f"Added mapping: {normalized_name} → {company_name}")
    
    return mapping

def map_company_name(query_company: str, mapping: dict) -> str:
    """Ánh xạ tên công ty từ truy vấn sang tên đầy đủ."""
    if not query_company:
        return query_company
    normalized_query = normalize_company_name(query_company)
    for key, value in mapping.items():
        if normalized_query in key or key in normalized_query:
            logger.debug(f"Mapped company: {query_company} → {value}")
            return value
    logger.warning(f"No mapping found for company: {query_company}")
    return query_company

def check_mapping_integrity(qdrant_client, collection_name: str) -> bool:
    """Kiểm tra tính toàn vẹn: so sánh mapping với metadata Qdrant."""
    mapping = build_company_mapping()
    results = qdrant_client.scroll(collection_name=collection_name, limit=100)
    qdrant_companies = set(hit.payload.get("company", "").lower() for hit in results[0])
    mapping_companies = set(mapping.values())
    missing = mapping_companies - qdrant_companies
    if missing:
        logger.warning(f"Companies in mapping but not in Qdrant: {missing}")
        return False
    logger.info("Company mapping is consistent with Qdrant")
    return True