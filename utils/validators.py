import os

def validate_rag_dir(rag_dir: str) -> None:
    if not os.path.exists(rag_dir):
        raise ValueError(f"RAG directory {rag_dir} does not exist")

def validate_database_url(db_url: str) -> None:
    if not db_url:
        raise ValueError("DATABASE_URL is not set")