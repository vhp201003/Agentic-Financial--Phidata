from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://admin:admin@localhost:5432/finance_db ")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
RAG_DATA_DIR = os.getenv("RAG_DATA_DIR", os.path.join(BASE_DIR, "data", "rag_documents"))
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")