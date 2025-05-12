from dotenv import load_dotenv
import os

load_dotenv()

Groq_API_KEY = os.getenv("GROQ_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
RAG_DATA_DIR = os.getenv("RAG_DATA_DIR")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")