Financial AI Agent
A financial AI system using Phidata with RAG (Qdrant) and TextToSQL (PostgreSQL) to process financial PDFs and query financial data from DJIA companies.
Prerequisites

Docker and Docker Compose
Python 3.10+
Groq API key (get from https://x.ai/api)

Setup

Clone the repository:
git clone <repo>
cd financial-ai-agent


Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:
pip install -r requirements.txt


Create .env file:
cp .env.example .env

Edit .env and add your GROQ_API_KEY.

Add financial PDFs to data/rag_documents/.

Start PostgreSQL and Qdrant:
docker-compose up -d


Initialize PostgreSQL database:
python scripts/init_db.py


Populate financial data:
python scripts/download_djia_companies.py
python scripts/download_djia_stock_prices.py


Populate RAG with PDFs:
python scripts/populate_rag.py


Run the application:
python main.py


Access the Playground UI at http://localhost:8000.


Database Access
Connect to PostgreSQL:
psql -h localhost -U admin -d finance_db

Password: admin
Directory Structure

agents/: Agent definitions (RAG, TextToSQL, Finance, Team)
tools/: Custom tools (RAG with Qdrant, TextToSQL with PostgreSQL)
data/: PDFs for RAG and migrations for PostgreSQL
config/: Environment and database configurations
playground/: Playground UI setup
utils/: Logging and validation utilities
scripts/: Database initialization, RAG population, and data download scripts
tests/: Unit tests for agents

Usage

Query financial reports: "Find the latest financial report for Apple Inc."
Query stock data: "What is the stock price of AAPL today?"
Analyze trends: "Analyze AAPL stock price trends over the past 6 months."

