import os
import sys
from pathlib import Path

# Thêm thư mục gốc dự án vào sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from sqlalchemy import create_engine, text
from config.env import DATABASE_URL
from utils.logging import setup_logging
from utils.validators import validate_database_url

logger = setup_logging()

def init_database():
    try:
        validate_database_url(DATABASE_URL)
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            # Create companies table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS companies (
                    symbol VARCHAR(10) PRIMARY KEY,
                    name VARCHAR(255),
                    sector VARCHAR(100),
                    industry VARCHAR(100),
                    country VARCHAR(100),
                    website VARCHAR(255),
                    market_cap BIGINT,
                    pe_ratio DECIMAL(10, 2),
                    dividend_yield DECIMAL(5, 2),
                    week_high_52 DECIMAL(10, 2),
                    week_low_52 DECIMAL(10, 2),
                    description TEXT
                )
            """))
            # Create stock_prices table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS stock_prices (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(10) REFERENCES companies(symbol),
                    date DATE,
                    open_price DECIMAL(10, 2),
                    high_price DECIMAL(10, 2),
                    low_price DECIMAL(10, 2),
                    close_price DECIMAL(10, 2),
                    volume BIGINT,
                    dividends DECIMAL(10, 2),
                    stock_splits DECIMAL(10, 2),
                    UNIQUE(symbol, date)
                )
            """))
            conn.commit()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise

if __name__ == "__main__":
    init_database()