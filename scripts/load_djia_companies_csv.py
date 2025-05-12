import os
import sys
from pathlib import Path

# Thêm thư mục gốc dự án vào sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import pandas as pd
from sqlalchemy import create_engine, text
from config.env import DATABASE_URL
from utils.logging import setup_logging
from utils.validators import validate_database_url

logger = setup_logging()

def truncate_companies_table(conn):
    try:
        conn.execute(text("TRUNCATE TABLE companies CASCADE"))
        conn.commit()
        logger.info("Table 'companies' truncated successfully")
    except Exception as e:
        logger.error(f"Failed to truncate table 'companies': {str(e)}")
        raise

def load_companies_csv(csv_file):
    try:
        # Đọc CSV
        df = pd.read_csv(csv_file)
        logger.info(f"Read {len(df)} records from {csv_file}")

        # Đổi tên cột để khớp với schema
        df = df.rename(columns={
            "52_week_high": "week_high_52",
            "52_week_low": "week_low_52"
        })

        # Chuyển đổi kiểu dữ liệu
        df["market_cap"] = df["market_cap"].astype("int64")
        df["pe_ratio"] = df["pe_ratio"].astype("float64")
        df["dividend_yield"] = df["dividend_yield"].astype("float64")
        df["week_high_52"] = df["week_high_52"].astype("float64")
        df["week_low_52"] = df["week_low_52"].astype("float64")

        return df
    except Exception as e:
        logger.error(f"Failed to read CSV {csv_file}: {str(e)}")
        raise

def save_to_postgres(df, conn):
    try:
        # Chèn dữ liệu vào bảng companies
        df.to_sql("companies", conn, if_exists="append", index=False)
        logger.info(f"Inserted {len(df)} records into 'companies' table")
    except Exception as e:
        logger.error(f"Failed to insert data into 'companies' table: {str(e)}")
        raise

def main():
    csv_file = os.path.join(BASE_DIR, "data", "djia_companies_20250426.csv")
    
    try:
        # Kiểm tra file CSV tồn tại
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

        # Kết nối PostgreSQL
        validate_database_url(DATABASE_URL)
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            # Truncate bảng companies
            truncate_companies_table(conn)
            
            # Đọc và xử lý CSV
            df = load_companies_csv(csv_file)
            
            # Lưu vào PostgreSQL
            save_to_postgres(df, conn)
        
        logger.info("Successfully loaded companies data from CSV to PostgreSQL")
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main()