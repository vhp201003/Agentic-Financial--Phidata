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

def check_symbol_exists(ticker, conn):
    try:
        result = conn.execute(
            text("SELECT COUNT(*) FROM companies WHERE symbol = :symbol"),
            {"symbol": ticker}
        ).scalar()
        return result > 0
    except Exception as e:
        logger.error(f"Failed to check symbol {ticker} in 'companies': {str(e)}")
        raise

def load_stock_prices_csv(csv_file):
    try:
        # Đọc CSV
        df = pd.read_csv(csv_file)
        logger.info(f"Read {len(df)} records from {csv_file}")

        # Đổi tên cột để khớp với schema
        df = df.rename(columns={
            "Date": "date",
            "Open": "open_price",
            "High": "high_price",
            "Low": "low_price",
            "Close": "close_price",
            "Volume": "volume",
            "Dividends": "dividends",
            "Stock Splits": "stock_splits",
            "Ticker": "symbol"
        })

        # Chuyển đổi kiểu dữ liệu
        df["date"] = pd.to_datetime(df["date"], utc=True)  # Xử lý múi giờ hỗn hợp
        df["open_price"] = df["open_price"].astype("float64")
        df["high_price"] = df["high_price"].astype("float64")
        df["low_price"] = df["low_price"].astype("float64")
        df["close_price"] = df["close_price"].astype("float64")
        df["volume"] = df["volume"].astype("int64")
        df["dividends"] = df["dividends"].astype("float64")
        df["stock_splits"] = df["stock_splits"].astype("float64")

        return df
    except Exception as e:
        logger.error(f"Failed to read CSV {csv_file}: {str(e)}")
        raise

def save_to_postgres(df, conn):
    try:
        valid_rows = 0
        for _, row in df.iterrows():
            if not check_symbol_exists(row["symbol"], conn):
                logger.warning(f"Skipping symbol {row['symbol']}: Not found in 'companies' table")
                continue
            conn.execute(
                text("""
                    INSERT INTO stock_prices (
                        symbol, date, open_price, high_price, low_price, close_price,
                        volume, dividends, stock_splits
                    )
                    VALUES (:symbol, :date, :open_price, :high_price, :low_price, :close_price,
                            :volume, :dividends, :stock_splits)
                    ON CONFLICT (symbol, date) DO UPDATE SET
                        open_price = EXCLUDED.open_price,
                        high_price = EXCLUDED.high_price,
                        low_price = EXCLUDED.low_price,
                        close_price = EXCLUDED.close_price,
                        volume = EXCLUDED.volume,
                        dividends = EXCLUDED.dividends,
                        stock_splits = EXCLUDED.stock_splits
                """),
                {
                    "symbol": row["symbol"],
                    "date": row["date"],
                    "open_price": row["open_price"],
                    "high_price": row["high_price"],
                    "low_price": row["low_price"],
                    "close_price": row["close_price"],
                    "volume": row["volume"],
                    "dividends": row["dividends"],
                    "stock_splits": row["stock_splits"]
                }
            )
            valid_rows += 1
        conn.commit()
        logger.info(f"Inserted {valid_rows} valid records into 'stock_prices' table")
    except Exception as e:
        logger.error(f"Failed to insert data into 'stock_prices' table: {str(e)}")
        raise

def main():
    csv_file = os.path.join(BASE_DIR, "data", "djia_prices_20250426.csv")
    
    try:
        # Kiểm tra file CSV tồn tại
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

        # Kết nối PostgreSQL
        validate_database_url(DATABASE_URL)
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            # Đọc và xử lý CSV
            df = load_stock_prices_csv(csv_file)
            
            # Lưu vào PostgreSQL
            save_to_postgres(df, conn)
        
        logger.info("Successfully loaded stock prices data from CSV to PostgreSQL")
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main()