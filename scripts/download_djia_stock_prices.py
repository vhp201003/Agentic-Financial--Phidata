import os
import sys
from pathlib import Path

# Thêm thư mục gốc dự án vào sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import requests
import pandas as pd
import time
import random
from datetime import datetime, date, timedelta
from sqlalchemy import create_engine, text
from config.env import DATABASE_URL, ALPHA_VANTAGE_API_KEY
from utils.logging import setup_logging
from utils.validators import validate_database_url

logger = setup_logging()

def get_djia_constituents():
    return [
        "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW",
        "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM",
        "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT"
    ]

def download_stock_prices_with_retry(ticker, start_date, end_date, max_retries=5, initial_delay=10):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}"
    retry_count = 0
    delay = initial_delay
    while retry_count < max_retries:
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if "Time Series (Daily)" not in data or "Note" in data:
                logger.warning(f"Rate limit or no data for {ticker}. Response: {data.get('Note', 'No data')}")
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = delay + random.uniform(1, 3)
                    logger.warning(f"Waiting {wait_time:.2f} seconds before retry {retry_count}/{max_retries} for {ticker}")
                    time.sleep(wait_time)
                    delay *= 2
                else:
                    logger.error(f"Max retries reached for {ticker}")
                    return pd.DataFrame({
                        'Date': [], 'Open': [], 'High': [], 'Low': [], 'Close': [], 'Volume': [],
                        'Dividends': [], 'Stock Splits': [], 'Ticker': []
                    })
            else:
                # Chuyển đổi dữ liệu thành DataFrame
                df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
                df = df.reset_index().rename(columns={
                    "index": "Date",
                    "1. open": "Open",
                    "2. high": "High",
                    "3. low": "Low",
                    "4. close": "Close",
                    "5. volume": "Volume"
                })
                df["Date"] = pd.to_datetime(df["Date"])
                df["Open"] = df["Open"].astype(float)
                df["High"] = df["High"].astype(float)
                df["Low"] = df["Low"].astype(float)
                df["Close"] = df["Close"].astype(float)
                df["Volume"] = df["Volume"].astype(int)
                df["Dividends"] = 0.0  # Alpha Vantage không cung cấp dividends ở endpoint này
                df["Stock Splits"] = 0.0  # Không cung cấp stock splits
                df["Ticker"] = ticker
                # Lọc theo khoảng thời gian
                df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
                logger.info(f"Successfully downloaded {ticker} data with {len(df)} records")
                return df[["Date", "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits", "Ticker"]]
        except requests.exceptions.RequestException as e:
            retry_count += 1
            if retry_count < max_retries:
                wait_time = delay + random.uniform(1, 3)
                logger.warning(f"Error for {ticker} (attempt {retry_count}/{max_retries}): {str(e)}. Waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
                delay *= 2
            else:
                logger.error(f"Max retries reached for {ticker}: {str(e)}")
                return pd.DataFrame({
                    'Date': [], 'Open': [], 'High': [], 'Low': [], 'Close': [], 'Volume': [],
                    'Dividends': [], 'Stock Splits': [], 'Ticker': []
                })
        except ValueError as e:
            logger.error(f"Invalid response for {ticker}: {str(e)}")
            return pd.DataFrame({
                'Date': [], 'Open': [], 'High': [], 'Low': [], 'Close': [], 'Volume': [],
                'Dividends': [], 'Stock Splits': [], 'Ticker': []
            })
    return pd.DataFrame({
        'Date': [], 'Open': [], 'High': [], 'Low': [], 'Close': [], 'Volume': [],
        'Dividends': [], 'Stock Splits': [], 'Ticker': []
    })

def save_to_postgres(prices):
    try:
        validate_database_url(DATABASE_URL)
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            for _, row in prices.iterrows():
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
                        "symbol": row["Ticker"],
                        "date": row["Date"],
                        "open_price": row["Open"],
                        "high_price": row["High"],
                        "low_price": row["Low"],
                        "close_price": row["Close"],
                        "volume": row["Volume"],
                        "dividends": row["Dividends"],
                        "stock_splits": row["Stock Splits"]
                    }
                )
            conn.commit()
        logger.info("Stock prices saved to PostgreSQL successfully")
    except Exception as e:
        logger.error(f"Failed to save stock prices to PostgreSQL: {str(e)}")
        raise

def main():
    start_date = "2022-01-01"
    end_date = date.today().strftime('%Y-%m-%d')
    logger.info(f"Downloading stock prices from {start_date} to {end_date}")

    logger.info("Fetching DJIA constituents")
    tickers = get_djia_constituents()
    if not tickers:
        logger.error("Failed to retrieve DJIA constituents")
        return
    logger.info(f"Found {len(tickers)} companies in the DJIA index")

    all_prices = pd.DataFrame()
    for i, ticker in enumerate(tickers, 1):
        logger.info(f"Downloading price data for {ticker} ({i}/{len(tickers)})")
        prices = download_stock_prices_with_retry(ticker, start_date, end_date)
        if not prices.empty:
            all_prices = pd.concat([all_prices, prices])
        time.sleep(random.uniform(12, 15))  # Alpha Vantage giới hạn 5 yêu cầu/phút

    if not all_prices.empty:
        logger.info("Saving stock prices to PostgreSQL")
        save_to_postgres(all_prices)
        logger.info(f"Processed {len(tickers)} companies, {all_prices['Ticker'].nunique()} with data, {len(all_prices)} records")
    else:
        logger.warning("No stock price data to save")

if __name__ == "__main__":
    main()