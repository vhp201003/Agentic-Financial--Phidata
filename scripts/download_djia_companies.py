import os
import sys
from pathlib import Path

# Thêm thư mục gốc dự án vào sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import requests
import time
import random
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

def get_company_info_with_retry(ticker, max_retries=5, initial_delay=10):
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
    retry_count = 0
    delay = initial_delay
    while retry_count < max_retries:
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if not data or "Note" in data:
                logger.warning(f"Rate limit or no data for {ticker}. Response: {data.get('Note', 'No data')}")
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = delay + random.uniform(1, 3)
                    logger.warning(f"Waiting {wait_time:.2f} seconds before retry {retry_count}/{max_retries} for {ticker}")
                    time.sleep(wait_time)
                    delay *= 2
                else:
                    logger.error(f"Max retries reached for {ticker}")
                    return None
            else:
                return {
                    "symbol": ticker,
                    "name": data.get("Name", ""),
                    "sector": data.get("Sector", ""),
                    "industry": data.get("Industry", ""),
                    "country": data.get("Country", ""),
                    "website": data.get("Website", ""),
                    "market_cap": int(data.get("MarketCapitalization", 0)),
                    "pe_ratio": float(data.get("PERatio", 0)),
                    "dividend_yield": float(data.get("DividendYield", 0)) * 100 if data.get("DividendYield") else 0,
                    "week_high_52": float(data.get("52WeekHigh", 0)),
                    "week_low_52": float(data.get("52WeekLow", 0)),
                    "description": data.get("Description", "")
                }
        except requests.exceptions.RequestException as e:
            retry_count += 1
            if retry_count < max_retries:
                wait_time = delay + random.uniform(1, 3)
                logger.warning(f"Error for {ticker} (attempt {retry_count}/{max_retries}): {str(e)}. Waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
                delay *= 2
            else:
                logger.error(f"Max retries reached for {ticker}: {str(e)}")
                return None
        except ValueError as e:
            logger.error(f"Invalid response for {ticker}: {str(e)}")
            return None
    return None

def save_to_postgres(companies_info):
    try:
        validate_database_url(DATABASE_URL)
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            for info in companies_info:
                if info is None:
                    continue
                conn.execute(
                    text("""
                        INSERT INTO companies (
                            symbol, name, sector, industry, country, website,
                            market_cap, pe_ratio, dividend_yield, week_high_52, week_low_52, description
                        )
                        VALUES (:symbol, :name, :sector, :industry, :country, :website,
                                :market_cap, :pe_ratio, :dividend_yield, :week_high_52, :week_low_52, :description)
                        ON CONFLICT (symbol) DO UPDATE SET
                            name = EXCLUDED.name,
                            sector = EXCLUDED.sector,
                            industry = EXCLUDED.industry,
                            country = EXCLUDED.country,
                            website = EXCLUDED.website,
                            market_cap = EXCLUDED.market_cap,
                            pe_ratio = EXCLUDED.pe_ratio,
                            dividend_yield = EXCLUDED.dividend_yield,
                            week_high_52 = EXCLUDED.week_high_52,
                            week_low_52 = EXCLUDED.week_low_52,
                            description = EXCLUDED.description
                    """),
                    info
                )
            conn.commit()
        logger.info("Company data saved to PostgreSQL successfully")
    except Exception as e:
        logger.error(f"Failed to save company data to PostgreSQL: {str(e)}")
        raise

def main():
    logger.info("Fetching DJIA constituents")
    tickers = get_djia_constituents()
    if not tickers:
        logger.error("Failed to retrieve DJIA constituents")
        return
    logger.info(f"Found {len(tickers)} companies in the DJIA index")

    companies_info = []
    for i, ticker in enumerate(tickers, 1):
        logger.info(f"Downloading information for {ticker} ({i}/{len(tickers)})")
        company_info = get_company_info_with_retry(ticker)
        if company_info:
            companies_info.append(company_info)
        time.sleep(random.uniform(12, 15))  # Alpha Vantage giới hạn 5 yêu cầu/phút

    if companies_info:
        logger.info("Saving company information to PostgreSQL")
        save_to_postgres(companies_info)
    else:
        logger.warning("No company data to save")

if __name__ == "__main__":
    main()