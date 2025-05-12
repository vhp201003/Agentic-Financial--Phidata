import os
import sys
from pathlib import Path

# Thêm thư mục gốc dự án vào sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from apscheduler.schedulers.blocking import BlockingScheduler
from scripts.download_djia_companies import main as download_companies
from scripts.download_djia_stock_prices import main as download_prices
from utils.logging import setup_logging

logger = setup_logging()

def run_scheduled_jobs():
    logger.info("Running scheduled data download")
    try:
        download_companies()
        download_prices()
        logger.info("Scheduled data download completed")
    except Exception as e:
        logger.error(f"Scheduled job failed: {str(e)}")

if __name__ == "__main__":
    scheduler = BlockingScheduler()
    scheduler.add_job(run_scheduled_jobs, 'interval', days=1, start_date='2025-05-12 00:00:00')
    logger.info("Starting scheduler")
    scheduler.start()