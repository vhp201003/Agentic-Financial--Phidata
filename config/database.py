from sqlalchemy import create_engine
from config.env import DATABASE_URL
from utils.logging import setup_logging
from utils.validators import validate_database_url

logger = setup_logging()

def get_engine():
    try:
        validate_database_url(DATABASE_URL)
        engine = create_engine(DATABASE_URL)
        logger.info("Database engine created successfully")
        return engine
    except Exception as e:
        logger.error(f"Failed to create database engine: {str(e)}")
        raise