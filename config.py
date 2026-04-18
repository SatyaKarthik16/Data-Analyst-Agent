"""
Configuration module for the Data Analyst application.

Handles environment variables, logging setup, and application settings.
"""

import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys with validation
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set in .env file")

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")

# Application Settings
MAX_DATA_SIZE = int(os.getenv("MAX_DATA_SIZE", "100000"))  # Max rows for processing
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", "app.log")

# Logging Setup
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Configuration loaded successfully")