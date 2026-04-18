"""
Data loading module for the Data Analyst application.

Supports loading data from various file formats with validation and error handling.
"""

import pandas as pd
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = ['.csv', '.xls', '.xlsx', '.json']

def is_supported(file_path: str) -> bool:
    """Check if the file extension is supported."""
    return any(file_path.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS)

def load_dataset(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load dataset from file with validation and error handling.

    Args:
        file_path: Path to the data file
        **kwargs: Additional arguments for pandas read functions

    Returns:
        pandas.DataFrame: Loaded dataset

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file type is unsupported
        Exception: For other loading errors
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    if not is_supported(file_path):
        logger.error(f"Unsupported file type: {file_path}")
        raise ValueError(f"Unsupported file type. Supported: {SUPPORTED_EXTENSIONS}")

    try:
        ext = file_path.lower().split('.')[-1]

        if ext == "csv":
            df = pd.read_csv(file_path, **kwargs)
        elif ext in ["xls", "xlsx"]:
            df = pd.read_excel(file_path, **kwargs)
        elif ext == "json":
            df = pd.read_json(file_path, **kwargs)

        logger.info(f"Successfully loaded dataset from {file_path}: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        raise
