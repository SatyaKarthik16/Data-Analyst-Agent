"""
Data analysis module for comprehensive dataset analysis.

Provides detailed statistical analysis and summary generation for DataFrames.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

def analyze_dataframe(df: pd.DataFrame, id_columns: Optional[List[str]] = None) -> str:
    """
    Generate comprehensive analysis summary of a DataFrame.

    Args:
        df: DataFrame to analyze
        id_columns: List of columns to check for duplicates (ID columns)

    Returns:
        str: Formatted analysis summary
    """
    if id_columns is None:
        id_columns = []

    logger.info(f"Starting analysis of DataFrame with shape {df.shape}")
    buffer = []

    # Dataset Shape
    buffer.append(f"Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

    # Data Types and Schema
    buffer.append("Data Types & Schema:")
    for col in df.columns:
        buffer.append(f"- {col}: {df[col].dtype}")
    buffer.append("")

    # Column Names
    buffer.append("Column Names:")
    buffer.append(", ".join(df.columns))
    buffer.append("")

    # Missing Values
    buffer.append("Missing Values (count and %):")
    total_rows = df.shape[0]
    for col in df.columns:
        missing = df[col].isnull().sum()
        if missing > 0:
            pct = round(100 * missing / total_rows, 2)
            buffer.append(f"- {col}: {missing} missing ({pct}%)")
        else:
            buffer.append(f"- {col}: 0 missing (0%)")
    buffer.append("")

    # Duplicate Records
    dup_rows = df.duplicated().sum()
    buffer.append(f"Duplicate Records: {dup_rows} duplicate rows based on all columns.")
    for col in id_columns:
        if col in df.columns:
            dup_ids = df[col].duplicated().sum()
            buffer.append(f"- {dup_ids} duplicate entries in `{col}` (ID check)")
    buffer.append("")

    # Summary Statistics (numerical only)
    buffer.append("Summary Statistics (Numerical Columns):")
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] > 0:
        desc = numeric_df.describe(percentiles=[.25, .5, .75]).T
        for col, stats in desc.iterrows():
            buffer.append(f"- {col}: mean = {stats['mean']:.2f}, std = {stats['std']:.2f}, "
                          f"min = {stats['min']:.2f}, max = {stats['max']:.2f}")
    else:
        buffer.append("No numerical columns found.")
    buffer.append("")

    # Unique Values / Cardinality
    buffer.append("Unique Values (Categorical Columns):")
    cat_df = df.select_dtypes(include='object')
    if cat_df.shape[1] > 0:
        for col in cat_df.columns:
            unique_vals = df[col].nunique()
            buffer.append(f"- {col}: {unique_vals} unique values")
    else:
        buffer.append("No categorical columns found.")
    buffer.append("")

    # Sample Data Preview
    buffer.append("Sample Data Preview:")
    buffer.append("First 5 rows:")
    buffer.append(df.head().to_markdown(index=False))
    buffer.append("\nLast 5 rows:")
    buffer.append(df.tail().to_markdown(index=False))

    logger.info("DataFrame analysis completed")
    return "\n".join(buffer)
