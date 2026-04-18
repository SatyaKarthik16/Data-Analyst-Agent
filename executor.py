"""
Secure code execution module for data cleaning scripts.

Uses RestrictedPython to safely execute LLM-generated code.
"""

import pandas as pd
import logging
from typing import Optional
from RestrictedPython import safe_builtins, limited_builtins, utility_builtins
from RestrictedPython import compile_restricted_exec
from config import logger

def execute_cleaning_code(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """
    Execute cleaning code in a restricted environment.

    Args:
        df: Input DataFrame
        code: Python code to execute

    Returns:
        pd.DataFrame: Cleaned DataFrame or original if execution fails
    """
    logger.info("Executing cleaning code in restricted environment")

    # Prepare local variables
    local_vars = {'df': df.copy(), 'pd': pd}

    # Define restricted builtins for safe execution
    restricted_globals = {
        '__builtins__': {
            **safe_builtins,
            **limited_builtins,
            **utility_builtins,
            # Add essential builtins for pandas operations
            'pd': pd,
            'print': print,
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': list,
            'dict': dict,
            'set': set,
            'tuple': tuple,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'sorted': sorted,
            'reversed': reversed,
            'sum': sum,
            'min': min,
            'max': max,
            'abs': abs,
            'round': round,
            'isinstance': isinstance,
            'hasattr': hasattr,
            'getattr': getattr,
            'setattr': setattr,
        }
    }

    try:
        # Compile and execute in restricted environment
        compiled_code = compile_restricted_exec(code, '<string>', restricted_globals)
        if compiled_code.errors:
            logger.error(f"Compilation errors: {compiled_code.errors}")
            return df
        exec(compiled_code.code, restricted_globals, local_vars)

        cleaned_df = local_vars.get('df', df)

        # Validate the result
        if not isinstance(cleaned_df, pd.DataFrame):
            logger.warning("Code execution did not return a DataFrame, using original")
            return df

        logger.info("Code executed successfully")
        return cleaned_df

    except Exception as e:
        logger.error(f"Error executing cleaning code: {e}")
        logger.info("Returning original DataFrame as fallback")
        return df  # fallback to original if failed
