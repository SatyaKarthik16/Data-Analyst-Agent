"""
Code generation module for creating Pandas cleaning scripts.

Uses LLM to generate executable Python code for data cleaning operations.
"""

import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from config import OPENAI_API_KEY, MODEL_NAME, logger

def generate_python_code_from_final_instructions(instructions: str) -> str:
    """
    Generate Python cleaning code based on instructions.

    Args:
        instructions: Combined cleaning instructions

    Returns:
        str: Generated Python code
    """
    logger.info("Generating Python cleaning code using LLM")

    system_prompt = (
        "You are a helpful data cleaning assistant. "
        "You were previously asked to suggest cleaning steps based on a dataset summary. "
        "Now, based on your earlier suggestions AND the user's additional instructions, "
        "generate a complete Python cleaning script using Pandas for a DataFrame named 'df'.\n\n"
        "Include comments to explain each step\n"
        "Do not include markdown formatting or quotes\n"
        "Only output raw Python code"
    )

    try:
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=MODEL_NAME)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Please generate a complete Python script for the following cleaning plan:\n\n{instructions}")
        ]

        response = llm.invoke(messages)
        code = response.content.strip()

        logger.info("Python cleaning code generated successfully")
        return code

    except Exception as e:
        logger.error(f"Error generating cleaning code: {e}")
        return "# Error: Unable to generate code. Please check your OpenAI API key and try again."
