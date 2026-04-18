"""
LLM-powered cleaning suggestions module.

Uses OpenAI GPT to generate intelligent data cleaning recommendations.
"""

import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from config import OPENAI_API_KEY, MODEL_NAME, logger

def get_cleaning_suggestions(summary: str) -> str:
    """
    Generate AI-powered cleaning suggestions based on dataset summary.

    Args:
        summary: Dataset analysis summary

    Returns:
        str: Formatted cleaning suggestions
    """
    logger.info("Generating cleaning suggestions using LLM")

    system_prompt = (
        "You are a helpful data cleaning assistant. "
        "Given the summary of a dataset, analyze and suggest the most important cleaning steps. "
        "Be concise and clear. Mention the specific column names and actions needed. "
        "Use a numbered list. Keep it understandable for non-technical users."
    )

    try:
        # Initialize OpenAI chat model
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=MODEL_NAME)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Here is the dataset summary:\n\n{summary}")
        ]

        response = llm.invoke(messages)
        suggestions = response.content.strip()

        logger.info("Cleaning suggestions generated successfully")
        return suggestions

    except Exception as e:
        logger.error(f"Error generating cleaning suggestions: {e}")
        return "Error: Unable to generate suggestions. Please check your OpenAI API key and try again."
