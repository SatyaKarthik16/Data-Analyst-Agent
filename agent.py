"""
Agent orchestration using LangGraph.

Manages the workflow with state memory and agentic execution.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Optional, Dict, Any
import pandas as pd
from data_loader import load_dataset
from analyzer import analyze_dataframe
from llm_suggester import get_cleaning_suggestions
from code_generator import generate_python_code_from_final_instructions
from executor import execute_cleaning_code
from data_analysis import DataAnalyzer
import logging

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """State for the data analyst agent."""
    df: Optional[pd.DataFrame]
    summary: Optional[str]
    cleaning_suggestions: Optional[str]
    final_cleaning_instructions: Optional[str]
    cleaning_code: Optional[str]
    cleaned_df: Optional[pd.DataFrame]
    analysis_results: Optional[Dict[str, Any]]
    insights: Optional[str]
    visualizations: Optional[list]
    user_feedback: Optional[str]
    task: str  # 'extract', 'clean', 'analyze'

class DataAnalystAgent:
    """Main agent orchestrating data analysis workflow."""

    def __init__(self):
        self.graph = self._build_graph()
        self.memory = MemorySaver()
        self.app = self.graph.compile(checkpointer=self.memory)

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node("extract_data", self.extract_data_node)
        graph.add_node("analyze_data", self.analyze_data_node)
        graph.add_node("suggest_cleaning", self.suggest_cleaning_node)
        graph.add_node("get_user_feedback", self.get_user_feedback_node)
        graph.add_node("generate_cleaning_code", self.generate_cleaning_code_node)
        graph.add_node("execute_cleaning", self.execute_cleaning_node)
        graph.add_node("perform_analysis", self.perform_analysis_node)
        graph.add_node("generate_insights", self.generate_insights_node)
        graph.add_node("create_visualizations", self.create_visualizations_node)

        # Define edges
        graph.set_entry_point("extract_data")
        graph.add_edge("extract_data", "analyze_data")
        graph.add_edge("analyze_data", "suggest_cleaning")
        graph.add_edge("suggest_cleaning", "get_user_feedback")
        graph.add_edge("get_user_feedback", "generate_cleaning_code")
        graph.add_edge("generate_cleaning_code", "execute_cleaning")
        graph.add_edge("execute_cleaning", "perform_analysis")
        graph.add_edge("perform_analysis", "generate_insights")
        graph.add_edge("generate_insights", "create_visualizations")
        graph.add_edge("create_visualizations", END)

        return graph

    def extract_data_node(self, state: AgentState) -> AgentState:
        """Extract data based on task."""
        logger.info("Extracting data")
        # This would be customized based on input
        # For now, assume file path is provided
        if 'file_path' in state:
            state['df'] = load_dataset(state['file_path'])
        return state

    def analyze_data_node(self, state: AgentState) -> AgentState:
        """Analyze the dataset."""
        logger.info("Analyzing data")
        state['summary'] = analyze_dataframe(state['df'])
        return state

    def suggest_cleaning_node(self, state: AgentState) -> AgentState:
        """Generate cleaning suggestions."""
        logger.info("Generating cleaning suggestions")
        state['cleaning_suggestions'] = get_cleaning_suggestions(state['summary'])
        return state

    def get_user_feedback_node(self, state: AgentState) -> AgentState:
        """Get user feedback on suggestions."""
        logger.info("Getting user feedback")
        # In interactive mode, this would prompt user
        # For now, assume acceptance
        state['final_cleaning_instructions'] = state['cleaning_suggestions']
        if state.get('user_feedback'):
            state['final_cleaning_instructions'] += "\n\nUser feedback: " + state['user_feedback']
        return state

    def generate_cleaning_code_node(self, state: AgentState) -> AgentState:
        """Generate cleaning code."""
        logger.info("Generating cleaning code")
        state['cleaning_code'] = generate_python_code_from_final_instructions(state['final_cleaning_instructions'])
        return state

    def execute_cleaning_node(self, state: AgentState) -> AgentState:
        """Execute cleaning code."""
        logger.info("Executing cleaning code")
        state['cleaned_df'] = execute_cleaning_code(state['df'], state['cleaning_code'])
        return state

    def perform_analysis_node(self, state: AgentState) -> AgentState:
        """Perform data analysis."""
        logger.info("Performing data analysis")
        analyzer = DataAnalyzer(state['cleaned_df'])
        state['analysis_results'] = {
            'basic_stats': analyzer.get_basic_stats(),
            'correlations': analyzer.correlation_analysis().to_dict(),
            'outliers': analyzer.outlier_detection()
        }
        return state

    def generate_insights_node(self, state: AgentState) -> AgentState:
        """Generate AI insights."""
        logger.info("Generating insights")
        analyzer = DataAnalyzer(state['cleaned_df'])
        state['insights'] = analyzer.generate_insights(state['analysis_results'])
        return state

    def create_visualizations_node(self, state: AgentState) -> AgentState:
        """Create visualizations."""
        logger.info("Creating visualizations")
        analyzer = DataAnalyzer(state['cleaned_df'])
        state['visualizations'] = analyzer.create_visualizations()
        return state

    def run_workflow(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the complete workflow."""
        config = {"configurable": {"thread_id": "data_analysis_thread"}}
        result = self.app.invoke(initial_state, config=config)
        return result