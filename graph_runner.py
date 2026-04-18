# graph_runner.py

from agent_graph import *

def run_agent():
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("welcome", welcome_node)
    graph.add_node("load_dataset", load_dataset_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("suggest", suggest_node)
    graph.add_node("feedback", human_feedback_node)
    graph.add_node("generate_code", generate_code_node)
    graph.add_node("execute", execute_code_node)
    graph.add_node("save_output", save_output_node)

    # Define flow
    graph.set_entry_point("welcome")
    graph.add_edge("welcome", "load_dataset")
    graph.add_edge("load_dataset", "analyze")
    graph.add_edge("analyze", "suggest")
    graph.add_edge("suggest", "feedback")
    graph.add_edge("feedback", "generate_code")
    graph.add_edge("generate_code", "execute")
    graph.add_edge("execute", "save_output")
    graph.add_edge("save_output", END)

    app = graph.compile()
    app.invoke({})  # Start with empty state


if __name__ == "__main__":
    run_agent()
