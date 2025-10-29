from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from ..core.logger import get_logger
from .state import CustomState
from .nodes import (
    language_detection_node,
    retrieval_node,
    generation_node,
    summarization_node,
)

logger = get_logger(__name__)


def build_graph(checkpointer, store=None) -> CompiledStateGraph:
    builder = StateGraph(CustomState)

    logger.info("Generating Node...")
    builder.add_node("detect_language", language_detection_node)
    builder.add_node("retrieve", retrieval_node)
    builder.add_node("generate", generation_node)
    builder.add_node("summarize", summarization_node)
    logger.info("Node generation complete..!")

    logger.info("Adding Edges...")
    builder.add_edge(START, "detect_language")
    builder.add_edge("detect_language", "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "summarize")
    builder.add_edge("summarize", END)
    logger.info("Edges added successfully..!")

    graph = builder.compile(checkpointer=checkpointer, store=store)
    logger.info("Successfully compiled the state graph :D")
    return graph


def visualize_graph(graph: CompiledStateGraph):
    from IPython.display import Image, display
    display(Image(graph.get_graph().draw_mermaid_png()))
