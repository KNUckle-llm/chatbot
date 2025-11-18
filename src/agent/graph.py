from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from ..core.logger import get_logger
from .state import CustomState
from .utils import initialize_components
from .nodes import (
    generate_query_or_response_node,
    language_detection_node,
    route_before_retrieval_node,
    collect_documents_node,
    rewrite_question_node,
    generation_node,
    summarization_node,
)

logger = get_logger(__name__)


def build_graph(checkpointer, store=None) -> CompiledStateGraph:
    builder = StateGraph(CustomState)
    _, _, retriever_tool = initialize_components()

    logger.info("Generating Nodes...")
    builder.add_node("detect_language", language_detection_node)
    builder.add_node("generate_query_or_respond", generate_query_or_response_node)
    builder.add_node("retrieve", ToolNode([retriever_tool]))
    builder.add_node("collect_documents", collect_documents_node)
    builder.add_node("rewrite_question", rewrite_question_node)
    builder.add_node("generate", generation_node)
    builder.add_node("summarize", summarization_node)
    logger.info("Nodes generated.")

    logger.info("Adding Edges...")
    builder.add_edge(START, "detect_language")
    builder.add_edge("detect_language", "generate_query_or_respond")
    builder.add_conditional_edges(
        "generate_query_or_respond",
        route_before_retrieval_node,
        {
            "retrieve": "retrieve",
            "rewrite_question": "rewrite_question"
        }
    )
    builder.add_edge("retrieve", "collect_documents")
    builder.add_edge("collect_documents", "generate")
    builder.add_edge("generate", "summarize")
    builder.add_edge("rewrite_question", END)
    builder.add_edge("summarize", END)
    logger.info("Edges added.")

    graph = builder.compile(checkpointer=checkpointer, store=store)
    logger.info("Graph compiled successfully.")
    return graph


def visualize_graph(graph: CompiledStateGraph):
    from IPython.display import Image, display
    display(Image(graph.get_graph().draw_mermaid_png()))
