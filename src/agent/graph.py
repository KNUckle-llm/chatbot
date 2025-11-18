from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from ..core.logger import get_logger
from .state import CustomState
from .utils import initialize_components
from .nodes import (
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
    model, store, retriever_tool_structured, RetrieverToolNode = initialize_components()

    # -----------------------
    # 노드 등록
    # -----------------------
    builder.add_node("detect_language", language_detection_node)
    builder.add_node("route_before_retrieval", route_before_retrieval_node)

    # retrieve ToolNode
    def retrieve_node(state):
        node = RetrieverToolNode(retriever_tool_structured)
        return node.run(state)
    builder.add_node("retrieve", retrieve_node)

    builder.add_node("collect_documents", collect_documents_node)
    builder.add_node("rewrite_question", rewrite_question_node)
    builder.add_node("generate", generation_node)
    builder.add_node("summarize", summarization_node)

    # -----------------------
    # 노드 연결 (Edges)
    # -----------------------
    builder.add_edge(START, "detect_language")
    builder.add_edge("detect_language", "route_before_retrieval")

    # route_before_retrieval 분기
    builder.add_conditional_edges(
        "route_before_retrieval",
        route_before_retrieval_node,
        {True: "rewrite_question", False: "retrieve"}
    )

    # retrieve → collect_documents
    builder.add_edge("retrieve", "collect_documents")

    # collect_documents 분기
    builder.add_conditional_edges(
        "collect_documents",
        lambda state: bool(state.get("documents")),
        {True: "generate", False: "rewrite_question"}
    )

    # generate → summarize
    builder.add_edge("generate", "summarize")

    # rewrite_question → summarize
    builder.add_edge("rewrite_question", "summarize")

    # 종료
    builder.add_edge("summarize", END)

    graph = builder.compile(checkpointer=checkpointer, store=store)
    logger.info("StateGraph compiled successfully!")
    return graph


def visualize_graph(graph: CompiledStateGraph):
    from IPython.display import Image, display
    display(Image(graph.get_graph().draw_mermaid_png()))
