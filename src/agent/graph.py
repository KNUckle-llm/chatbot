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
    # initialize_components에서 ToolNode 인스턴스 직접 받음
    model, store, retriever_tool_structured, retriever_node_instance = initialize_components()

    # -----------------------
    # 노드 등록
    # -----------------------
    logger.info("Generating Node...")
    builder.add_node("detect_language", language_detection_node)
    builder.add_node("generate_query_or_respond", generate_query_or_response_node)
    
    # ※ wrapper를 통해 등록해야 LangGraph가 invoke 호출함
    def retrieve_node(state):
        return retriever_node_instance.invoke(state)
    
    builder.add_node("retrieve", retrieve_node)

    builder.add_node("collect_documents", collect_documents_node)
    builder.add_node("rewrite_question", rewrite_question_node)
    builder.add_node("generate", generation_node)
    builder.add_node("summarize", summarization_node)
    logger.info("Node generation complete..!")

    logger.info("Adding Edges...")
    builder.add_edge(START, "detect_language")
    builder.add_edge("detect_language", "generate_query_or_respond")
    builder.add_conditional_edges(
        "generate_query_or_respond",
        route_before_retrieval_node,
        {
            "retrieve": "retrieve",
            "rewrite_question": "rewrite_question",
        }
    )
    builder.add_edge("retrieve", "collect_documents")
    builder.add_edge("collect_documents", "generate")
    builder.add_edge("generate", "summarize")
    builder.add_edge("rewrite_question", END)
    builder.add_edge("summarize", END)
    logger.info("Edges added successfully..!")

    graph = builder.compile(checkpointer=checkpointer, store=store)
    logger.info("Successfully compiled the state graph :D")
    return graph


def visualize_graph(graph: CompiledStateGraph):
    from IPython.display import Image, display
    display(Image(graph.get_graph().draw_mermaid_png()))
