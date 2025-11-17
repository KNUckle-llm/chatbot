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
    _, _, retriever_tool = initialize_components()

    logger.info("Generating Node...")
    
    # 노드 등록
    builder.add_node("detect_language", language_detection_node)
    builder.add_node("retrieve", ToolNode([retriever_tool]))
    builder.add_node("collect_documents", collect_documents_node)
    builder.add_node("rewrite_question", rewrite_question_node)
    builder.add_node("generate", generation_node)
    builder.add_node("summarize", summarization_node)
    logger.info("Node generation complete..!")

    logger.info("Adding Edges...")
    builder.add_edge(START, "detect_language")
    
    # 언어 감지 → 분기 판단
    builder.add_conditional_edges(
        "detect_language",
        route_before_retrieval_node,
        {
            "retrieve": "retrieve",
            "rewrite_question": "rewrite_question"
        }
    )
    
    # retrieve 경로
    # ("retrieve", ToolNode([retriever_tool]))하면 "retrieve" 노드는 ToolNode를 실행한 결과를 상태(state)에 추가하게 된다.
    builder.add_edge("retrieve", "collect_documents")
    builder.add_edge("collect_documents", "generate")
    builder.add_edge("generate", "summarize")
    builder.add_edge("summarize", END)
    
    # HITL 경로
    builder.add_edge("rewrite_question", END)
    
    logger.info("Edges added successfully..!")

    graph = builder.compile(checkpointer=checkpointer, store=store)
    logger.info("Successfully compiled the state graph :D")
    return graph


def visualize_graph(graph: CompiledStateGraph):
    from IPython.display import Image, display
    display(Image(graph.get_graph().draw_mermaid_png()))
