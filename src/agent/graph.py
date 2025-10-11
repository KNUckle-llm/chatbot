from langgraph.checkpoint.memory import InMemorySaver  # short-term memory
from langgraph.store.memory import InMemoryStore  # long-term memory
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from ..core.logger import get_logger
from .state import CustomState
from .nodes import (
    answer_node,
    detect_language_node,
)

logger = get_logger(__name__)


def build_graph():
    checkpointer = InMemorySaver()  # short-term memory
    # store = InMemoryStore()  # long-term memory
    builder = StateGraph(CustomState)

    logger.debug("그래프 빌드 시작")
    builder.add_node("detect_language", detect_language_node)
    builder.add_node("answer", answer_node)

    builder.add_edge(START, "detect_language")
    builder.add_edge("detect_language", "answer")
    builder.add_edge("answer", END)

    graph = builder.compile(checkpointer=checkpointer,)
    logger.info("그래프 빌드 완료")
    return graph


def visualize_graph(graph: CompiledStateGraph):
    from IPython.display import Image, display
    display(Image(graph.get_graph().draw_mermaid_png()))
