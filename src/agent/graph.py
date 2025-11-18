from typing import Optional
from langgraph.graph.state import CompiledStateGraph
from src.agent.state import CustomState
from src.agent.utils import initialize_components
from src.agent.nodes import (
    language_detection_node,
    route_before_retrieval_node,
    collect_documents_node,
    rewrite_question_node,
    generation_node,
    summarization_node,
)
from ..core.logger import get_logger

logger = get_logger(__name__)
model, store, retriever_tool_structured, RetrieverToolNode = initialize_components()


# -------------------------------
# 노드 매핑
# -------------------------------
NODE_MAP = {
    "detect_language": language_detection_node,
    "route_before_retrieval": route_before_retrieval_node,
    "retrieve": lambda state: RetrieverToolNode(retriever_tool_structured).run(state),
    "collect_documents": collect_documents_node,
    "rewrite_question": rewrite_question_node,
    "generate": generation_node,
    "summarize": summarization_node,
}


# -------------------------------
# next_node 기반 실행
# -------------------------------
def run_state_machine(state: CustomState, start_node: str = "detect_language"):
    state.next_node = start_node

    while state.next_node:
        node_name = state.next_node
        logger.info(f"▶ 실행 노드: {node_name}")
        state.next_node = None  # 실행 전 초기화

        node_func = NODE_MAP.get(node_name)
        if not node_func:
            logger.error(f"Node {node_name} 미등록")
            break

        state = node_func(state)

        # 종료 조건
        if node_name == "summarize":
            logger.info("✅ 최종 노드 summarize 실행 완료")
            break

    return state