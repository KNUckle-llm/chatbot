from typing import Literal
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.agent.state import CustomState
from src.agent.utils import initialize_components, detect_language
from src.agent.prompts import GRADE_PROMPT, HITL_PROMPT, SYSTEM_PROMPT
from ..core.logger import get_logger

logger = get_logger(__name__)
model, store, retriever_tool_structured, RetrieverToolNode = initialize_components()


def language_detection_node(state: CustomState):
    logger.info("🔹 [language_detection_node] 시작")
    user_message = state.get("messages")[-1]
    state.set("language", detect_language(str(user_message.content)))
    state.set("next_node", "route_before_retrieval")
    return state

def route_before_retrieval_node(state: CustomState):
    logger.info("🔹 [route_before_retrieval_node] 시작")
    messages = state.get("messages", [])
    user_message = next((msg for msg in reversed(messages) if getattr(msg, "role", None) == "user"), None)
    question_text = str(user_message.content).strip() if user_message else ""

    if not question_text:
        state.set("unclear_reason", "질문이 비어있습니다. 명확하게 질문해 주세요.")
        state.set("next_node", "rewrite_question")
        return state
    
    # LLM에게 질문 평가
    eval_prompt = f"""
    사용자가 보낸 질문을 평가하여 반드시 'yes' 또는 'no'로 판단하세요. 
    - 'yes'는 질문이 충분히 구체적이거나 일부 정보가 부족해도 답변 가능할 때.  
    - 'no'는 질문이 챗봇 사용자, 챗봇 제공 정보와 전혀 관련이 없거나 불명확하고 질문이 너무 짧을 때.
         이 경우, 들어온 질문인 "{question_text}"에 기반하여 아래 정보를 근거로 역질문을 생성합니다.
    
    챗봇의 사용자 : 공주대학교 SW 사업단 주관학과 학생들(컴퓨터공학과, 소프트웨어학과, 인공지능학부, 스마트정보기술공학과)
    챗봇이 제공하는 정보 : 학과 정보(학과별 교과과정표, 학과별 교수님 정보, 학과별 공지사항, 학과별 자료/서식, SW사업단 소식, SW사업단 혜택, SW사업단 공지사항, SW사업단 대회정보)
    """
    try:
        eval_response = model.invoke([SystemMessage(content=eval_prompt)])
        logger.info(f"LLM 응답: {eval_response.content}")
        state.set("messages", messages + [AIMessage(content=eval_response.content)])
        content_lower = str(eval_response.content).lower()
        unclear = "no" in content_lower or "아니요" in content_lower
    except Exception as e:
        logger.error(f"LLM 평가 중 예외 발생: {e}")
        state.set("next_node", "rewrite_question")
        return state

    state.set("next_node", "rewrite_question" if unclear else "retrieve")
    return state


def retrieve_node(state: CustomState):
    logger.info("🔹 [retrieve_node] 실행 시작")
    node = RetrieverToolNode(retriever_tool_structured)
    state = node.run(state)

    # tool 메시지 여부에 따라 다음 노드 결정
    tool_msgs = [msg for msg in state.get("messages", []) if getattr(msg, "role", None) == "tool"]
    state.set("next_node", "collect_documents" if tool_msgs else "rewrite_question")
    return state
  

def collect_documents_node(state: CustomState):
    logger.info("🔹 [collect_documents_node] 시작")
    messages = state.get("messages", [])
    tool_msgs = [msg for msg in messages if getattr(msg, "role", None) == "tool"]

    if not tool_msgs:
        state.set("documents", [])
        state.set("no_docs", True)
        state.set("next_node", "rewrite_question")
        return state

    collected = []
    for msg in tool_msgs:
        try:
            docs = msg.content
            if isinstance(docs, list):
                collected.extend(docs)
        except Exception as e:
            logger.error(f"Failed to parse tool output: {e}")

    state.set("documents", collected[:3])
    state.set("no_docs", False)
    state.set("next_node", "generate")
    logger.info(f"Collected {len(collected)} documents. Next node: generate")
    return state



def rewrite_question_node(state: CustomState):
    logger.info("🔹 [rewrite_question_node] HITL 시작")
    unclear_info = state.get("unclear_reason", "질문을 명확하게 해주세요.")
    response_message = AIMessage(content=f"{unclear_info}")
    state.set("messages", state.get("messages") + [response_message])

    # HITL 후 summarize로 이동
    state.set("next_node", "summarize")
    return state


def generation_node(state: CustomState):
    logger.info("🔹 [generation_node] 시작")
    messages = state.get("messages", [])
    user_message_obj = next((msg for msg in reversed(messages) if getattr(msg, "role", None) == "user"), None)

    if not user_message_obj:
        logger.warning("마지막 유저 메시지가 없습니다.")
        state.set("next_node", "summarize")
        return state

    user_message = user_message_obj.content
    language = state.get("language")
    documents = state.get("documents", [])

    formatted_docs = ""
    for idx, doc in enumerate(documents, start=1):
        content = doc.get("content", "")
        url = doc.get("metadata", {}).get("url", "URL 없음")
        formatted_docs += f"문서 {idx}:\n  내용: {content}\n  출처: {url}\n\n"

    system_message = SYSTEM_PROMPT.format(documents=formatted_docs, input=user_message)
    final_message = f"Answer in {language}.\n\n" + system_message
    response = model.invoke([SystemMessage(content=final_message)])

    state.set("messages", messages + [AIMessage(content=response.content)])
    state.set("next_node", "summarize")
    return state



def summarization_node(state: CustomState):
    logger.info("🔹 [summarization_node] 시작")
    messages = state.get("messages")
    summarization = state.get("summarization") or ""

    summary_message = f"This is a summary of the conversation to date:\n{summarization}\nExtend considering new messages above:" \
        if summarization else "Create a summary of the conversation above:"

    response = model.invoke(messages + [HumanMessage(content=summary_message)])
    state.set("summarization", str(response.content).strip())
    state.set("messages", messages[-8:])  # 최신 8개만 유지

    # summarize 후 종료
    state.set("next_node", None)
    return state


