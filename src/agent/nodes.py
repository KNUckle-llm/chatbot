from typing import Literal
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from src.agent.state import CustomState
from src.agent.utils import initialize_components, detect_language
from src.agent.prompts import HITL_PROMPT, SYSTEM_PROMPT
from ..core.logger import get_logger

logger = get_logger(__name__)
model, store, retriever_tool = initialize_components()


def language_detection_node(state: CustomState):
    logger.info(">>> [NODE] language_detection_node START")
    last_msg = state.get("messages")[-1]
    text = str(last_msg.content)
    state["language"] = detect_language(text)
    logger.info(f"Detected language: {state['language']}")
    return {"language": state["language"]}


def generate_query_or_response_node(state: CustomState):
    logger.info(">>> [NODE] generate_query_or_response_node START")
    last_msg = state.get("messages")[-1]
    prompt = (
        f"질문: {last_msg.content}\n"
        "아래 기준을 바탕으로 이 질문이 검색 가능한 문서로 답변 가능한지 판단하세요.\n\n"

        "### 판단 기준\n"
        "1) 검색 가능한 문서 범위 내에서 답변 가능한 질문이면 'yes'입니다.\n"
        "   답변 불가하거나 불확실하면 'no'입니다.\n\n"

        "2) 검색 가능한 문서 범위는 다음과 같습니다.\n"
        "   - 공주대학교 통합 수강신청/장학/비자/논문/순환버스\n"
        "   - 학과별 교수님(연락처, 이메일 등)/교과과정표/공지사항/자료/서식/규정\n"
        "   - SW사업단 소개/공지사항/소식/대회일정(TOPCIT, SW알고리즘 경진대회 등)\n"

        "3) 개인정보 포함 여부는 적절성 판단 기준이 아닙니다.\n"
        "   위 판단 기준으로 답이 가능한지 여부만 고려하세요.\n\n"

        "### 출력 형식\n"
        "- 첫 줄: 반드시 영어 'yes' 또는 'no'로 시작 (대소문자 혼용 금지)\n"
        "- 둘째 줄: 판단 이유 1~2문장\n"
        "  - yes일 때: 질문이 검색 가능한 문서 범위 내에서 답변 가능하다는 이유\n"
        "  - no일 때: 질문이 검색 가능한 문서 범위 내에서 답변할 수 없는 이유 (왜 부적절한지 간단하게 설명)\n"
    )
    response = model.invoke([SystemMessage(content=prompt)])
    raw_text = response.content.strip()
    
    # 질문 적절성 판단 및 이유 저장
    if raw_text.lower().startswith("no"):
        state["question_appropriate"] = False
        state["question_reason"] = raw_text[2:].strip()
    else:
        state["question_appropriate"] = True
        state["question_reason"] = None

    # 디버그 로그
    logger.info("===== Question Evaluation Debug =====")
    logger.info(f"User Question: {last_msg.content}")
    logger.info(f"Raw LLM Response: {raw_text}")
    logger.info(f"question_appropriate: {state['question_appropriate']}")
    if not state["question_appropriate"]:
        logger.info(f"question_reason: {state['question_reason']}")
    logger.info("===================================")

    #state.get("messages").append(response)
    return {
        "question_appropriate": state["question_appropriate"],
        "question_reason": state.get("question_reason", None)
    }



def route_before_retrieval_node(state: CustomState) -> Literal["retrieve", "rewrite_question"]:
    logger.info(">>> [NODE] route_before_retrieval_node START")
    # 안전하게 기본값 False 지정
    is_appropriate = state.get("question_appropriate")
    if is_appropriate is None:
        logger.warning("question_appropriate가 None입니다. 기본값 False로 처리합니다.")
        is_appropriate = False
        state["question_appropriate"] = is_appropriate

    logger.info(f"Routing decision - question_appropriate: {is_appropriate}")
    return "retrieve" if is_appropriate else "rewrite_question"



def retrieve_documents_node(state: CustomState, max_docs: int = 3):
    logger.info(">>> [NODE] retrieve_documents_node START")
    last_msg = state.get("messages")[-1]
    query = str(last_msg.content).strip()

    # 학과 후보 리스트
    departments = [
        "소프트웨어학과",
        "컴퓨터공학과",
        "공주대학교",
        "공주대학교 SW중심대학사업단",
        "SW중심대학사업단",
        "스마트정보기술공학과",
        "인공지능학부",
        "공주대학교 현장실습지원센터"
    ]
    
    # 2) alias 매핑 (여기서 OR 조건 처리)
    alias_map = {
        "공주대학교 SW중심대학사업단": [
            "공주대학교 SW중심대학사업단",
            "SW중심대학사업단",
        ],
        "SW중심대학사업단": [
            "공주대학교 SW중심대학사업단",
            "SW중심대학사업단",
        ],
    }
    
    # LLM에게 질문 관련 학과 예측
    dept_prompt = (
        f"사용자 질문: {query}\n"
        f"질문을 보고 아래 목록 중에서 관련 학과/부서를 하나 선택하세요:\n"
        f"반드시 목록 중 하나를 그대로 출력하세요.\n"
        f"목록: {', '.join(departments)}"
    )
    dept_response = model.invoke([SystemMessage(content=dept_prompt)])
    predicted_department = dept_response.content.strip()
    logger.info(f"Predicted department: {predicted_department}")

    # store에서 similarity_search로 검색 (필터 적용)
    if predicted_department in departments:
        # alias 지원 (OR 검색)
        aliases = alias_map.get(predicted_department, [predicted_department])
        filter_expr = {"department": {"$in": aliases}}
        logger.info(f"Using filter: {filter_expr}")

        docs = store.similarity_search(query, k=max_docs, filter=filter_expr)

    else:
        # 학과 판단 실패 시 필터 없이 검색
        logger.info("Predicted department not recognized. Running search without filter.")
        docs = store.similarity_search(query, k=max_docs)
    
    #중요!!!!!! : store에서 similarity_search로 바로 검색
    #docs = store.similarity_search(query, k=max_docs)

    state["documents"] = [
        {
            "content": d.page_content,
            "metadata": {
                "file_name": d.metadata.get("file_name"),
                "department": d.metadata.get("department"),
                "url": d.metadata.get("url"),
                "date": d.metadata.get("date")
            }
        }
        for d in docs
    ]

    logger.info(f"Retrieved {len(docs)} documents for query: {query}")
    return {"documents": state["documents"]}



def rewrite_question_node(state: CustomState):
    logger.info(">>> [NODE] rewrite_question_node START")
    if state.get("question_appropriate"):
        # 적절한 질문이면 그냥 메시지 반환
        return {"messages": state.get("messages")}

    last_msg = state.get("messages")[-1]
    reason = state.get("question_reason", "불명확한 이유 없음")
    previous_summary = state.get("summarization", "")
    
    prompt = (
        f"사용자가 한 질문: {last_msg.content}\n"
        f"불명확한 이유: {reason}\n\n"
        f"이전 대화 요약: {previous_summary}\n\n"
        "사용자에게 보여줄 안내 메시지를 작성하세요. 형식은 다음과 같아야 합니다:\n"
        "첫 문단입니다. '질문은 다음과 같은 이유로 불명확합니다. 질문을 다시 입력해주세요.'\n"
        "두 번째 문단입니다. 이전 대화 요약과 불명확한 이유를 참고하여, 질문이 검색되지 않은 이유를 명확하게 설명하세요.n"
        "세 번째 문단입니다. '이렇게 질문하는건 어떨까요?' 라는 문장으로 시작하고,\n"
        "   사용자의 질문과 불명확한 이유를 기반으로 더 구체적이고 적절한 질문 예시 1~2개를 bullet 형식으로 제시하세요."
    )    
    
    # 3) LLM 호출 후 메시지 추가
    response = model.invoke([SystemMessage(content=prompt)])
    state.get("messages").append(response)
    logger.info("Rewritten question/feedback added.")

    return {"messages": state.get("messages")}


def generation_node(state: CustomState):
    logger.info(">>> [NODE] generation_node START")
    language = state.get("language", "ko")
    documents = state.get("documents", [])
    summarization = state.get("summarization", "")
    last_msg = state.get("messages")[-1]

    # 문서 내용을 정리해서 문자열로 만듦
    #docs_text = "\n".join([f"문서 {i+1}:\n    내용: {d['content']}" for i, d in enumerate(documents)])
    docs_text = "\n\n".join([
        f"문서 {i+1}\n"
        f"본문 내용: {d['content']}\n"
        f"제목: {d.get('metadata', {}).get('file_name', '')}\n"
        f"부서: {d.get('metadata', {}).get('department', '')}\n"
        f"작성일: {d.get('metadata', {}).get('date', '')}"
        f"출처: {d.get('metadata', {}).get('url', '')}\n"
        for i, d in enumerate(documents)
    ])
    
    # 시스템 메시지 생성
    system_message = SYSTEM_PROMPT.format(
        input=last_msg.content,
        documents=docs_text,
        summary=summarization
    )

    # LLM 호출
    response = model.invoke([SystemMessage(content=system_message)])
    state.get("messages").append(response)
    
    return {"messages": state.get("messages")}



def summarization_node(state: CustomState):
    logger.info(">>> [NODE] summarization_node START")
    messages = state.get("messages")
    summary_prompt = "대화를 요약하세요:\n" + "\n".join([msg.content for msg in messages])
    response = model.invoke([SystemMessage(content=summary_prompt)])

    delete_msgs = [RemoveMessage(id=msg.id) for msg in messages[:-8]]
    state["summarization"] = str(response.content).strip()
    logger.info("Conversation summarized.")
    return {"summarization": state["summarization"], "messages": delete_msgs}