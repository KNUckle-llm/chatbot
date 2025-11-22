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
        "이 질문이 챗봇 사용자와 챗봇의 검색 가능한 문서를 기준으로 어느정도 관련성이 있는 경우 'yes', 불명확하면 'no'로 판단하세요."
        "반드시 맨 처음 시작으로 영어로 yes/no 중에 하나가 답변으로 있어야 합니다.\n"
        "교수님에 대한 질문이 들어오면 무조건 yes로 답하세요.\n"
        "'no'라면 불명확하면 이유를 1~2문장으로 구체적으로 작성하세요.\n"
        "챗봇의 사용자는 SW사업단 소속 학부생으로 컴퓨터공학과, 소프트웨어학과, 인공지능학부, 스마트정보기술공학과가 있다.\n"
        "검색 가능한 문서로는 학과별 교수님 공식정보(이메일, 전화번호), 학과별 교과과정표, 학과별 공지사항, 학과별 자료/서식, 학과별 규정이 있고 SW사업단 소식, SW사업단 공지사항, SW사업단 대회일정 등이 존재한다.\n"
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
    
    # 1) 질문 유형 판단
    qtype_prompt = (
        f"질문: {last_msg.content}\n"
        "다음 중 하나로 분류하세요: "
        "1. 이전 대화 관련, "
        "2. 일반 일상 대화, "
        "3. 문서/정보 검색용 질문\n"
        "출력은 숫자(1,2,3)만 사용하세요."
    )
    qtype_resp = model.invoke([SystemMessage(content=qtype_prompt)])
    qtype = qtype_resp.content.strip()
    
    # 2) 유형별 프롬프트 작성
    if qtype == "1":
        prompt = (
            f"사용자가 한 질문: {last_msg.content}\n"
            f"이전 대화 요약: {previous_summary}\n\n"
            f"이 질문은 이전 대화와 관련 있습니다.\n"
            "사용자 질문에 대한 답변으로 이전 대화 요약을 참고하세요."
        )
    elif qtype == "2":
        prompt = (
            f"사용자가 한 질문: {last_msg.content}\n"
            f"불명확한 이유: {reason}\n\n"
            "사용자에게 보여줄 안내 메시지를 작성하세요. 형식은 다음과 같아야 합니다:\n"
            "첫 문단입니다. '질문은 다음과 같은 이유로 불명확합니다. 질문을 다시 입력해주세요.'\n"
            "두 번째 문단입니다. 불명확한 이유를 서술하세요.\n"
            "세 번째 문단입니다. '이렇게 질문하는건 어떨까요?' 형식으로, "
            "   사용자가 입력한 질문을 기반으로 조금 더 구체적이고 적절하게 만든 1~2개의 질문 예시 제공."
        )
    else:
        prompt = (
            f"사용자가 한 질문: {last_msg.content}\n"
            f"불명확한 이유: {reason}\n\n"
            "사용자에게 보여줄 안내 메시지를 작성하세요. 형식은 다음과 같아야 합니다:\n"
            "첫 문단입니다. '질문은 다음과 같은 이유로 불명확합니다. 질문을 다시 입력해주세요.'\n"
            "두 번째 문단입니다. 불명확한 이유를 서술하세요.\n"
            "세 번째 문단입니다. '이렇게 질문하는건 어떨까요?' 형식으로, "
            "   사용자가 입력한 질문을 기반으로 조금 더 구체적이고 적절하게 만든 1~2개의 질문 예시 제공."
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
        f"작성일: {d.get('metadata', {}).get('date', '')}\n"
        f"출처: {d.get('metadata', {}).get('url', '')}\n"
        for i, d in enumerate(documents)
    ])
    
    # 시스템 메시지 생성
    system_message = SYSTEM_PROMPT.format(
        input=last_msg.content,
        documents=docs_text
    )

    # 이전 대화 요약이 존재하면 앞에 포함
    if summarization:
        system_message = f"이전 대화 요약:\n{summarization}\n\n" + system_message

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