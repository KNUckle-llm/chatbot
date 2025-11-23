import re
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
    messages = state.get("messages")
    current_question = messages[-1].content  # 현재 사용자 질문
    prev_department = state.get("current_department")

    if state.get("follow_up_chain") is None:
        state["follow_up_chain"] = []
    
    # 🔹 현재 질문을 체인에 append    
    state["follow_up_chain"].append(current_question)
    logger.info(f"현재 질문 append 후 follow_up_chain: {state['follow_up_chain']}")
    
    is_follow_up = False
    # 🔹 체인이 2개 이상일 때만 follow-up 판단
    if len(state["follow_up_chain"]) > 1:
        previous_questions = " / ".join(state["follow_up_chain"][:-1])
    
        followup_prompt = (
            "너는 공주대학교 정보를 안내하는 챗봇이다.\n"
            f"현재 질문: {current_question}\n"
            f"이전 질문들: {previous_questions}\n"
            f"관련 학과: {prev_department}\n"
            
            "현재 질문이 follow-up인지 판단하여 반드시 영문 yes/no 둘중에 하나로만 답하세요.\n"
            
            "판단 기준:\n"
            "- 동일한 대상/행사/문서 등에 대한 추가 질문이면 follow-up\n"
            "- '그럼, 그거, 그러면'처럼 이전 질문을 지시하면 follow-up\n"
            "- 질문 대상이나 주제가 바뀌면 follow-up 아님\n"
        )
        followup_response = model.invoke([SystemMessage(content=followup_prompt)])
        followup_text = followup_response.content.strip().lower()
        is_follow_up = followup_text.startswith("yes")
        
        if is_follow_up:
            # 🔹 FOLLOW-UP 처리: 체인 유지, question_appropriate True
            state["follow_up"] = True
            logger.info(f"Follow-up 판단: YES, follow_up_chain 유지: {state['follow_up_chain']}")

            # 🔹 FOLLOW-UP 질문 재작성 (체인 기반)
            combined_question = " / ".join(state["follow_up_chain"])
            rewrite_prompt = (
                f"이전 질문들을 참고하여, 마지막 질문을 자연스럽게 검색하기 적합한 한 문장으로 바꾸세요.\n"
                f"{combined_question}"
            )
            rewritten = model.invoke([SystemMessage(content=rewrite_prompt)]).content.strip()
            state["follow_up_chain"][-1] = rewritten  # 마지막 질문을 재작성
            logger.info(f"Follow-up 질문 재작성: {rewritten}")

            state["question_appropriate"] = True
            state["question_reason"] = None
            return {
                "follow_up": state["follow_up"],
                "question_appropriate": state["question_appropriate"],
                "follow_up_chain": list(state.get("follow_up_chain", []))
            }
        else:
            # 연관 없는 새 질문이면 follow-up False, 체인 초기화 후 현재 질문만 남김
            state["follow_up"] = False
            state["follow_up_chain"] = [current_question]
            logger.info(f"Follow-up 판단: NO, follow_up_chain 초기화 후 상태: {state['follow_up_chain']}")

        logger.info(f"Follow-up 판단 결과: {is_follow_up}, 체인 상태: {state['follow_up_chain']}")
   

    # 🔹 질문 적절성 판단 (follow-up이 아니면)
    appropriateness_prompt = (
        "너는 공주대학교 정보를 알려주는 챗봇입니다.\n"
        f"사용자 질문: {current_question}\n"
        "아래 기준을 바탕으로 현재 사용자 질문이 검색 가능한 문서로 답변 가능한지 판단하세요.\n\n"

        "### 판단 기준\n" 
        "1) 검색 가능한 문서 범위 내에서 답변 가능한 질문이면 'yes'입니다.\n"
        "   단, 부서가 명시적으로 적혀있지 않으면 'no'입니다.\n"

        "2) 검색 가능한 문서 범위는 다음과 같습니다.\n"
        "   - 공주대학교 통합 수강신청/장학/비자/논문/순환버스\n"
        "   - 학과별 교수님(연락처, 이메일 등)/교과과정표/공지사항/자료/서식/규정\n"
        "   - SW사업단 소개/공지사항/소식/대회일정(TOPCIT, SW알고리즘 경진대회 등)\n"

        "3) 개인정보 포함 여부는 적절성 판단 기준이 아닙니다.\n"
        "   위 판단 기준으로 답이 가능한지 여부만 고려하세요.\n\n"

        "### 출력 형식\n"
        "- 첫 줄: 반드시 영어 'yes' 또는 'no'로 시작 (대소문자 혼용 금지)\n"
        "- 둘째 줄: 판단 이유 1~2문장\n"
        "  - yes일 때: 사용자 질문에 대하여 왜 검색이 가능한지 설명합니다.\n"
        "  - no일 때: 사용자 질문에 대하여 왜 검색을 진행 못하는지 설명합니다. (질문이 불명확한 이유)\n"
    )
    response = model.invoke([SystemMessage(content=appropriateness_prompt)])
    raw_text = response.content.strip()
    
    # 🔹 Regex로 yes/no 체크
    match = re.match(r"^(yes|no)", raw_text.lower())
    if match:
        if match.group(1) == "no":
            state["question_appropriate"] = False
            state["question_reason"] = raw_text[len(match.group(1)):].strip()
        else:
            state["question_appropriate"] = True
            state["question_reason"] = None
    else:
        logger.warning("LLM 출력이 예상 형식과 다릅니다. 기본값 no 처리")
        state["question_appropriate"] = False
        state["question_reason"] = "LLM 출력 형식 오류"

    logger.info(f"follow_up_chain: {state['follow_up_chain']}")
    logger.info(f"question_appropriate: {state['question_appropriate']}, reason: {state.get('question_reason')}")
    return {
        "follow_up": state.get("follow_up", False),
        "question_appropriate": state["question_appropriate"],
        "question_reason": state.get("question_reason", None),
        "follow_up_chain": list(state.get("follow_up_chain", []))
    }



def route_before_retrieval_node(state: CustomState) -> Literal["retrieve", "rewrite_question"]:
    logger.info(">>> [NODE] route_before_retrieval_node START")
    # follow-up이면 바로 retrieve
    if state.get("follow_up"):
        return "retrieve"
    # follow-up 아니더라도 적절성 판단 결과에 따라 결정
    return "retrieve" if state.get("question_appropriate") else "rewrite_question"



def retrieve_documents_node(state: CustomState, max_docs: int = 2):
    logger.info(">>> [NODE] retrieve_documents_node START")
    messages = state.get("messages")
    #query = messages[-1].content
    query = state['follow_up_chain'][-1].strip()
    follow_up = state.get("follow_up", False)
    logger.info(f"retrieve_documents_node: follow_up={follow_up}, current_department={state.get('current_department')}")

    # 학과 후보 리스트
    departments = [
        "소프트웨어학과",
        "컴퓨터공학과",
        "공주대학교",
        "공주대학교 SW중심대학사업단",
        "SW중심대학사업단",
        "스마트정보기술공학과",
        "인공지능학부",
        # "공주대학교 현장실습지원센터"
    ]
    
    # 2) alias 매핑 (여기서 OR 조건 처리)
    alias_map = {
        "공주대학교 SW중심대학사업단": ["공주대학교 SW중심대학사업단", "SW중심대학사업단"],
        "SW중심대학사업단": ["공주대학교 SW중심대학사업단", "SW중심대학사업단"],
    }


    # FOLLOW-UP이면 이전 학과 유지, 재예측 금지
    if follow_up and state.get("current_department"):
        predicted_department = state["current_department"]
        logger.info(f"Follow-up이므로 이전 학과 유지: {predicted_department}")
    else:
        dept_prompt = (
            f"사용자 질문: {query}\n"
            f"질문을 보고 아래 목록 중에서 관련 학과/부서를 하나 선택하세요:\n"
            f"반드시 목록 중 하나를 그대로 출력하세요.\n"
            f"목록: {', '.join(departments)}"
        )
        dept_response = model.invoke([SystemMessage(content=dept_prompt)])
        predicted_department = dept_response.content.strip()
        logger.info(f"Predicted department: {predicted_department}")

    state["current_department"] = predicted_department
    
    # 🔹 쿼리 확장
    last_question = state['follow_up_chain'][-1]
    extended_query = last_question.strip()
    logger.info(f"검색용 extended_query (마지막 질문 기준): {extended_query}")
    
    # store에서 검색
    if predicted_department in departments:
        aliases = alias_map.get(predicted_department, [predicted_department])
        filter_expr = {"department": {"$in": aliases}}
        logger.info(f"Using filter: {filter_expr}")
        docs = store.similarity_search(extended_query, k=max_docs, filter=filter_expr)
    else:
        logger.info("Predicted department not recognized. Running search without filter.")
        docs = store.similarity_search(extended_query, k=max_docs)

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
    
    logger.info(f"Retrieved {len(docs)} documents for query: {extended_query}")
    return {"documents": state["documents"]}



def rewrite_question_node(state: CustomState):
    logger.info(">>> [NODE] rewrite_question_node START")
    if state.get("question_appropriate"):
        return {"messages": state.get("messages")}

    # last_msg = state.get("messages")[-1]
    last_question = state['follow_up_chain'][-1]
    reason = state.get("question_reason", "불명확한 이유 없음")
    
    prompt = (
        f"사용자가 한 질문: {last_question}\n"
        f"불명확한 이유: {reason}\n\n"
        "사용자에게 보여줄 안내 메시지를 작성하세요. 형식은 다음과 같아야 합니다:\n"
        "첫 문단입니다. '질문은 다음과 같은 이유로 불명확합니다. 질문을 다시 입력해주세요.'\n"
        "두 번째 문단에는 불명확한 이유를 서술하세요.\n"
        "세 번째 문단입니다. '이렇게 질문하는건 어떨까요?' 형식으로,\n"
        "   사용자가 입력한 질문과 불명확한 이유를 기반으로 더 구체적이고 적절한 1~2개의 질문을 예시로 제공.(bullet형)"
    )
    
    # AI메세지 추가
    response = model.invoke([SystemMessage(content=prompt)])
    state.get("messages").append(response)
    logger.info("Rewritten question/feedback added.")

    return {"messages": state.get("messages")}


def generation_node(state: CustomState):
    logger.info(">>> [NODE] generation_node START")
    language = state.get("language", "ko")
    documents = state.get("documents", [])
    summarization = state.get("summarization", "")
    
    #last_msg = state.get("messages")[-1]
    last_question = state['follow_up_chain'][-1]
    
    # 문서 내용 그대로 전달 + 개행 유지 + 문서 사이 빈 줄 추가
    docs_text = "\n\n---\n\n".join([
        f"[검색된 문서 {i+1}]\n\n"
        f"본문 내용:\n{d['content']}\n\n"
        f"제목:\n{d.get('metadata', {}).get('file_name', '')}\n\n"
        f"부서:\n{d.get('metadata', {}).get('department', '')}\n\n"
        f"작성일:\n{d.get('metadata', {}).get('date', '')}\n\n"
        f"출처:\n{d.get('metadata', {}).get('url', '')}\n"
        for i, d in enumerate(documents)
    ])
    
    # 시스템 메시지 생성
    system_message = SYSTEM_PROMPT.format(
        #input=last_msg.content,
        input=last_question,
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