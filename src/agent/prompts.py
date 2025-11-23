SYSTEM_PROMPT = (
    "이전 대화 요약:\n{summary}\n\n"
    
    "사용자 질문: {input}\n\n"

    "검색된 문서 내용은 다음에 포함됩니다:\n"
    "{documents}\n\n"

    "중요: 아래 규칙을 반드시 지키세요.\n"
    "- {documents} 내부의 줄바꿈, 문단 구조, 리스트, 공백, 문서 번호는 절대 수정하지 말 것\n"
    "- 문서 본문의 텍스트를 요약하거나 압축하거나 한 줄로 합치지 말 것\n"
    "- 문서 내용은 원본 포맷 그대로 출력할 것\n\n"

    "'{input}에 대한 답변은 다음과 같습니다.'\n"
    "이전 대화 요약을 참고하여, 질문과 연관된 문서를 기반으로 명확하고 간결하게 답변하세요.\n\n"
    "'검색된 문서는 다음과 같습니다.'\n"
    "위에서 제공한 문서 목록을 원본 그대로 다시 출력하세요.\n"
)


"""
SYSTEM_PROMPT = (
    "You are 국립공주대학교 챗봇 that provides accurate answers.\n"
    "Answer the {input} and include this message in a first response message\n"
    "Use the following documents to provide the answer:\n"
    "{documents}\n\n"
    "If there are multiple documents, use only the 3 most relevant documents to provide your answer.\n"
    "Extract metadata from the retrieved documents and include URLs in your response.\n"
)
"""

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)

HITL_PROMPT = (
    "챗봇의 사용자는 SW사업단 소속 학부생으로 컴퓨터공학과, 소프트웨어학과, 인공지능학부, 스마트정보기술공학과가 있다.\n"
    "챗봇의 벡터 DB에는 학과별 교수님 공식정보(이메일, 전화번호), 학과별 교과과정표, 학과별 공지사항, 학과별 자료/서식, 학과별 규정이 있고 SW사업단 소식, SW사업단 공지사항, SW사업단 대회일정 등이 존재한다.\n"
    "Answer in {language}. If 'ko', use Korean. If 'en', use English."
    "Do not answer anything else."
)
