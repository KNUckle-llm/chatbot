# from typing import Dict, Any
# import logging
# import json

# logger = logging.getLogger(__name__)


# # 메시지 파싱 유틸리티 함수
# def parse_message_content(message_json: str) -> Dict[str, Any]:
#     """MySQL에서 가져온 JSON 메시지를 파싱"""
#     try:
#         message_data = json.loads(message_json)
#         return message_data
#     except json.JSONDecodeError as e:
#         logger.error(f"JSON 파싱 오류: {e}")
#         return {"type": "unknown", "data": {"content": "파싱 오류"}}


# def extract_message_info(message_data: Dict[str, Any]) -> Dict[str, str]:
#     """메시지 데이터에서 타입과 내용 추출"""
#     try:
#         message_type = message_data.get("type", "unknown")

#         # AI 메시지인 경우
#         if message_type == "ai":
#             content = message_data.get("data", {}).get("content", "")
#         # Human 메시지인 경우
#         elif message_type == "human":
#             content = message_data.get("data", {}).get("content", "")
#         # 시스템 메시지인 경우
#         elif message_type == "system":
#             content = message_data.get("data", {}).get("content", "")
#         else:
#             # 다른 구조인 경우 content 필드 직접 찾기
#             content = message_data.get("content", str(message_data))

#         return {
#             "type": message_type,
#             "content": content
#         }

#     except Exception as e:
#         logger.error(f"메시지 정보 추출 오류: {e}")
#         return {
#             "type": "error",
#             "content": "메시지 파싱 중 오류가 발생했습니다."
#         }

