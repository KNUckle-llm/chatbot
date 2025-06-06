from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import logging
from datetime import datetime
from langchain_community.chat_message_histories.sql import SQLChatMessageHistory
import pymysql
from urllib.parse import urlparse
from chatbot_api.services.chat_history_service import parse_message_content, extract_message_info

# 로깅 설정
logger = logging.getLogger(__name__)

router = APIRouter()


# 응답 모델들
class ChatMessage(BaseModel):
    id: Optional[int] = None
    session_id: str
    message_type: str  # "human" 또는 "ai"
    content: str
    timestamp: Optional[datetime] = None
    additional_data: Optional[Dict[str, Any]] = None


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: List[ChatMessage]
    total_messages: int
    created_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None


class SessionListResponse(BaseModel):
    sessions: List[Dict[str, Any]]
    total_sessions: int


def get_db_connection():
    """PyMySQL 데이터베이스 연결 생성"""
    mysql_url = os.getenv("DATABASE_URL")
    if not mysql_url:
        raise HTTPException(status_code=500, detail="데이터베이스 연결 정보가 없습니다")

    parsed_url = urlparse(mysql_url)

    return pymysql.connect(
        host=parsed_url.hostname,
        port=parsed_url.port or 3306,
        user=parsed_url.username,
        password=parsed_url.password,
        database=parsed_url.path[1:],  # Remove leading '/'
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor,  # 딕셔너리 형태로 결과 반환
        autocommit=True
    )


# 채팅 기록 조회 엔드포인트
@router.get("/history/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(
        session_id: str,
        limit: Optional[int] = Query(default=50, description="가져올 메시지 수 제한"),
        offset: Optional[int] = Query(default=0, description="건너뛸 메시지 수")
):
    """특정 세션의 채팅 기록 조회"""
    try:
        # 데이터베이스 URL 가져오기
        mysql_url = os.getenv("DATABASE_URL")
        if not mysql_url:
            raise HTTPException(status_code=500, detail="데이터베이스 연결 정보가 없습니다")

        # SQLChatMessageHistory 인스턴스 생성
        chat_history = SQLChatMessageHistory(
            connection=mysql_url,
            table_name="chat_history",
            session_id=session_id,
            session_id_field_name="session_id",
        )

        # 메시지 가져오기
        messages = chat_history.messages

        # 메시지가 없는 경우
        if not messages:
            return ChatHistoryResponse(
                session_id=session_id,
                messages=[],
                total_messages=0
            )

        # 메시지 파싱 및 변환
        parsed_messages = []

        for i, message in enumerate(messages[offset:offset + limit]):
            try:
                # LangChain 메시지 객체에서 정보 추출
                message_type = message.type if hasattr(message, 'type') else "unknown"
                content = message.content if hasattr(message, 'content') else str(message)

                # 메시지 타입 정규화
                if message_type == "ai":
                    display_type = "ai"
                elif message_type == "human":
                    display_type = "human"
                elif message_type == "system":
                    display_type = "system"
                else:
                    display_type = "unknown"

                parsed_message = ChatMessage(
                    id=offset + i + 1,
                    session_id=session_id,
                    message_type=display_type,
                    content=content,
                    timestamp=datetime.now(),  # 실제로는 DB에서 timestamp 가져와야 함
                    additional_data={
                        "original_type": message_type,
                        "has_additional_kwargs": hasattr(message, 'additional_kwargs'),
                        "response_metadata": getattr(message, 'response_metadata', {})
                    }
                )

                parsed_messages.append(parsed_message)

            except Exception as e:
                logger.error(f"메시지 파싱 오류 (index {i}): {e}")
                # 오류가 발생한 메시지도 포함하되 오류 표시
                error_message = ChatMessage(
                    id=offset + i + 1,
                    session_id=session_id,
                    message_type="error",
                    content=f"메시지 파싱 중 오류: {str(e)}",
                    timestamp=datetime.now()
                )
                parsed_messages.append(error_message)

        # 전체 메시지 수 계산
        total_messages = len(messages)

        # 세션 생성/수정 시간 (첫 번째와 마지막 메시지 기준)
        created_at = datetime.now()  # 실제로는 첫 번째 메시지 시간
        last_updated = datetime.now()  # 실제로는 마지막 메시지 시간

        return ChatHistoryResponse(
            session_id=session_id,
            messages=parsed_messages,
            total_messages=total_messages,
            created_at=created_at,
            last_updated=last_updated
        )

    except Exception as e:
        logger.error(f"채팅 기록 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"채팅 기록 조회 중 오류가 발생했습니다: {str(e)}")


# 모든 세션 목록 조회 엔드포인트
@router.get("/sessions", response_model=SessionListResponse)
async def get_chat_sessions(
        limit: Optional[int] = Query(default=20, description="가져올 세션 수 제한"),
        offset: Optional[int] = Query(default=0, description="건너뛸 세션 수")
):
    """모든 채팅 세션 목록 조회"""
    try:
        # PyMySQL 연결
        connection = get_db_connection()
        cursor = connection.cursor()

        # 세션별 메시지 수와 최신 메시지 시간 조회
        query = """
        SELECT 
            session_id,
            COUNT(*) as message_count,
            MIN(id) as first_message_id,
            MAX(id) as last_message_id,
            MAX(id) as last_activity
        FROM chat_history 
        GROUP BY session_id 
        ORDER BY last_activity DESC 
        LIMIT %s OFFSET %s
        """

        cursor.execute(query, (limit, offset))
        sessions_data = cursor.fetchall()

        # 전체 세션 수 조회
        cursor.execute("SELECT COUNT(DISTINCT session_id) as total FROM chat_history")
        total_result = cursor.fetchone()
        total_sessions = total_result['total'] if total_result else 0

        # 각 세션의 첫 메시지와 마지막 메시지 내용 가져오기
        sessions = []
        for session_data in sessions_data:
            session_id = session_data['session_id']

            # 첫 번째 메시지 가져오기
            cursor.execute(
                "SELECT message FROM chat_history WHERE session_id = %s ORDER BY id ASC LIMIT 1,1",
                (session_id,)
            )
            first_message_result = cursor.fetchone()
            first_message = ""
            if first_message_result:
                try:
                    first_msg_data = parse_message_content(first_message_result['message'])
                    first_message = extract_message_info(first_msg_data)['content'][:100] + "..."
                except:
                    first_message = "메시지 파싱 오류"

            # 마지막 메시지 가져오기
            cursor.execute(
                "SELECT message FROM chat_history WHERE session_id = %s ORDER BY id DESC LIMIT 1",
                (session_id,)
            )
            last_message_result = cursor.fetchone()
            last_message = ""
            if last_message_result:
                try:
                    last_msg_data = parse_message_content(last_message_result['message'])
                    last_message = extract_message_info(last_msg_data)['content'][:100] + "..."
                except:
                    last_message = "메시지 파싱 오류"

            sessions.append({
                "session_id": session_id,
                "message_count": session_data['message_count'],
                "first_message": first_message,
                "last_message": last_message,
                "last_activity": session_data['last_activity']
            })

        cursor.close()
        connection.close()

        return SessionListResponse(
            sessions=sessions,
            total_sessions=total_sessions
        )

    except Exception as e:
        logger.error(f"세션 목록 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"세션 목록 조회 중 오류가 발생했습니다: {str(e)}")


# 특정 세션 삭제 엔드포인트
@router.delete("/history/{session_id}")
async def delete_chat_session(session_id: str):
    """특정 세션의 채팅 기록 삭제"""
    try:
        mysql_url = os.getenv("DATABASE_URL")
        if not mysql_url:
            raise HTTPException(status_code=500, detail="데이터베이스 연결 정보가 없습니다")

        # SQLChatMessageHistory 인스턴스 생성
        chat_history = SQLChatMessageHistory(
            connection=mysql_url,
            table_name="chat_history",
            session_id=session_id,
            session_id_field_name="session_id",
        )

        # 메시지 수 확인
        message_count = len(chat_history.messages)

        if message_count == 0:
            raise HTTPException(status_code=404, detail="해당 세션이 존재하지 않습니다")

        # 메시지 삭제 (LangChain의 clear 메서드 사용)
        chat_history.clear()

        # session 테이블에서도 삭제 (선택사항)
        try:
            connection = get_db_connection()
            cursor = connection.cursor()
            cursor.execute("DELETE FROM session WHERE session_id = %s", (session_id,))
            cursor.close()
            connection.close()
        except Exception as e:
            logger.warning(f"session 테이블 삭제 중 오류 (무시 가능): {e}")

        return {
            "message": f"세션 '{session_id}'의 {message_count}개 메시지가 삭제되었습니다",
            "session_id": session_id,
            "deleted_count": message_count
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"세션 삭제 오류: {e}")
        raise HTTPException(status_code=500, detail=f"세션 삭제 중 오류가 발생했습니다: {str(e)}")


# 세션 통계 조회 엔드포인트
@router.get("/sessions/stats")
async def get_chat_sessions_stats():
    """채팅 세션 통계 정보 조회"""
    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        # 통계 쿼리들
        stats_queries = {
            "total_sessions": "SELECT COUNT(DISTINCT session_id) as count FROM chat_history",
            "total_messages": "SELECT COUNT(*) as count FROM chat_history",
            "avg_messages_per_session": """
                SELECT AVG(message_count) as avg_count 
                FROM (
                    SELECT session_id, COUNT(*) as message_count 
                    FROM chat_history 
                    GROUP BY session_id
                ) as session_stats
            """,
            "most_active_session": """
                SELECT session_id, COUNT(*) as message_count 
                FROM chat_history 
                GROUP BY session_id 
                ORDER BY message_count DESC 
                LIMIT 1
            """,
            "recent_activity": """
                SELECT COUNT(DISTINCT session_id) as active_sessions 
                FROM chat_history 
                WHERE id > (
                    SELECT MAX(id) - 100 FROM chat_history
                )
            """
        }

        stats = {}

        for stat_name, query in stats_queries.items():
            cursor.execute(query)
            result = cursor.fetchone()

            if stat_name == "most_active_session":
                stats[stat_name] = result if result and result['message_count'] else None
            elif stat_name == "avg_messages_per_session":
                stats[stat_name] = round(float(result['avg_count']), 2) if result and result['avg_count'] else 0
            else:
                stats[stat_name] = result['count'] if result else 0

        cursor.close()
        connection.close()

        return {
            "statistics": stats,
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"통계 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"통계 조회 중 오류가 발생했습니다: {str(e)}")


# 세션 정보 조회 엔드포인트 (새로 추가)
@router.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    """특정 세션 정보 조회"""
    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        # session 테이블에서 세션 정보 조회
        cursor.execute(
            "SELECT session_id, user_id, started_at, ended_at FROM session WHERE session_id = %s",
            (session_id,)
        )
        session_info = cursor.fetchone()

        if not session_info:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")

        # 해당 세션의 메시지 수 조회
        cursor.execute(
            "SELECT COUNT(*) as message_count FROM chat_history WHERE session_id = %s",
            (session_id,)
        )
        message_result = cursor.fetchone()
        message_count = message_result['message_count'] if message_result else 0

        cursor.close()
        connection.close()

        return {
            "session_info": session_info,
            "message_count": message_count
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"세션 정보 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"세션 정보 조회 중 오류가 발생했습니다: {str(e)}")


# 새 세션 생성 엔드포인트 (새로 추가)
@router.post("/sessions/create")
async def create_new_session(user_id: Optional[str] = "dummy_user_001"):
    """새 세션 생성"""
    try:
        import uuid

        # 고유 세션 ID 생성
        session_id = f"session_{uuid.uuid4().hex[:12]}"

        # session 테이블에 레코드 생성
        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            "INSERT INTO session (session_id, user_id, started_at) VALUES (%s, %s, %s)",
            (session_id, user_id, datetime.now())
        )

        cursor.close()
        connection.close()

        logger.info(f"새 세션 생성: {session_id}")

        return {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"세션 생성 오류: {e}")
        raise HTTPException(status_code=500, detail=f"세션 생성 중 오류가 발생했습니다: {str(e)}")


# 사용 예시
"""
# 1. 특정 세션의 채팅 기록 조회
GET /history/session_123?limit=20&offset=0

# 2. 모든 세션 목록 조회
GET /sessions?limit=10&offset=0

# 3. 세션 삭제
DELETE /history/session_123

# 4. 통계 조회
GET /sessions/stats

# 5. 세션 정보 조회 (새로 추가)
GET /sessions/session_123

# 6. 새 세션 생성 (새로 추가)  
POST /sessions/create
"""