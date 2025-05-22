from dotenv import load_dotenv
import os
from langchain_community.chat_message_histories.sql import SQLChatMessageHistory

# Load environment variables from .env file
load_dotenv()

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

llm.invoke("What is the capital of France?")

# # MySQL connection parameters from environment variables
# mysql_url = os.getenv("DATABASE_URL")

# # Create a SQLChatMessageHistory instance
# chat_history = SQLChatMessageHistory(
#     connection=mysql_url,
#     table_name="chat_history",
#     session_id="session-001",
#     session_id_field_name="session_id",
# )

# print(chat_history.clear())

# chat_history.add_user_message("Hello, how are you?")
# chat_history.add_ai_message("I'm fine, thank you! How can I assist you today?")

# print(chat_history.messages)
# print(chat_history.get_messages())