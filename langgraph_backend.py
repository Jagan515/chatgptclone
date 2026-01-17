from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import sqlite3
import os
import re

load_dotenv()

#  LLM 

# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash-lite",
#     temperature=0.7,
# )
chat = HuggingFaceEndpoint(
    repo_id="HuggingFaceTB/SmolLM3-3B",  # Supported for chat_completion & text-generation
    task="text-generation",  # Or "conversational" if available
    temperature=0.7
)
llm = ChatHuggingFace(llm=chat)


# State 

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# ================= SQLite =================

conn = sqlite3.connect("chatbot.db", check_same_thread=False)

# ---- Create metadata table ----
conn.execute("""
CREATE TABLE IF NOT EXISTS chats (
    thread_id TEXT PRIMARY KEY,
    title TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

# ---- LangGraph Checkpointer ----
checkpointer = SqliteSaver(conn=conn)

# Graph 

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# DB Utilities 

def save_chat_title(thread_id: str, title: str):
    conn.execute(
        "INSERT OR IGNORE INTO chats (thread_id, title) VALUES (?, ?)",
        (thread_id, title),
    )
    conn.commit()

def get_all_chats():
    rows = conn.execute(
        "SELECT thread_id, title FROM chats ORDER BY created_at DESC"
    ).fetchall()
    return rows
