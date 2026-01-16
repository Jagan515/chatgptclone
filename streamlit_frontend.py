import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage
import time

CONFIG = {"configurable": {"thread_id": "thread-1"}}

# Initialize session state
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# Load conversation history (use markdown!)
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    # Save & render user message
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # Stream assistant response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        buffer = ""

        for message_chunk, metadata in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=CONFIG,
            stream_mode="messages",
        ):
            if not message_chunk.content:
                continue

            buffer += message_chunk.content

            # Render only when safe (markdown stability)
            if buffer.endswith(("\n", " ", ".", "!", "?")):
                full_response += buffer
                buffer = ""

                placeholder.markdown(full_response)
                time.sleep(0.015)  # ChatGPT-like speed

        # Flush remaining buffer
        if buffer:
            full_response += buffer
            placeholder.markdown(full_response)

    # Save assistant message AFTER streaming completes
    st.session_state["message_history"].append(
        {"role": "assistant", "content": full_response}
    )
