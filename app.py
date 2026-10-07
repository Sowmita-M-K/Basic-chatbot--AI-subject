import streamlit as st
import os
import sys

# Ensure local src directory is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from src.nlp_engine import DepartmentChatbot

# Page configuration
st.set_page_config(
    page_title="AI & DS Department Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Initialize NLP chatbot in session state
@st.cache_resource
def load_chatbot():
    return DepartmentChatbot()

bot = load_chatbot()

# Sidebar Information
with st.sidebar:
    st.header("🎓 AI & DS Department")
    st.markdown("**Academic Year:** 3rd Year (B.Tech / B.E.)")
    st.markdown("---")
    st.markdown("### 💡 Quick Prompts")
    quick_questions = [
        "What are 3rd year subjects?",
        "Tell me about lab sessions",
        "Mini project guidelines",
        "Minimum attendance required",
        "Who is the HOD?"
    ]

    for q in quick_questions:
        if st.button(q, use_container_width=True):
            st.session_state["user_prompt"] = q

    st.markdown("---")
    show_confidence = st.checkbox("Show Model Confidence & Intent", value=True)
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Main Header
st.title("🤖 AI & DS Department Chatbot")
st.caption("Ask questions about subjects, labs, faculty, attendance, and project guidelines.")

# Initialize chat message history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your AI & DS Department Assistant. Ask me anything about your 3rd-year coursework or college rules."
        }
    ]

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if show_confidence and "meta" in message:
            st.caption(f"🎯 *Intent:* `{message['meta']['tag']}` | *Score:* `{message['meta']['confidence']}`")

# Check if a button click populated user_prompt
prompt_to_send = None
if "user_prompt" in st.session_state and st.session_state["user_prompt"]:
    prompt_to_send = st.session_state.pop("user_prompt")
else:
    prompt_to_send = st.chat_input("Type your question here...")

# Handle query processing
if prompt_to_send:
    # Append user question
    st.session_state.messages.append({"role": "user", "content": prompt_to_send})
    with st.chat_message("user"):
        st.markdown(prompt_to_send)

    # Bot response
    result = bot.get_response(prompt_to_send)
    bot_reply = result["response"]

    meta_info = {
        "tag": result["tag"],
        "confidence": result["confidence"]
    }

    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_reply,
        "meta": meta_info
    })

    with st.chat_message("assistant"):
        st.markdown(bot_reply)
        if show_confidence:
            st.caption(f"🎯 *Intent:* `{meta_info['tag']}` | *Score:* `{meta_info['confidence']}`")
