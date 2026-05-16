import os
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

# --- Config ---
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
SYSTEM_PROMPT = "Answer concisely"

# --- Client ---
@st.cache_resource
def build_client() -> Groq:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Set GROQ_API_KEY in .env or environment variables.")
    return Groq(api_key=api_key, http_client=None)

# --- LLM Call ---
def invoke_model(client: Groq, prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=1000
    )
    return response.choices[0].message.content.strip()

# --- UI ---
st.set_page_config(page_title="LLM Chatbot", page_icon="🤖")
st.title("🤖 Ask the LLM")
st.caption(f"Powered by Groq · {MODEL_NAME}")

client = build_client()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = invoke_model(client, prompt)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
