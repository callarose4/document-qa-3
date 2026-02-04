import tiktoken
import streamlit as st
from openai import OpenAI

st.title("MY Lab 3 question answering chatbot")

openAI_model = st.sidebar.selectbox("Which Model?", ["mini", "regular"])

if openAI_model == "mini":
    model_to_use = "gpt-4o-mini"
else:
    model_to_use = "gpt-4o"

# Create OpenAI client once
if "client" not in st.session_state:
    openai_api_key = st.secrets["OPEN_API_KEY"]
    st.session_state.client = OpenAI(api_key=openai_api_key)

# Initialize chat history once
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
prompt = st.chat_input("Ask me anything!")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    client = st.session_state.client
    stream = client.chat.completions.create(
        model=model_to_use,
        messages=st.session_state.messages,
        stream=True,
    )

    full_response = ""
    with st.chat_message("assistant"):
        response_box = st.empty()
        for event in stream:
            delta = event.choices[0].delta.content
            if delta:
                full_response += delta
                response_box.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
