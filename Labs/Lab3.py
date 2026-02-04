import streamlit as st
from openai import OpenAI
import tiktoken

def count_tokens(messages, model_name):
    try: 
        enc = tiktoken.encoding_for_model(model_name)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")  # fallback encoding
    
    total = 0 
    for m in messages:
        total += len(enc.encode(m.get("content", "")))
        total += 4  # every message has a role and content, plus some overhead
    return total

def enforce_max_tokens(messages, model_name, max_tok):
    # Always keep one system message if present
    system = []
    rest = messages[:]

    if rest and rest[0]["role"] == "system":
        system = [rest[0]]
        rest = rest[1:]

    # Trim oldest non-system messages until under limit
    while rest and count_tokens(system + rest, model_name) > max_tok:
        rest.pop(0)

    # Never return empty: at minimum return system, otherwise return a fallback user message
    if system:
        return system + rest if (system + rest) else system
    return rest if rest else [{"role": "user", "content": "Hi"}]


max_tokens = st.sidebar.number_input("Max tokens to send to the model", min_value=200, max_value=8000, value=1200, step=100)    

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
        {"role": "system", "content": "You are a helpful assistant."},
    ]

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue  # Don't display system messages   
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def last_two_user_turns(messages):
    # keep the first system message (if any)
    system = [m for m in messages if m["role"] == "system"[:1]]
    
    # get the last two user turns
    user_idxs = [i for i, m in enumerate(messages) if m["role"] == "user"]
    if len (user_idxs) <= 2:
        return system + [m for m in messages if m["role"] != "system"]
    return system + system

# User input
prompt = st.chat_input("Ask me anything!")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    client = st.session_state.client
    messages_to_send = last_two_user_turns(st.session_state.messages)  
    messages_to_send = enforce_max_tokens(messages_to_send, model_to_use, max_tokens) 
    st.sidebar.write("Messages sent:", len(messages_to_send))
    stream = client.chat.completions.create(
        model=model_to_use,
        messages=messages_to_send,
        stream=True,
    )

    tokens_sent = count_tokens(messages_to_send, model_to_use)
    st.sidebar.write(f"Tokens sent this request: {tokens_sent}")

    full_response = ""
    with st.chat_message("assistant"):
        response_box = st.empty()
        for event in stream:
            delta = event.choices[0].delta.content
            if delta:
                full_response += delta
                response_box.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
