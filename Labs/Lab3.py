import streamlit as st
from openai import OpenAI
import tiktoken

# ---------------- Part B helpers ----------------
def count_tokens(messages, model_name):
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")

    total = 0
    for m in messages:
        total += len(enc.encode(m.get("content", "")))
        total += 4  # approx overhead per message
    return total

def enforce_max_tokens(messages, model_name, max_tok):
    system = []
    rest = messages[:]

    if rest and rest[0]["role"] == "system":
        system = [rest[0]]
        rest = rest[1:]

    while rest and count_tokens(system + rest, model_name) > max_tok:
        rest.pop(0)

    if system:
        return system + rest if (system + rest) else system
    return rest if rest else [{"role": "user", "content": "Hi"}]

def last_two_user_turns(messages):
    system = [m for m in messages if m["role"] == "system"][:1]

    user_idxs = [i for i, m in enumerate(messages) if m["role"] == "user"]
    if len(user_idxs) <= 2:
        return system + [m for m in messages if m["role"] != "system"]

    start = user_idxs[-2]
    tail = [m for m in messages[start:] if m["role"] != "system"]
    return system + tail

# ---------------- Part C helpers ----------------
def yes_no_intent(text: str) -> str:
    t = text.strip().lower()
    if t in {"yes", "y", "yeah", "yep", "sure", "yea"}:
        return "yes"
    if t in {"no", "n", "nope", "nah"}:
        return "no"
    return "other"

# ---------------- One streaming function (removes redundancy) ----------------
def stream_assistant_reply(client, model_to_use, messages_to_send):
    """Calls the model with streaming, renders output, returns full text."""
    stream = client.chat.completions.create(
        model=model_to_use,
        messages=messages_to_send,
        stream=True,
    )

    full_response = ""
    with st.chat_message("assistant"):
        box = st.empty()
        for event in stream:
            delta = event.choices[0].delta.content
            if delta:
                full_response += delta
                box.markdown(full_response)
    return full_response

# ---------------- App ----------------
st.title("MY Lab 3 question answering chatbot")

openAI_model = st.sidebar.selectbox("Which Model?", ["mini", "regular"])
model_to_use = "gpt-4o-mini" if openAI_model == "mini" else "gpt-4o"

# keep max_tokens defined in the app (simpler than a slider)
MAX_TOKENS = 1200

SYSTEM_PROMPT = (
    "Explain answers so a 10-year-old can understand (simple words, short sentences).\n"
    "After answering, ask exactly: Do you want more info?\n"
    "If the user says Yes, give more info about the last question and ask again.\n"
    "If the user says No, respond exactly: Okay—what can I help you with?\n"
)

# Client
if "client" not in st.session_state:
    st.session_state.client = OpenAI(api_key=st.secrets["OPEN_API_KEY"])

# State for Part C
if "awaiting_more_info" not in st.session_state:
    st.session_state.awaiting_more_info = False
if "last_question" not in st.session_state:
    st.session_state.last_question = ""

# Messages
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

# Display history (skip system)
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask me anything!")

if prompt:
    # show/store user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    intent = yes_no_intent(prompt)

    # Decide what we want the model to do (or bypass model)
    if st.session_state.awaiting_more_info and intent == "no":
        bot_text = "Okay—what can I help you with?"
        with st.chat_message("assistant"):
            st.markdown(bot_text)
        st.session_state.messages.append({"role": "assistant", "content": bot_text})
        st.session_state.awaiting_more_info = False
        st.session_state.last_question = ""

    else:
        # If awaiting and user said yes, add a followup instruction tied to last_question
        if st.session_state.awaiting_more_info and intent == "yes":
            st.session_state.messages.append({
                "role": "user",
                "content": f"Give more info about: {st.session_state.last_question}"
            })
            # keep awaiting_more_info = True

        # Otherwise treat as a new question
        elif not st.session_state.awaiting_more_info or intent == "other":
            st.session_state.last_question = prompt
            st.session_state.awaiting_more_info = True

        # Build messages to send (Part B)
        messages_to_send = last_two_user_turns(st.session_state.messages)
        messages_to_send = enforce_max_tokens(messages_to_send, model_to_use, MAX_TOKENS)

        tokens_sent = count_tokens(messages_to_send, model_to_use)
        st.sidebar.write(f"Tokens sent this request: {tokens_sent}")

        # Single streaming call (no duplication)
        full_response = stream_assistant_reply(
            st.session_state.client,
            model_to_use,
            messages_to_send
        )
        st.session_state.messages.append({"role": "assistant", "content": full_response})


