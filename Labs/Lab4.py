import streamlit as st
from openai import OpenAI
import tiktoken
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
import fitz  # PyMuPDF

# ----------------------------
# Paths (your PDFs are in Labs/Lab-04-Data)
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Lab-04-Data"

# ----------------------------
# Helpers: PDF text extraction
# ----------------------------
def extract_text_from_pdf(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# ----------------------------
# Chroma helpers
# ----------------------------
def add_to_collection(collection, text: str, file_name: str):
    # Chroma will embed automatically because the collection has embedding_function=openai_ef
    collection.add(
        documents=[text],
        ids=[file_name],
        metadatas=[{"source": file_name}]
    )

def load_pdfs_to_collection(folder_path: Path, collection) -> bool:
    pdf_files = sorted(folder_path.glob("*.pdf"))
    if not pdf_files:
        return False

    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        add_to_collection(collection, text, pdf_file.name)

    return True

def get_rag_context(collection, query: str, k: int = 3):
    """Return (context_text, sources_list) from Chroma for a user query."""
    results = collection.query(query_texts=[query], n_results=k)
    docs = results.get("documents", [[]])[0]
    ids = results.get("ids", [[]])[0]
    context_text = "\n\n---\n\n".join(docs) if docs else ""
    return context_text, ids

# ----------------------------
# OpenAI client (for chat)
# ----------------------------
if "client" not in st.session_state:
    st.session_state.client = OpenAI(api_key=st.secrets["OPEN_API_KEY"])

# ----------------------------
# ChromaDB (LOCAL) setup (persistent, avoids tenant errors)
# ----------------------------
if "Lab4_VectorDB" not in st.session_state:
    chroma_client = chromadb.PersistentClient(path=str(BASE_DIR / "chroma_db"))

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=st.secrets["OPEN_API_KEY"],
        model_name="text-embedding-3-small"
    )

    st.session_state.Lab4_VectorDB = chroma_client.get_or_create_collection(
        name="Lab4Collection",
        embedding_function=openai_ef
    )

collection = st.session_state.Lab4_VectorDB

# ----------------------------
# Load PDFs once (only if collection is empty)
# ----------------------------
debug = st.sidebar.checkbox("Debug RAG", value=False)

if debug:
    st.sidebar.write("Looking in:", str(DATA_DIR))
    st.sidebar.write("PDFs:", [p.name for p in DATA_DIR.glob("*.pdf")])

if collection.count() == 0:
    loaded = load_pdfs_to_collection(DATA_DIR, collection)
    if debug:
        if loaded:
            st.sidebar.success("PDFs loaded into ChromaDB collection.")
        else:
            st.sidebar.warning("No PDFs found to load.")

if debug:
    st.sidebar.write("Collection count:", collection.count())

# ----------------------------
# Token utilities (your existing code)
# ----------------------------
def count_tokens(messages, model_name):
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")

    total = 0
    for m in messages:
        total += len(enc.encode(m.get("content", "")))
        total += 4
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

def yes_no_intent(text: str) -> str:
    t = text.strip().lower()
    if t in {"yes", "y", "yeah", "yep", "sure", "yea"}:
        return "yes"
    if t in {"no", "n", "nope", "nah"}:
        return "no"
    return "other"

def stream_assistant_reply(client, model_to_use, messages_to_send):
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

# ----------------------------
# UI + Chatbot
# ----------------------------
st.title("Lab 4 App: PDF Q&A with ChromaDB and OpenAI")

openAI_model = st.sidebar.selectbox("Which Model?", ["mini", "regular"])
model_to_use = "gpt-4o-mini" if openAI_model == "mini" else "gpt-4o"

MAX_TOKENS = 1200

SYSTEM_PROMPT = (
    "Explain answers so a 10-year-old can understand (simple words, short sentences).\n"
    "After answering, ask exactly: Do you want more info?\n"
    "If the user says Yes, give more info about the last question and ask again.\n"
    "If the user says No, respond exactly: Okay—what can I help you with?\n"
)

if "awaiting_more_info" not in st.session_state:
    st.session_state.awaiting_more_info = False
if "last_question" not in st.session_state:
    st.session_state.last_question = ""

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask me anything!")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    intent = yes_no_intent(prompt)

    if st.session_state.awaiting_more_info and intent == "no":
        bot_text = "Okay—what can I help you with?"
        with st.chat_message("assistant"):
            st.markdown(bot_text)
        st.session_state.messages.append({"role": "assistant", "content": bot_text})
        st.session_state.awaiting_more_info = False
        st.session_state.last_question = ""

    else:
        if st.session_state.awaiting_more_info and intent == "yes":
            st.session_state.messages.append({
                "role": "user",
                "content": f"Give more info about: {st.session_state.last_question}"
            })
        elif not st.session_state.awaiting_more_info or intent == "other":
            st.session_state.last_question = prompt
            st.session_state.awaiting_more_info = True

        # -------- RAG: retrieve context from Chroma --------
        context_text, sources = get_rag_context(collection, prompt, k=3)

        if debug:
            st.sidebar.write("RAG sources:", sources)
            st.sidebar.write("Context chars:", len(context_text))
            st.sidebar.write("Context preview:", context_text[:400])

        rag_instruction = {
            "role": "system",
            "content": (
                "You are a course information chatbot.\n"
                "Use ONLY the CONTEXT below to answer.\n"
                "If the answer is not in the context, say exactly: "
                "'I could not find that information in the provided syllabi.'\n"
                "If you use the context, start your answer with: 'Based on the course documents,'\n\n"
                f"CONTEXT:\n{context_text}\n\n"
                f"SOURCES (filenames): {sources}"
            )
        }

        # ✅ THIS is the fixed part: prepend RAG instruction, then send a small chat tail
        messages_to_send = [rag_instruction] + last_two_user_turns(st.session_state.messages)
        messages_to_send = enforce_max_tokens(messages_to_send, model_to_use, MAX_TOKENS)

        tokens_sent = count_tokens(messages_to_send, model_to_use)
        st.sidebar.write(f"Tokens sent this request: {tokens_sent}")

        full_response = stream_assistant_reply(
            st.session_state.client,
            model_to_use,
            messages_to_send
        )
        st.session_state.messages.append({"role": "assistant", "content": full_response})


