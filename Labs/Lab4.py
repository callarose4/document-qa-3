import re
from pathlib import Path

import chromadb
import fitz  # PyMuPDF
import streamlit as st
import tiktoken
from chromadb.utils import embedding_functions
from openai import OpenAI


# ----------------------------
# Paths (your PDFs are in Labs/Lab-04-Data per your screenshot)
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Lab-04-Data"


# ----------------------------
# PDF text extraction
# ----------------------------
def extract_text_from_pdf(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


# ----------------------------
# Chunking + course parsing
# ----------------------------
def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200):
    """Split text into overlapping chunks to improve retrieval precision."""
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += max(1, chunk_size - overlap)
    return chunks


def extract_course_code(q: str):
    """
    Returns 'IST 314' if the user typed something like 'IST314' or 'IST 314'.
    Otherwise returns None.
    """
    m = re.search(r"\bIST\s*0?(\d{3})\b", q, re.IGNORECASE)
    if not m:
        return None
    return f"IST {m.group(1)}"


# ----------------------------
# Chroma add/load
# ----------------------------
def add_to_collection(collection, text: str, file_name: str):
    chunks = chunk_text(text)

    ids = [f"{file_name}::chunk{i}" for i in range(len(chunks))]
    metadatas = [{"source": file_name} for _ in chunks]

    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )


def load_pdfs_to_collection(folder_path: Path, collection) -> bool:
    pdf_files = sorted(folder_path.glob("*.pdf"))
    if not pdf_files:
        return False

    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        add_to_collection(collection, text, pdf_file.name)

    return True


# ----------------------------
# RAG retrieval
# (filters to the course syllabus if user mentions IST ###)
# ----------------------------
def get_rag_context(collection, query: str, k: int = 6):
    course = extract_course_code(query)

    # Try querying with metadatas included (your Chroma does NOT allow "ids" in include)
    try:
        results = collection.query(
            query_texts=[query],
            n_results=k * 3,  # grab more, then filter down
            include=["documents", "metadatas"],
        )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
    except TypeError:
        # Fallback for older Chroma versions that don't support include=
        results = collection.query(query_texts=[query], n_results=k * 3)
        docs = results.get("documents", [[]])[0]
        metas = [{} for _ in docs]  # no metadatas available

    # If they asked about a specific course, keep only chunks from that syllabus filename
    if course:
        filtered_docs = []
        filtered_sources = set()

        for d, m in zip(docs, metas):
            src = (m or {}).get("source", "")
            if course.lower() in src.lower():
                filtered_docs.append(d)
                if src:
                    filtered_sources.add(src)

        # If filtering found something, use it; otherwise fall back to unfiltered
        if filtered_docs:
            docs = filtered_docs
            sources = sorted(filtered_sources)
        else:
            sources = sorted({(m or {}).get("source", "") for m in metas if (m or {}).get("source")})
    else:
        sources = sorted({(m or {}).get("source", "") for m in metas if (m or {}).get("source")})

    # Keep top-k
    docs = docs[:k]
    sources = sources[:5]
    context_text = "\n\n---\n\n".join(docs) if docs else ""

    return context_text, sources


# ----------------------------
# OpenAI client (chat)
# ----------------------------
if "client" not in st.session_state:
    st.session_state.client = OpenAI(api_key=st.secrets["OPEN_API_KEY"])


# ----------------------------
# Chroma local persistent setup (avoids tenant errors)
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
# Optional debug + rebuild controls
# ----------------------------
debug = st.sidebar.checkbox("Debug RAG", value=False)

st.sidebar.write("Data folder:", str(DATA_DIR))
if debug:
    st.sidebar.write("PDFs found:", [p.name for p in DATA_DIR.glob("*.pdf")])

# IMPORTANT: chunking requires a fresh DB. Provide a button so you can rebuild easily.
if st.sidebar.button("Rebuild Vector DB (delete + reload PDFs)"):
    # This deletes all embeddings in the collection and reloads from PDFs.
    # It’s safe for this lab.
    try:
        all_ids = collection.get().get("ids", [])
        if all_ids:
            collection.delete(ids=all_ids)
    except Exception:
        # If delete/get behaves differently, just warn and continue
        pass

    loaded = load_pdfs_to_collection(DATA_DIR, collection)
    if loaded:
        st.sidebar.success("Rebuilt vector DB from PDFs.")
    else:
        st.sidebar.warning("No PDFs found to load. Check Lab-04-Data folder name/location.")

# If empty, load once
if collection.count() == 0:
    loaded = load_pdfs_to_collection(DATA_DIR, collection)
    if debug:
        st.sidebar.write("Loaded PDFs:", loaded)

if debug:
    st.sidebar.write("Collection count:", collection.count())


# ----------------------------
# Token utilities
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

# Render chat history
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask me anything!")

if prompt:
    # store user message
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

        # -------- RAG retrieval --------
        context_text, sources = get_rag_context(collection, prompt, k=6)

        if debug:
            st.sidebar.write("RAG sources:", sources)
            st.sidebar.write("Context chars:", len(context_text))
            st.sidebar.write("Context preview:", context_text[:500])

        rag_instruction = {
            "role": "system",
            "content": (
                "You are a course information chatbot.\n"
                "Use ONLY the CONTEXT below to answer the user's question.\n"
                "If the answer is not in the context, say exactly: "
                "'I could not find that information in the provided syllabi.'\n"
                "If you use the context, start your answer with: 'Based on the course documents,'\n"
                "When possible, be specific and quote small phrases from the context.\n\n"
                f"CONTEXT:\n{context_text}\n\n"
                f"SOURCES (filenames): {sources}"
            )
        }

        # ✅ Correct message assembly: RAG system message FIRST, then short chat tail
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


