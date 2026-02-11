import streamlit as st
from openai import OpenAI
import tiktoken
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
import fitz  # PyMuPDF


### using chroma db with openai embeddings
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets["OPEN_API_KEY"])

#client created
client = chromadb.Client()

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=st.secrets["OPEN_API_KEY"],
    model_name="text-embedding-3-small"
)

# create or get collection
collection = client.get_or_create_collection(name="lab4_collection", embedding_function=openai_ef)
# A function that will add documents to collectiin
# Collection = collection, already established
# text = extracted text from PDF files
# Embeddings inserted into the collection from OpenAI
def add_to_collection(collection, text, file_name):
    
    # create an embedding function
    client = st.session_state.openai_client
    response = client.embeddings.create(
        input=[text], model="text-embedding-3-small"
    )

    #get the embedding
    embedding = response.data[0].embedding


    #add embedding and document to ChromaDB
    collection.add(
        documents=[text],
        ids=[file_name],
        embeddings=[embedding],
    )

# EXTRACT TEXT FROM PDF ####
# This function extracts text from each syllabus to pass to add_to_collection
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

### POPULATE COLLECTION WITH PDFS ####
# this function uses extract_text_from_pdf and add_to_collection to put syllabi in the ChromaDB collection 
def load_pdfs_to_collection(folder_path, collection):
    pdf_files = path.glob(folder_path + "/*.pdf")
    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        add_to_collection(collection, text, pdf_file.name)

#check if collection is empty and load pdfs
if collection.count() == 0:
    loaded = load_pdfs_to_collection('.Lab-04-Data/', collection)
    if loaded:
        st.write("PDFs loaded into ChromaDB collection.")
    else:
        st.write("No PDFs found to load.")  



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


st.title("MY Lab 3 question answering chatbot")

openAI_model = st.sidebar.selectbox("Which Model?", ["mini", "regular"])
model_to_use = "gpt-4o-mini" if openAI_model == "mini" else "gpt-4o"

# keep max_tokens defined in the app 
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

    # Decide what we want the model to do 
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

        # Single streaming call 
        full_response = stream_assistant_reply(
            st.session_state.client,
            model_to_use,
            messages_to_send
        )
        st.session_state.messages.append({"role": "assistant", "content": full_response})


