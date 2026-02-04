import streamlit as st
from openai import OpenAI
import fitz # PyMuPDF

st.title("Lab 2")
st.write("Upload a PDF below and the app will summarize it.")

summary_type = st.sidebar.selectbox(
    "Summary type",
    ["100 words", "2 connected paragraphs", "5 bullet points"]
)
language = st.sidebar.selectbox("Output language", ["English", "French", "Spanish"])


use_advanced = st.sidebar.checkbox("Use advanced model")
model = "gpt-4" if use_advanced else "gpt-4.1-nano"

openai_api_key = st.secrets["OPEN_API_KEY"]  # make sure this matches your secrets key
client = OpenAI(api_key=openai_api_key)

def read_pdf(uploaded_file):
    text = ""
    pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    for page in pdf:
        text += page.get_text()
    return text

uploaded_file = st.file_uploader("Upload a document (.pdf)", type=("pdf",))

if uploaded_file:
    document = read_pdf(uploaded_file)

    if summary_type == "100 words":
        instruction = "Summarize the document in exactly 100 words."
    elif summary_type == "2 connected paragraphs":
        instruction = "Summarize the document in 2 connected paragraphs."
    else:
        instruction = "Summarize the document in 5 concise bullet points."


    instruction = f"{instruction} Write the summary in {language}."
    prompt = f"{instruction}\n\nDocument:\n{document}"



    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    def stream_text(stream):
        for event in stream:
            delta = event.choices[0].delta.content
            if delta:
                yield delta

    st.write_stream(stream_text(stream))
