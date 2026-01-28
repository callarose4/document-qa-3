import streamlit as st
from openai import OpenAI

# Show title and description.
st.title("Lab 2")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.secrets.OPEN_API_KEY



client = OpenAI(api_key=openai_api_key)
st.success("API key validated")
    
    # Let the user upload a file via `st.file_uploader`.
uploaded_file = st.file_uploader(
        "Upload a document (.txt or .md)", type=("txt", "md")
    )

    # Ask the user for a question via `st.text_area`.
question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

if uploaded_file and question:

        # Process the uploaded file and question.
        document = uploaded_file.read().decode()
        messages = [
            {
                "role": "user",
                "content": f"Here's a document: {document} \n\n---\n\n {question}",
            }
        ]

        # Generate an answer using the OpenAI API.
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True,
        )

        def stream_text ():
            for event in stream:
                  delta = event.choices[0].delta.content
                  if delta: 
                        yield delta

        # Stream the response to the app using `st.write_stream`.
st.write_stream(stream_text())
