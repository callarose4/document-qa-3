import streamlit as st
from openai import OpenAI

st.title(" MY Lab 3 question answering chatbot")

openAI_model = st.sidebar.selectbox("Which Model?",  
                                    ["mini", "regular"])

if openAI_model == "mini":
    model_to_use = "gpt-4o-mini"
else: 
    model_to_use = "gpt-4o"

#create an OpenAI client
if 'client' not in st.session_state:
    openai_api_key = st.secrets["OPEN_API_KEY"]  # make sure this matches your secrets key
    st.session_state.client = OpenAI(api_key=openai_api_key)

if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    #st.chat_message(msg["role"]).markdown(msg["content"])
    #with st.chat_message(msg["role"]):
    # st.write(msg["content"])
    chat_msg = st.chat_message(msg["role"])
    chat_msg.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    client = st.session_state.client
    stream  = client.chat.completions.create(
        model=model_to_use,
        messages=st.session_state.messages,
        stream=True,
    )

full_response= ""

with st.chat_message("assistant"):
        response_box = st.empty()
        for event in stream:
            delta = event.choices[0].delta.content
            if delta:
                full_response += delta
                response_box.markdown(full_response)

st.session_state.messages.append({"role": "assistant", "content": full_response})    




