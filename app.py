import streamlit as st
from src.orchestrator import graph, config

st.set_page_config(page_title="Medical Chatbot",
                   page_icon="🤖", layout="centered", )
st.title("Medical Chatbot 🤖🩺")
st.info("This is a simple chatbot that can answer questions about medical conditions, treatments, and medications.")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! How can I assist you today?"}
    ]

# Display chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Respond to user input
if prompt := st.chat_input("Ask a question:"):

    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # Add user message to the session state
    st.session_state["messages"].append(
        {"role": "user", "content": prompt})

    with st.spinner("Thinking...", show_time=True):
        response = graph.invoke(
            {"messages": st.session_state["messages"]}, config
        )

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response["messages"][-1].content)

    # Add assistant response to the session state
    st.session_state["messages"].append(
        {"role": "assistant", "content": response["messages"][-1].content})
