import os

from dotenv import load_dotenv

from typing import TypedDict, Annotated
from uuid import UUID

# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.prompts import ChatPromptTemplate

from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from src.helpers import download_embedding_model
from src.prompts import *

from pprint import pprint

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

load_dotenv()

embedding_model = download_embedding_model()
index_name = "medical-chatbot"


# Load Existing Index

doc_store = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding_model
)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

retriever = doc_store.as_retriever(
    search_type="similarity", search_kwargs={"k": 3})

memory_saver = MemorySaver()


class State(TypedDict):
    messages: Annotated[list, add_messages]


def retrieve(state: State):
    """Retrieve relevant documents for the latest message"""
    latest_message = state["messages"][-1].content
    docs = retriever.get_relevant_documents(latest_message)
    context = "\n".join([doc.page_content for doc in docs])

    # Add system message with context and prompt
    message = [
        {
            "role": "system",
            "content": system_prompt.format(context=context)
        },
        *state["messages"]
    ]

    return {"messages": message}


def generate_response(state: State):
    """Generate response using the LLM"""
    response = llm.invoke(state["messages"])
    return {"messages": [*state["messages"], response]}


graph_builder = StateGraph(State)

config = {"configurable": {"thread_id": UUID(int=0)}}

# Update graph with new nodes
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate_response)

# Update graph with new edges
graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile(checkpointer=memory_saver)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data['question']

    response = graph.invoke({
        "messages": [
            {"role": "user", "content": question}
        ],
    },
        config,
    )

    for message in response["messages"]:
        pprint(message.__dict__, indent=4)
        print("\n")

    return jsonify({'response': response['messages'][-1].content})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
