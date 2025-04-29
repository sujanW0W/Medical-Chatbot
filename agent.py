import os
from dotenv import load_dotenv

from typing import TypedDict, Annotated
from uuid import UUID

from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from src.helpers import download_embedding_model
from src.prompts import *

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

    ai_message = {
        "role": "assistant",
        "content": response.content
    }
    return {"messages": [*state["messages"], ai_message]}


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
