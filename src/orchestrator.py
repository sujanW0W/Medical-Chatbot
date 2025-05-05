from dotenv import load_dotenv

from uuid import UUID

from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.helpers import download_embedding_model
from src.prompts import *

from src.agents.state import AgentState
from src.agents.retrieval import RetrievalAgent
from src.agents.synthesis import SynthesisAgent
from src.agents.web_search import WebSearchAgent

load_dotenv()

langchain_embeddings, raw_model = download_embedding_model()
index_name = "medical-chatbot"


# Load Existing Index

doc_store = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=langchain_embeddings
)

retriever = doc_store.as_retriever(
    search_type="similarity", search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

memory_saver = MemorySaver()

# Initialize agents
retrieval_agent = RetrievalAgent(retriever, raw_model)
web_search_agent = WebSearchAgent()
synthesis_agent = SynthesisAgent(llm)
# tool_manager = ToolManager()

graph_builder = StateGraph(AgentState)

config = {"configurable": {"thread_id": UUID(int=0)}}

# Update graph with new nodes
graph_builder.add_node("retrieve", retrieval_agent.run)
graph_builder.add_node("web_search", web_search_agent.run)
graph_builder.add_node("synthesize", synthesis_agent.run)


# Define the routing function
def route_tools(state: AgentState):
    """Route query to appropriate search node based on agent's decision."""

    query = state["messages"][-1].content
    # Let LLM decide which tool to use
    messages = [
        {"role": "system", "content": "Decide which tool to use: 'retrieve', 'web_search'"},
        {"role": "user", "content": query},
    ]

    response = llm.invoke(messages)
    tool_choice = response.content.lower().strip()

    if tool_choice == "web_search":
        return "web_search"
    # elif tool_choice == "both":
    #     return ["retrieve", "web_search"]
    return "retrieve"


# Add conditional routing based on query type
graph_builder.add_conditional_edges(
    START,
    route_tools
)

# Update graph with new edges
graph_builder.add_edge("retrieve", "synthesize")
graph_builder.add_edge("web_search", "synthesize")
graph_builder.add_edge("synthesize", END)

graph = graph_builder.compile(checkpointer=memory_saver)
