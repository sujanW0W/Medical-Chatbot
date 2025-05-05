from typing import TypedDict, Annotated, List, Dict, Optional
from langchain.schema import Document
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    retrieved_docs: Optional[List[Document]]
    relevance_score: Optional[float]
    last_tool: Optional[str]
    web_results: Optional[List[Dict[str, str]]]
