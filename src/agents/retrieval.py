from langchain.schema import BaseRetriever
from .base import BaseAgent
from .state import AgentState
from sentence_transformers import SentenceTransformer, util
import torch


class RetrievalAgent(BaseAgent):
    """Retrieval Agent for the medical chatbot."""

    def __init__(self, retriever: BaseRetriever, raw_model: SentenceTransformer):
        self.retriever = retriever
        self.model = raw_model

    def _calculate_relevance(self, docs, query):
        """Calculate relevance score based on the retrieved documents."""
        if not docs:
            return 0.0

        # Encode query and documents
        query_embedding = self.model.encode(query)
        doc_embeddings = self.model.encode(
            [doc.page_content for doc in docs])

        # Calculate cosine similarity between query and each document
        similarities = util.cos_sim(query_embedding, doc_embeddings)[0]

        # Average the top 3 similarity scores
        top_k_similarities = torch.topk(
            similarities, k=min(3, len(similarities)))
        avg_similarity = float(torch.mean(top_k_similarities.values))

        return avg_similarity

    def run(self, state: AgentState):
        """Retrieval agent responsible for retrieving relevant documents."""

        print("## Running Retrieval Agent ##")

        query = state["messages"][-1].content
        docs = self.retriever.get_relevant_documents(query)
        relevance_score = self._calculate_relevance(docs, query)

        return {
            "messages": state["messages"],
            "retrieved_docs": docs,
            "relevance_score": relevance_score,
            "last_tool": "retriever"
        }
