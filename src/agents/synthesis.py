from langchain.chat_models.base import BaseChatModel
from .base import BaseAgent
from .state import AgentState
from src.prompts import system_prompt


class SynthesisAgent(BaseAgent):
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def run(self, state: AgentState):
        """Synthesize information from all available sources."""

        context_parts = []

        if "retrieved_docs" in state:
            retrieved_context = []
            for i, doc in enumerate(state["retrieved_docs"]):
                retrieved_context.append(
                    f"Retrieval Index: {i+1}\n\n"
                    f"Source: {doc.metadata['source']}\n\n"
                    f"Content: {doc.page_content}\n"
                )

            context_parts.append(
                "Here are the relevant documents from our medical knowledge base: " +
                "\n---\n".join(retrieved_context) +
                "Please use this verified medical information to generate a response."
            )

        if "web_results" in state:
            web_context = []
            for i, result in enumerate(state["web_results"]):
                web_context.append(
                    f"Web Result: {i+1}\n\n"
                    f"Title: {result['title']}\n\n"
                    f"Content: {result['snippet']}\n\n"
                    f"Source: {result['link']}"
                )

            context_parts.append(
                "Here are the Web Search Results: " +
                "\n---\n".join(web_context) +
                "Please use this information to generate a response."
            )

        # Generate final response using gathered context

        context = "\n\n".join(context_parts)

        messages = [
            *state["messages"],
            {
                "role": "system",
                "content": system_prompt.format(context=context)
            },
        ]

        response = self.llm.invoke(messages)

        ai_message = {
            "role": "assistant",
            "content": response.content
        }

        return {"messages": [*state["messages"], ai_message]}
