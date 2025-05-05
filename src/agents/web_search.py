from typing import List, Dict, Any
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.tools import Tool
from .base import BaseAgent
from .state import AgentState


class WebSearchAgent(BaseAgent):
    """Web Search Agent for performing web searching using DuckDuckGo."""

    def __init__(self):
        self.search_engine = DuckDuckGoSearchAPIWrapper()

    def search(self, query: str, num_results: int = 3):
        """Perform a web search and return the top results."""
        try:
            results = self.search_engine.results(query, num_results)

            if not results:
                return [{
                    "title": "No Results",
                    "snippet": "No relevant web results found.",
                    "link": ""
                }]

            return [{
                "title": result["title"],
                "snippet": result["snippet"],
                "link": result["link"]
            } for result in results]

        except Exception as e:
            print(f"Error during web search: {e}")
            return [{
                "title": "Error",
                "snippet": str(e),
                "link": ""
            }]

    def run(self, state: AgentState):
        """Run the web search agent."""

        print("## Running Web Search Agent ##")

        query = state["messages"][-1].content
        results = self.search(query)

        return {
            "messages": state["messages"],
            "web_results": results,
            "last_tool": "web_search",
        }
