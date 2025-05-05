from abc import ABC, abstractmethod
from .state import AgentState


class BaseAgent(ABC):
    """Base class for all agents."""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def run(self, state: AgentState):
        pass
