from pydantic import BaseModel
from typing import Literal


class Message(BaseModel):
    role: Literal["user"] | Literal["assistant"]
    content: str


class Query(BaseModel):
    content: str
