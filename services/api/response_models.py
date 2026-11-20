from pydantic import BaseModel
from typing import Generic, TypeVar, Optional
from models.jobs import JobStatus
from datetime import datetime


T = TypeVar('T')


class ProjectReturn(BaseModel, Generic[T]):
    status: str
    data: Optional[T] = None
    error: Optional[T] = None


class SessionSchema(BaseModel):
    id: str
    name: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ConversationSchema(BaseModel):
    id: str
    role: str
    content: str
    session_id: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class JobSchema(BaseModel):
    id: str
    conversation_id: str
    status: JobStatus
    result_conversation_id: Optional[str]
    error_conversation_id: Optional[str]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
