from pydantic import BaseModel
from datetime import datetime


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
