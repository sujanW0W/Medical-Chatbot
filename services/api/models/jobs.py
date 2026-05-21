from database import Base
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, DateTime, func, Enum, ForeignKey, Text
import uuid
import enum


class JobStatus(enum.Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    FAILED = "failed"
    SUCCESS = "success"
    CANCELLED = "cancelled"


class Job(Base):
    __tablename__ = "job"

    id: Mapped[str] = mapped_column(
        String(36), default=lambda: str(uuid.uuid4()), primary_key=True,
    )
    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversation.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus), default=JobStatus.QUEUED, server_default='QUEUED'
    )
    result_conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversation.id", ondelete="CASCADE"), nullable=True
    )
    error_conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversation.id", ondelete="CASCADE"), nullable=True
    )
    created_at = mapped_column(DateTime, server_default=func.now())
    updated_at = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now())

    def __repr__(self) -> str:
        return f"Job(id={self.id}, conversation_id={self.conversation_id}, status={self.status})"
