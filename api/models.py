from sqlalchemy import ForeignKey, String, Text, DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
import uuid


class Base(DeclarativeBase):
    pass


class Session(Base):
    __tablename__ = "session"

    id: Mapped[str] = mapped_column(
        String(36), default=lambda: str(uuid.uuid4()), primary_key=True)
    name: Mapped[str] = mapped_column(String(256))

    conversations = relationship("Conversation", passive_deletes=True)

    created_at = mapped_column(DateTime, server_default=func.now())
    updated_at = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now())

    def __repr__(self) -> str:
        return f"Session(id={self.id}, name={self.name})"


class Conversation(Base):
    __tablename__ = "conversation"

    id: Mapped[str] = mapped_column(
        String(36), default=lambda: str(uuid.uuid4()), primary_key=True)
    role: Mapped[str] = mapped_column(String(128))
    content: Mapped[str] = mapped_column(Text)
    session_id: Mapped[str] = mapped_column(String(36), ForeignKey(
        "session.id", ondelete="CASCADE"), nullable=False)

    created_at = mapped_column(DateTime, server_default=func.now())
    updated_at = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now())

    def __repr__(self) -> str:
        return f"Conversation(id={self.id}, session_id={self.session_id})"
