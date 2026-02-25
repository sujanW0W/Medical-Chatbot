from sqlalchemy import select
from src.orchestrator import graph
from api.sql import SessionLocal
from api.models import *
from api.types import *


def create_session():
    with SessionLocal.begin() as session:
        session_instance = Session(name="new session")
        session.add(session_instance)

    return session_instance


def generate_title(session_id: str, user_msg, ai_msg):
    try:
        prompt = f"""
Generate a short 3-6 word title for this conversation.

User: {user_msg}

Assistant: {ai_msg}

Return only the title.
"""

        response = graph.invoke({
            "messages": [prompt]
        })

        title = response["messages"][-1].content

        with SessionLocal.begin() as session:
            stmt = select(Session).where(Session.id == session_id)
            session_instance = session.scalar(stmt)

            session_instance.name = title

    except Exception as e:
        print(e)
        raise Exception(e)


def add_conversation(session_id: str, message: Message):
    with SessionLocal.begin() as session:
        convo = Conversation(**message, session_id=session_id)
        session.add(convo)

    return convo


def session_exists(session_id: str):
    with SessionLocal() as session:
        stmt = select(Session).where(Session.id == session_id)
        s = session.scalar(stmt)
        print(s)

        if not s:
            return False
        return True
