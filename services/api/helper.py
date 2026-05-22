from sqlalchemy import select
from database import SessionLocal
from models.sessions import Session, Conversation
from models.jobs import Job, JobStatus
from schemas import *


def create_session():
    with SessionLocal.begin() as session:
        session_instance = Session(name="new session")
        session.add(session_instance)

    return session_instance


def generate_title(session_id: str, user_msg):
    try:
        #         prompt = f"""
        # Generate a short 3-6 word title for this conversation.

        # User: {user_msg}

        # Assistant: {ai_msg}

        # Return only the title.
        # """

        #         response = graph.invoke({
        #             "messages": [prompt]
        #         })

        #         title = response["messages"][-1].content

        title = ' '.join(user_msg["content"].split()[:10])

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


def create_job(conversation: Conversation):
    with SessionLocal.begin() as session:
        job = Job(conversation_id=conversation.id)
        session.add(job)

    return job


def update_job_status(job_id: str, job_status: str):
    try:
        with SessionLocal.begin() as session:
            stmt = select(Job).where(Job.id == job_id)
            job = session.scalar(stmt)

            job.status = JobStatus[job_status]

    except Exception as e:
        print(e)
        raise Exception(e)


def update_job_success(job_id: str, result_conversation_id: str):
    try:
        with SessionLocal.begin() as session:
            stmt = select(Job).where(Job.id == job_id)
            job = session.scalar(stmt)

            job.status = JobStatus["SUCCESS"]
            job.result_conversation_id = result_conversation_id

    except Exception as e:
        print(e)
        raise Exception(e)


def update_job_failure(job_id: str, failed_conversation_id: str):
    try:
        with SessionLocal.begin() as session:
            stmt = select(Job).where(Job.id == job_id)
            job = session.scalar(stmt)

            job.status = JobStatus["FAILED"]
            job.error_conversation_id = failed_conversation_id

    except Exception as e:
        print(e)
        raise Exception(e)
