import os
from fastapi import APIRouter, status
from api.helper import *
from api.types import *
from api.utils import project_return
from api.database import SessionLocal
from api.models import *
from api.response_models import *

router = APIRouter()


@router.get("/")
def get_sessions():
    try:
        with SessionLocal() as session:
            result = session.scalars(select(Session)).all()
            sessions = [SessionSchema.model_validate(
                r).model_dump(mode='json') for r in result]

        return project_return(
            status_code=status.HTTP_200_OK,
            data=sessions
        )
    except Exception as e:
        print(e)
        return project_return(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error="Something went wrong"
        )


@router.get("/{session_id}/conversations")
def get_conversations(session_id: str):
    try:
        if not session_exists(session_id):
            return project_return(
                status_code=status.HTTP_404_NOT_FOUND,
                error="Session does not exist"
            )

        with SessionLocal() as session:
            stmt = select(Conversation).where(
                Conversation.session_id == session_id)
            result = session.scalars(stmt).all()
            conversations = [ConversationSchema.model_validate(
                r).model_dump(mode='json') for r in result]

        return project_return(
            status_code=status.HTTP_200_OK,
            data=conversations
        )

    except Exception as e:
        print(e)
        return project_return(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error="Something went wrong"
        )


@router.put("/{session_id}/rename")
def rename_session(session_id: str, query: Query):
    try:
        if not session_exists(session_id):
            return project_return(
                status_code=status.HTTP_404_NOT_FOUND,
                error="Session does not exist"
            )

        with SessionLocal.begin() as session:
            stmt = select(Session).where(Session.id == session_id)
            session_instance = session.scalar(stmt)

            session_instance.name = query.content

        return project_return(
            status_code=status.HTTP_200_OK,
            data="Rename Successful"
        )

    except Exception as e:
        print(e)
        return project_return(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error="Something went wrong"
        )


@router.delete("/{session_id}")
def delete_session(session_id: str):
    try:

        if not session_exists(session_id):
            return project_return(
                status_code=status.HTTP_404_NOT_FOUND,
                error="Session does not exist"
            )

        with SessionLocal.begin() as session:
            session_instance = session.scalar(
                select(Session).where(Session.id == session_id))

            session.delete(session_instance)

        return project_return(
            status_code=status.HTTP_200_OK,
            data="Session deleted successfully"
        )

    except Exception as e:
        print(e)
        return project_return(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error="Something went wrong"
        )
