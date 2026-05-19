from fastapi import APIRouter, status
from api.helper import *
from api.types import *
from api.utils import project_return
from src.orchestrator import graph
from api.models import *
from api.response_models import *

router = APIRouter()


@router.post("/ask")
def ask_new_session(query: Query):
    try:
        session = create_session()
        print(session)
        session_id = session.id

        user_msg: Message = {
            "role": "user",
            "content": query.content
        }
        add_conversation(session_id, user_msg)

        response = graph.invoke({
            "messages": [user_msg]
        })

        ai_msg: Message = {
            "role": "assistant",
            "content": response["messages"][-1].content
        }

        add_conversation(session_id, ai_msg)

        generate_title(session_id, user_msg, ai_msg)

        return project_return(
            status_code=status.HTTP_201_CREATED,
            data={
                "session_id": str(session_id),
                "messages": [
                    user_msg,
                    ai_msg
                ]
            }
        )

    except Exception as e:
        print(e)
        return project_return(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error="Something went wrong"
        )


@router.post("/ask/{session_id}")
def ask(session_id: str, query: Query):
    try:
        if not session_exists(session_id):
            return project_return(
                status_code=status.HTTP_404_NOT_FOUND,
                error="Session does not exist"
            )

        user_msg: Message = {
            "role": "user",
            "content": query.content
        }

        add_conversation(session_id, user_msg)

        with SessionLocal() as session:
            stmt = select(Conversation).where(
                Conversation.session_id == session_id)
            result = session.scalars(stmt).all()
            convos = [ConversationSchema.model_validate(
                r).model_dump(mode='json') for r in result]

        conversations = [{"role": c["role"], "content": c["content"]}
                         for c in convos]
        print(conversations)

        response = graph.invoke({
            "messages": conversations
        })

        ai_msg: Message = {
            "role": "assistant",
            "content": response["messages"][-1].content
        }
        add_conversation(session_id, ai_msg)

        return project_return(
            status_code=status.HTTP_201_CREATED,
            data={
                "session_id": str(session_id),
                "messages": [
                    user_msg,
                    ai_msg
                ]
            }
        )

    except Exception as e:
        print(e)
        return project_return(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error="Something went wrong"
        )
