from fastapi import APIRouter, status
from helper import *
from schemas import *
from utils import project_return
from celery_app import celery_app
from models.sessions import Conversation
from response_models import *
from typing import Literal


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
        convo = add_conversation(session_id, user_msg)

        # response = graph.invoke({
        # "messages": [user_msg]
        # })

        job = create_job(convo)

        celery_app.send_task(
            "ask",
            task_id=job.id,
            args=[job.id, session_id, [user_msg]]
        )

        # ai_msg: Message = {
        #     "role": "assistant",
        #     "content": response["messages"][-1].content
        # }

        # add_conversation(session_id, ai_msg)

        generate_title(session_id, user_msg)

        return project_return(
            status_code=status.HTTP_201_CREATED,
            data={
                # "session_id": str(session_id),
                # "messages": [
                #     user_msg,
                #     ai_msg
                # ],
                "job_id": str(job.id)
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

        new_convo = add_conversation(session_id, user_msg)

        with SessionLocal() as session:
            stmt = select(Conversation).where(
                Conversation.session_id == session_id)
            result = session.scalars(stmt).all()
            convos = [ConversationSchema.model_validate(
                r).model_dump(mode='json') for r in result]

        conversations = [{"role": c["role"], "content": c["content"]}
                         for c in convos]
        # print(conversations)

        # response = graph.invoke({
        #     "messages": conversations
        # })

        job = create_job(new_convo)

        celery_app.send_task(
            "ask",
            task_id=job.id,
            args=[job.id, session_id, conversations]
        )

        # ai_msg: Message = {
        #     "role": "assistant",
        #     "content": response["messages"][-1].content
        # }
        # add_conversation(session_id, ai_msg)

        return project_return(
            status_code=status.HTTP_201_CREATED,
            data={
                # "session_id": str(session_id),
                # "messages": [
                #     user_msg,
                #     ai_msg
                # ],
                "job_id": str(job.id)
            }
        )

    except Exception as e:
        print(e)
        return project_return(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error="Something went wrong"
        )


# APIs for LLM-worker

@router.patch('/job-status-change/{job_id}/{job_status}')
def update_status(job_id: str, job_status: Literal["PROCESSING"] | Literal["CANCELLED"]):
    update_job_status(
        job_id=job_id,
        job_status=job_status
    )

    return project_return(
        status_code=status.HTTP_200_OK,
        data={
            "message": "Job status updated successfully."
        }
    )


@router.post('/add-conversation/{session_id}/{job_id}/{job_status}')
def post_conversation(session_id: str, job_id: str, job_status: Literal["SUCCESS"] | Literal["FAILED"], query: Query):
    ai_msg: Message = {
        "role": "assistant",
        "content": query.content
    }

    if job_status == "SUCCESS":
        convo = add_conversation(session_id, ai_msg)
        update_job_success(job_id, convo.id)

    elif job_status == "FAILED":
        convo = add_conversation(session_id, ai_msg)
        update_job_failure(job_id, convo.id)

    else:
        return project_return(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error="Wrong job status"
        )

    return project_return(
        status_code=status.HTTP_200_OK,
        data={
            "message": "Conversation inserted and job status updated successfully."
        }
    )
