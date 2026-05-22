from fastapi import APIRouter, status
from helper import *
from schemas import *
from utils import project_return
from models.jobs import Job
from response_models import *
from typing import Literal
from database import SessionLocal

router = APIRouter()


@router.get("/{job_id}", response_model=ProjectReturn[JobSchema])
def get_job(job_id: str):
    with SessionLocal.begin() as session:
        stmt = select(Job).where(Job.id == job_id)
        job = session.scalar(stmt)

    job_data = JobSchema.model_validate(job).model_dump(mode="json")

    return project_return(
        status_code=status.HTTP_200_OK,
        data=job_data
    )


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
