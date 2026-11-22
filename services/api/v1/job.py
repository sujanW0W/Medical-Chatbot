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

    return project_return(status_code=status.HTTP_200_OK, data=job_data)


@router.patch("/job-status-change/{job_id}/{job_status}")
def update_status(
    job_id: str, job_status: Literal["PROCESSING"] | Literal["CANCELLED"]
):
    update_job_status(job_id=job_id, job_status=job_status)

    return project_return(
        status_code=status.HTTP_200_OK,
        data={"message": "Job status updated successfully."},
    )


@router.post("/{job_id}/success")
def job_success(job_id: str, body: JobCompletionReport):
    ai_msg: Message = {"role": "assistant", "content": body.content}

    convo = add_conversation(body.session_id, ai_msg)
    update_job_success(job_id, convo.id)

    return project_return(
        status_code=status.HTTP_200_OK,
        data={"message": "Conversation inserted and job marked successful."},
    )


@router.post("/{job_id}/failure")
def job_failure(job_id: str, body: JobCompletionReport):
    ai_msg: Message = {"role": "assistant", "content": body.content}

    convo = add_conversation(body.session_id, ai_msg)
    update_job_failure(job_id, convo.id)

    return project_return(
        status_code=status.HTTP_200_OK,
        data={"message": "Conversation inserted and job marked failed."},
    )
