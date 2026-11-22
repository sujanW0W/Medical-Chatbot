import os
import logging
from celery_app import celery_app
import requests
from src.orchestrator import graph

logger = logging.getLogger(__name__)

BACKEND_URL = os.getenv("BACKEND_URL")


@celery_app.task(name="ask")
def invoke_llm(job_id, session_id, msg):
    logger.info("job %s: received, session=%s", job_id, session_id)
    try:
        requests.patch(f"{BACKEND_URL}/jobs/job-status-change/{job_id}/PROCESSING")

        response = graph.invoke({"messages": msg})

        logger.info("job %s: succeeded", job_id)
        requests.post(
            f"{BACKEND_URL}/jobs/{job_id}/success",
            json={
                "session_id": session_id,
                "content": response["messages"][-1].content,
            },
        )

    except Exception:
        logger.exception("job %s: failed", job_id)
        requests.post(
            f"{BACKEND_URL}/jobs/{job_id}/failure",
            json={
                "session_id": session_id,
                "content": "⚠️ Something went wrong while generating a response. Please try again.",
            },
        )
