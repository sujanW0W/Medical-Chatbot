import os
from celery_app import celery_app
import requests
from src.orchestrator import graph

BACKEND_URL = os.getenv("BACKEND_URL")


@celery_app.task(name="ask")
def invoke_llm(job_id, session_id, msg):
    try:
        status = "PROCESSING"
        requests.patch(
            f"{BACKEND_URL}/jobs/job-status-change/{job_id}/{status}"
        )

        response = graph.invoke({
            "messages": msg
        })

        status = "SUCCESS"
        requests.post(
            f"{BACKEND_URL}/jobs/add-conversation/{session_id}/{job_id}/{status}",
            json={
                "content": response["messages"][-1].content
            }
        )

    except Exception as e:
        status = "FAILED"
        requests.post(
            f"{BACKEND_URL}/jobs/add-conversation/{session_id}/{job_id}/{status}",
            json={
                "content": str(e)
            }
        )
