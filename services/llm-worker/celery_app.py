import os
from celery import Celery


celery_app = Celery(
    "llm-worker",
    broker=os.getenv("BROKER_URL")
)
