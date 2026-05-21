import os
from celery import Celery


celery_app = Celery(
    "backend",
    broker=os.getenv("BROKER_URL")
)
