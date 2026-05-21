from dotenv import load_dotenv
from src.tasks import celery_app

_ = load_dotenv()

if __name__ == "__main__":
    celery_app.worker_main([
        "worker",
        "--loglevel=info"
    ])
