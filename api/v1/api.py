from fastapi import APIRouter
from api.v1.sessions import router as session_router
from api.v1.chat import router as chat_router


router = APIRouter()


@router.get("/")
def root():
    return {"message": "Welcome to the Medical Chatbot v1 backend"}


router.include_router(session_router, prefix="/sessions", tags=["Sessions"])
router.include_router(chat_router, prefix="/chat", tags=["Chat"])
