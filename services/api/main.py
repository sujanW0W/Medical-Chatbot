from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from limiter import limiter
from utils import project_return
from v1.api import router as api_v1_router

app = FastAPI()

app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    # exc.detail is the specific limit that was hit (e.g. "5 per 1 minute" or
    # "50 per 1 day") - tailor the message so the user knows which one, since
    # "try again in a moment" is misleading advice for the daily cap.
    if "day" in exc.detail:
        message = "You've reached today's message limit. Please try again tomorrow."
    else:
        message = "You're sending messages too fast. Please wait a moment and try again."

    response = project_return(status_code=429, error=message)
    return limiter._inject_headers(response, request.state.view_rate_limit)

origins = [
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=['*']
)

app.include_router(api_v1_router, prefix="/api/v1")
