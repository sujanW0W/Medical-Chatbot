from fastapi.responses import JSONResponse
from typing import Any


def project_return(status_code, data: Any | None = None, error: str | None = None):
    if error:
        return JSONResponse(
            status_code=status_code,
            content={
                "status": status_code,
                "data": None,
                "error": error
            }
        )

    return JSONResponse(
        status_code=status_code,
        content={
            "status": status_code,
            "data": data,
            "error": None
        }
    )
