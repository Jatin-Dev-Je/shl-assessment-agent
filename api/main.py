"""
api/main.py
"""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agent.orchestrator import get_orchestrator
from api.schemas import ChatRequest, ChatResponse, ErrorResponse, HealthResponse
from config import settings
from logging_config import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up SHL Assessment Recommender")
    get_orchestrator()
    from agent.retriever import _load_lancedb_table, _warm_cross_encoder_async

    _load_lancedb_table()
    _warm_cross_encoder_async()
    logger.info("Startup complete — LanceDB loaded; CrossEncoder warming in background")
    yield
    logger.info("Shutting down SHL Assessment Recommender")


app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational AI agent that recommends SHL assessments through natural dialogue.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    logger.info(
        "Request completed",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(elapsed * 1000, 2),
        },
    )
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status_code": exc.status_code},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception", extra={"path": request.url.path, "error": str(exc)})
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "status_code": 500},
    )


@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"model": ChatResponse},
        400: {"model": ErrorResponse},
        422: {"description": "Validation error"},
        500: {"model": ErrorResponse},
        504: {"description": "Agent timed out"},
    },
)
async def chat(request: ChatRequest) -> ChatResponse:
    messages = request.to_dict_list()
    logger.info(
        "Chat request received",
        extra={
            "turn_count": len(messages),
            "latest_message": messages[-1]["content"][:80] if messages else "",
        },
    )
    orchestrator = get_orchestrator()
    try:
        agent_response = await asyncio.wait_for(
            orchestrator.run(messages),
            timeout=settings.REQUEST_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Agent timed out after {settings.REQUEST_TIMEOUT}s",
        )
    except Exception as e:
        logger.error("Agent error", extra={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Agent encountered an error",
        )

    # Keep RECS_SENTINEL in the reply so the stateless client echoes it back
    # in the next turn's history. _clean_messages_for_llm strips it before
    # any LLM call, so neither Groq nor Gemini ever sees it.
    response = ChatResponse.from_agent_response(agent_response, strip_sentinel=False)
    logger.info(
        "Chat response sent",
        extra={
            "recommendations": len(response.recommendations),
            "end_of_conversation": response.end_of_conversation,
        },
    )
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,
        log_level="info",
    )