from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.agents.agent import agent
from app.config.settings import settings
from app.schemas.chat import ChatRequest, ChatResponse, ToolsListResponse


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown events."""
    print(f"[INFO] Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"[INFO] Loaded Tools: {len(agent.tool_functions)} registered tools")
    if not settings.GEMINI_API_KEY:
        print("[WARNING] GEMINI_API_KEY is not configured in .env")
    yield
    print(f"[INFO] Shutting down {settings.APP_NAME}")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
# 🤖 AI Agent API

A modular, tool-calling AI agent backend powered by FastAPI and Google Gemini.

### ✨ Features
- **Autonomous Tool Execution**: Evaluates user prompts and executes relevant tools.
- **Calculator Tool**: Safe arithmetic and advanced math evaluation.
- **Date Tool**: Timezone-aware date calculations and differences.
- **Text Analyzer Tool**: Computes word counts, reading duration, and keyword frequencies.
- **Inspectable Tool Registry**: Discover available tools and parameter schemas at `/api/v1/tools`.
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def root() -> Dict[str, str]:
    """Root endpoint welcoming users and pointing to API documentation."""
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health",
        "tools": "/api/v1/tools",
    }


@app.get("/health", tags=["General"])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint to verify system and configuration status."""
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "model": settings.MODEL_NAME,
        "gemini_api_key_configured": bool(settings.GEMINI_API_KEY),
        "registered_tools": list(agent.tool_functions.keys()),
    }


@app.get(
    "/api/v1/tools",
    response_model=ToolsListResponse,
    tags=["Agent & Tools"],
    summary="List registered agent tools",
)
async def list_tools() -> ToolsListResponse:
    """Retrieve metadata and schema details for all tools accessible to the AI Agent."""
    tools_info = agent.get_registered_tools_info()
    return ToolsListResponse(
        total_tools=len(tools_info),
        tools=tools_info,
    )


@app.post(
    "/api/v1/chat",
    response_model=ChatResponse,
    tags=["Agent & Tools"],
    summary="Send prompt to AI Agent",
)
async def chat_with_agent(request: ChatRequest) -> ChatResponse:
    """
    Send a message to the AI agent.

    The agent will reason through the query, invoke any necessary tools (such as Calculator,
    Date Tool, or Text Analyzer), and return the final answer along with recorded tool calls.
    """
    try:
        response = await agent.run(
            message=request.message,
            history=request.history,
            system_prompt=request.system_prompt,
        )
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution failed: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
