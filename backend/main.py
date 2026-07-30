"""FastAPI application entrypoint for lemmesearch backend."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.config import get_settings
from backend.db import VectorStore
from backend.embeddings import LocalEmbedder
from backend.router import router as api_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("lemmesearch")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for initializing database and embedding model.

    Args:
        app: FastAPI application instance.
    """
    settings = get_settings()
    logger.info("Initializing ChromaDB persistent storage at %s", settings.chroma_db_dir)
    app.state.vector_store = VectorStore(
        db_dir=settings.chroma_db_dir, collection_name=settings.collection_name
    )

    logger.info("Loading local sentence-transformer embedding model: %s", settings.default_embedding_model)
    app.state.embedder = LocalEmbedder(
        model_name=settings.default_embedding_model,
        cache_dir=settings.models_dir,
    )

    logger.info("Initialization complete. lemmesearch backend server ready.")
    yield
    logger.info("Shutting down lemmesearch backend server.")


def create_app() -> FastAPI:
    """FastAPI application factory.

    Returns:
        FastAPI: Configured FastAPI application instance.
    """
    app = FastAPI(
        title="lemmesearch Local RAG API",
        version="0.1.0",
        description="Local RAG backend with ChromaDB vector search and OpenRouter SSE streaming",
        lifespan=lifespan,
    )

    # CORS setup for web interface
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API router
    app.include_router(api_router)

    # Handle favicon.ico to prevent 404 logs
    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        return Response(status_code=204)

    # Root route serving index.html
    @app.get("/", include_in_schema=False)
    async def serve_index() -> FileResponse:
        index_path = Path("UI.html")
        if index_path.exists():
            return FileResponse(index_path, media_type="text/html")
        return FileResponse(Path.cwd() / "index.html", media_type="text/html")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=7777)
