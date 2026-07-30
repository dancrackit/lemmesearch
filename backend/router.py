"""FastAPI API routes module for document ingestion, CRUD, and RAG chat."""

import asyncio
import json
import logging
import math
import os
import re
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import APIRouter, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.config import (
    get_credential_dir,
    get_saved_custom_models_from_file,
    get_settings,
    save_models_config,
    save_openrouter_api_key,
)
from backend.llm import build_rag_prompt, generate_openrouter_stream
from backend.parser import chunk_text, extract_text_from_file
from backend.prompts import load_persona_prompts, save_persona_prompts
from backend.retrieval import MultiStageRetriever

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["RAG API"])


class ChatQueryRequest(BaseModel):
    """Schema for chat query POST payload."""

    prompt: str = Field(..., description="User prompt or question")
    history: list[dict[str, str]] = Field(default_factory=list, description="Prior conversation history turns")
    model: str | None = Field(default=None, description="OpenRouter model tag")
    api_key: str | None = Field(default=None, description="OpenRouter API key override")
    top_k: int = Field(default=4, ge=1, le=10, description="Top-K vector context matches")
    search_enabled: bool = Field(default=True, description="If True, query ChromaDB brain. If False, use pretrained LLM knowledge.")
    reasoning_enabled: bool = Field(default=False, description="If True, send OpenRouter reasoning request.")
    reasoning_effort: str = Field(default="low", description="OpenRouter reasoning effort level: low, medium, high, etc.")
    system_prompt: str | None = Field(default=None, description="Custom system prompt string selected by user.")


class UpdatePromptsRequest(BaseModel):
    """Schema for updating system prompts."""

    prompts: dict[str, str] = Field(..., description="Map of prompt keys to prompt strings")


class UpdateHistoryRequest(BaseModel):
    """Schema for saving chat history."""

    sessions: list[dict[str, Any]] = Field(..., description="List of chat session objects")


class UpdateCredentialsRequest(BaseModel):
    """Schema for updating credentials & model configuration."""

    api_key: str | None = Field(default=None, description="OpenRouter API key")
    model: str | None = Field(default=None, description="OpenRouter model tag")
    embedding_model: str | None = Field(default=None, description="Embedding model name")
    prompts: dict[str, str] | None = Field(default=None, description="Persona prompts dictionary")




def format_file_size(size_bytes: int) -> str:
    """Format size in bytes to human-readable string.

    Args:
        size_bytes: Raw file size in bytes.

    Returns:
        str: Human-readable file size string (e.g. '1.5 MB').
    """
    if size_bytes == 0:
        return "0 Bytes"
    size_name = ("Bytes", "KB", "MB", "GB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"


@router.post("/ingest")
async def ingest_document(
    request: Request,
    file: UploadFile = File(...),
    chunk_size: int = Form(default=512),
    chunk_overlap: int = Form(default=50),
) -> StreamingResponse:
    """Upload and ingest a document file with live SSE progress streaming.

    Args:
        request: FastAPI Request instance.
        file: Multipart uploaded document file.
        chunk_size: Token/character chunk size setting.
        chunk_overlap: Token/character chunk overlap setting.

    Returns:
        StreamingResponse: SSE stream yielding progress frames.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    if chunk_overlap >= chunk_size:
        raise HTTPException(status_code=400, detail="Chunk overlap must be strictly less than chunk size.")

    vector_store = request.app.state.vector_store
    embedder = request.app.state.embedder

    content = await file.read()
    filename = file.filename
    file_size_str = format_file_size(len(content))

    async def progress_generator() -> AsyncGenerator[str, None]:
        temp_dir = Path("scratch/uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        doc_id = f"doc-{uuid.uuid4().hex[:8]}"
        temp_path = temp_dir / f"{doc_id}_{filename}"

        try:
            # Step 1: Upload / Save
            yield f"data: {json.dumps({'stage': 'upload', 'pct': 5, 'message': f'Saving {filename} ({file_size_str})...'})}\n\n"
            temp_path.write_bytes(content)
            await asyncio.sleep(0.01)

            # Step 2: Extract text
            yield f"data: {json.dumps({'stage': 'parsing', 'pct': 15, 'message': 'Extracting text content from document...'})}\n\n"
            logger.info("Extracting text from %s (%s)", filename, file_size_str)
            extracted_text = extract_text_from_file(temp_path)
            await asyncio.sleep(0.01)

            # Step 3: Chunking
            yield f"data: {json.dumps({'stage': 'chunking', 'pct': 25, 'message': 'Splitting text into character chunks...'})}\n\n"
            chunks = chunk_text(
                text=extracted_text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                filename=filename,
            )

            if not chunks:
                err_msg = f"Document '{filename}' contains no parseable text content."
                yield f"data: {json.dumps({'stage': 'error', 'pct': 0, 'message': err_msg})}\n\n"
                yield "data: [DONE]\n\n"
                return

            total_chunks = len(chunks)
            logger.info("Extracted %d characters into %d chunks for %s", len(extracted_text), total_chunks, filename)
            yield f"data: {json.dumps({'stage': 'chunked', 'pct': 30, 'total_chunks': total_chunks, 'message': f'Extracted text. Created {total_chunks} chunks.'})}\n\n"

            # Step 4: Batch embedding with progress queue
            queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def on_progress(processed: int, total: int) -> None:
                pct = 30 + int((processed / total) * 60)
                msg = f"Embedding chunks: {processed}/{total} ({pct}%)"
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {
                        "stage": "embedding",
                        "pct": pct,
                        "processed": processed,
                        "total": total,
                        "message": msg,
                    },
                )

            def run_ingest_blocking() -> int:
                return vector_store.ingest_document(
                    doc_id=doc_id,
                    filename=filename,
                    file_size_str=file_size_str,
                    chunks=chunks,
                    embedder=embedder,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    progress_callback=on_progress,
                )

            ingest_task = loop.run_in_executor(None, run_ingest_blocking)

            while not ingest_task.done() or not queue.empty():
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=0.1)
                    yield f"data: {json.dumps(item)}\n\n"
                except asyncio.TimeoutError:
                    await asyncio.sleep(0.02)

            inserted_count = await ingest_task

            # Step 5: Complete
            yield f"data: {json.dumps({'stage': 'complete', 'pct': 100, 'doc_id': doc_id, 'filename': filename, 'file_size': file_size_str, 'chunks': inserted_count, 'chunk_size': chunk_size, 'chunk_overlap': chunk_overlap, 'message': f'Successfully ingested {filename} ({inserted_count} chunks)'})}\n\n"

        except Exception as e:
            logger.error("Error during SSE document ingestion for %s: %s", filename, e)
            yield f"data: {json.dumps({'stage': 'error', 'pct': 0, 'message': f'Failed to ingest document: {e}'})}\n\n"
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError as cleanup_err:
                logger.warning("Could not remove temp file %s: %s", temp_path, cleanup_err)

        yield "data: [DONE]\n\n"

    return StreamingResponse(progress_generator(), media_type="text/event-stream")


@router.get("/documents")
async def list_documents(request: Request) -> dict[str, Any]:
    """List all ingested documents in the vector database.

    Args:
        request: FastAPI Request instance.

    Returns:
        dict[str, Any]: Object containing documents list and total count.
    """
    vector_store = request.app.state.vector_store
    docs = vector_store.list_documents()
    return {"documents": docs, "total": len(docs)}


@router.delete("/documents")
async def purge_all_documents(request: Request) -> dict[str, Any]:
    """Purge all documents and vector chunks from ChromaDB.

    Args:
        request: FastAPI Request instance.

    Returns:
        dict[str, Any]: Purge status summary object.
    """
    vector_store = request.app.state.vector_store
    deleted_count = vector_store.purge_all()
    logger.info("Purged all documents (%d vector chunks deleted) from ChromaDB.", deleted_count)
    return {
        "status": "success",
        "message": "All documents purged from ChromaDB",
        "deleted_chunks": deleted_count,
    }


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, request: Request) -> dict[str, Any]:
    """Remove a document and all its vector chunks from ChromaDB.

    Args:
        doc_id: Document unique identifier string.
        request: FastAPI Request instance.

    Returns:
        dict[str, Any]: Deletion result summary object.
    """
    vector_store = request.app.state.vector_store
    deleted_count = vector_store.delete_document(doc_id=doc_id)
    logger.info("Purged document '%s' (%d vector chunks deleted) from ChromaDB.", doc_id, deleted_count)
    return {
        "status": "success",
        "doc_id": doc_id,
        "deleted_chunks": deleted_count,
    }


class UpdateCredentialsRequest(BaseModel):
    """Schema for updating credentials & model configuration."""

    api_key: str | None = Field(default=None, description="OpenRouter API key")
    model: str | None = Field(default=None, description="OpenRouter model tag")
    embedding_model: str | None = Field(default=None, description="Embedding model name")
    saved_custom_models: list[Any] | None = Field(default=None, description="List of saved custom model tags")
    prompts: dict[str, str] | None = Field(default=None, description="Persona prompts dictionary")


@router.get("/prompts")
async def get_system_prompts() -> dict[str, Any]:
    """Retrieve system persona prompts loaded directly from the credential folder.

    Returns:
        dict[str, Any]: Dictionary containing system persona prompt configurations.
    """
    return {"prompts": load_persona_prompts()}


@router.post("/prompts")
async def update_system_prompts(body: UpdatePromptsRequest) -> dict[str, Any]:
    """Update system persona prompts and save into credential/system_prompts.md.

    Args:
        body: UpdatePromptsRequest containing map of prompt keys to prompt strings.

    Returns:
        dict[str, Any]: Status object.
    """
    save_persona_prompts(body.prompts)
    return {"status": "success", "message": "System prompts updated and saved to credential/system_prompts.md"}


def get_history_dir() -> Path:
    """Get the path to the chat_history directory inside credential/."""
    history_dir = get_credential_dir() / "chat_history"
    history_dir.mkdir(parents=True, exist_ok=True)
    return history_dir


def sanitize_filename(name: str) -> str:
    """Sanitize string to be safe for use as a filename on Windows/Linux."""
    name = name.replace('\n', ' ').replace('\r', ' ')
    name = re.sub(r'[\\/*?:"<>|]', '', name)
    name = name.strip().strip('.')
    if not name:
        name = "Untitled Chat"
    return name[:100]


def save_session_to_markdown(session: dict, directory: Path) -> None:
    """Save a chat session as a markdown file, keeping only user and ai messages."""
    session_id = session.get("id")
    if not session_id:
        return
    title = session.get("title", "")
    timestamp = session.get("timestamp", "")
    
    md_lines = []
    md_lines.append("---")
    md_lines.append(f"id: {session_id}")
    md_lines.append(f"title: {title}")
    md_lines.append(f"timestamp: {timestamp}")
    md_lines.append("---")
    md_lines.append("")
    
    for msg in session.get("messages", []):
        role = msg.get("role")
        content = msg.get("content", "").strip()
        if role == "user":
            md_lines.append("### User")
            md_lines.append(content)
            md_lines.append("")
        elif role == "ai":
            md_lines.append("### AI")
            md_lines.append(content)
            md_lines.append("")
            
    # Find existing file for this session ID to handle renames
    existing_file = None
    if directory.exists():
        for f in directory.glob("*.md"):
            try:
                content = f.read_text(encoding="utf-8")
                if f"id: {session_id}" in content:
                    existing_file = f
                    break
            except Exception:
                pass
                
    # Generate the target filename based on the current title
    safe_title = sanitize_filename(title)
    target_name = f"{safe_title}.md"
    
    # Handle duplicate filename collision for different sessions
    if (directory / target_name).exists():
        try:
            target_content = (directory / target_name).read_text(encoding="utf-8")
            if f"id: {session_id}" not in target_content:
                suffix = session_id[-4:] if len(session_id) >= 4 else "1"
                target_name = f"{safe_title} ({suffix}).md"
        except Exception:
            pass
            
    target_path = directory / target_name
    
    # If old file has a different name, delete it first
    if existing_file and existing_file != target_path:
        try:
            existing_file.unlink()
        except OSError:
            pass
            
    target_path.write_text("\n".join(md_lines), encoding="utf-8")


def parse_markdown_to_session(file_path: Path) -> dict | None:
    """Parse a markdown file back into a chat session dictionary."""
    try:
        content = file_path.read_text(encoding="utf-8")
        # Parse frontmatter
        frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
        if not frontmatter_match:
            return None
            
        fm_text = frontmatter_match.group(1)
        body_text = frontmatter_match.group(2)
        
        metadata = {}
        for line in fm_text.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                metadata[k.strip()] = v.strip()
                
        session_id = metadata.get("id", file_path.stem)
        title = metadata.get("title", "Untitled Chat")
        timestamp = metadata.get("timestamp", "")
        
        messages = []
        current_role = None
        current_content = []
        
        for line in body_text.splitlines():
            line_strip = line.strip()
            if line_strip == "### User":
                if current_role and current_content:
                    messages.append({
                        "role": current_role,
                        "content": "\n".join(current_content).strip()
                    })
                current_role = "user"
                current_content = []
            elif line_strip == "### AI":
                if current_role and current_content:
                    messages.append({
                        "role": current_role,
                        "content": "\n".join(current_content).strip()
                    })
                current_role = "ai"
                current_content = []
            else:
                if current_role is not None:
                    current_content.append(line)
                    
        if current_role and current_content:
            messages.append({
                "role": current_role,
                "content": "\n".join(current_content).strip()
            })
            
        return {
            "id": session_id,
            "title": title,
            "timestamp": timestamp,
            "messages": messages
        }
    except Exception as e:
        logger.warning("Failed to parse markdown chat history file %s: %s", file_path, e)
        return None


@router.get("/history")
async def get_chat_history() -> dict[str, Any]:
    """Retrieve chat history from credential/chat_history/ as separate markdown files."""
    directory = get_history_dir()
    
    # Auto-migration of legacy sessions.json if it exists
    legacy_file = directory / "sessions.json"
    if legacy_file.exists():
        try:
            content = legacy_file.read_text(encoding="utf-8")
            legacy_sessions = json.loads(content)
            for s in legacy_sessions:
                save_session_to_markdown(s, directory)
            legacy_file.unlink()
        except Exception as e:
            logger.warning("Failed to migrate legacy sessions.json: %s", e)
            
    sessions = []
    if directory.exists():
        for f in directory.glob("*.md"):
            session = parse_markdown_to_session(f)
            if session:
                sessions.append(session)
                
    # Sort sessions newest first (descending by timestamp in ID)
    def get_sort_key(s: dict) -> float:
        session_id = s.get("id", "")
        if session_id.startswith("chat-"):
            try:
                return float(session_id.split("-")[1])
            except (IndexError, ValueError):
                pass
        return 0.0

    sessions.sort(key=get_sort_key, reverse=True)
    return {"sessions": sessions}


@router.post("/history")
async def update_chat_history(body: UpdateHistoryRequest) -> dict[str, Any]:
    """Save/update chat history in credential/chat_history/ as separate markdown files."""
    directory = get_history_dir()
    try:
        active_ids = set()
        for session in body.sessions:
            session_id = session.get("id")
            if session_id:
                active_ids.add(session_id)
                save_session_to_markdown(session, directory)
                
        # Clean up any markdown files that are no longer active
        if directory.exists():
            for f in directory.glob("*.md"):
                session = parse_markdown_to_session(f)
                if session:
                    f_id = session.get("id")
                    if f_id and f_id not in active_ids:
                        try:
                            f.unlink()
                        except OSError:
                            pass
                        
        return {"status": "success", "message": "Chat history saved successfully."}
    except OSError as e:
        logger.error("Failed to write chat history to files: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to save chat history: {e}")


@router.get("/credentials")
async def get_credentials() -> dict[str, Any]:
    """Get active credential configurations and models settings.

    Returns:
        dict[str, Any]: Current API key status, model tags, and system prompts.
    """
    settings = get_settings()
    api_key = settings.openrouter_api_key
    masked_key = (api_key[:8] + "..." + api_key[-4:]) if len(api_key) > 12 else ("Set" if api_key else "Not Set")

    source = "Not Set"
    if api_key:
        if os.getenv("OPENROUTER_API_KEY"):
            source = "Environment Variable (OPENROUTER_API_KEY)"
        else:
            source = "credential/openrouter_api_key.md"

    return {
        "api_key": api_key,
        "api_key_masked": masked_key,
        "api_key_source": source,
        "openrouter_model": settings.default_openrouter_model,
        "saved_custom_models": get_saved_custom_models_from_file(),
        "embedding_model": settings.default_embedding_model,
        "prompts": load_persona_prompts(),
    }


@router.post("/credentials")
async def update_credentials(body: UpdateCredentialsRequest) -> dict[str, Any]:
    """Update OpenRouter API key, models, or prompts and sync directly to credential files.

    Args:
        body: UpdateCredentialsRequest payload.

    Returns:
        dict[str, Any]: Update result object.
    """
    if body.api_key:
        save_openrouter_api_key(body.api_key)

    if body.model or body.embedding_model or body.saved_custom_models is not None:
        save_models_config(
            openrouter_model=body.model,
            embedding_model=body.embedding_model,
            saved_custom_models=body.saved_custom_models,
        )

    if body.prompts:
        save_persona_prompts(body.prompts)

    return {"status": "success", "message": "Credentials and configurations saved to credential folder."}


@router.post("/chat")
async def chat_stream(
    request: Request,
    body: ChatQueryRequest,
    x_openrouter_key: str | None = Header(default=None, alias="X-OpenRouter-Key"),
) -> StreamingResponse:
    """Stream RAG response using Query Analysis Agent, Multi-Stage Retrieval Pipeline, and SSE.

    Args:
        request: FastAPI Request instance.
        body: ChatQueryRequest payload.
        x_openrouter_key: Optional OpenRouter API key header.

    Returns:
        StreamingResponse: SSE stream yielding reasoning trace and response tokens.
    """
    vector_store = request.app.state.vector_store
    embedder = request.app.state.embedder

    api_key = (body.api_key or "").strip() or (x_openrouter_key or "").strip()
    if api_key:
        save_openrouter_api_key(api_key)

    if body.model:
        save_models_config(openrouter_model=body.model)

    retrieved_chunks = []
    reasoning_steps = []

    if body.search_enabled:
        retriever = MultiStageRetriever(vector_store=vector_store, embedder=embedder)
        retrieved_chunks, pipeline_steps = retriever.retrieve(
            user_query=body.prompt,
            history=body.history,
            top_k=body.top_k,
        )
        reasoning_steps = pipeline_steps

    system_prompt, user_prompt = build_rag_prompt(
        user_query=body.prompt,
        context_chunks=retrieved_chunks,
        search_enabled=body.search_enabled,
        custom_system_prompt=body.system_prompt,
    )

    generator = generate_openrouter_stream(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        context_chunks=retrieved_chunks,
        history=body.history,
        model_tag=body.model,
        api_key=api_key,
        search_enabled=body.search_enabled,
        reasoning_steps=reasoning_steps,
        reasoning_enabled=body.reasoning_enabled,
        reasoning_effort=body.reasoning_effort,
    )

    return StreamingResponse(generator, media_type="text/event-stream")

