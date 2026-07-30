"""OpenRouter LLM client module with SSE streaming generator and Query Analysis Agent."""

import json
import logging
from typing import Any, AsyncGenerator, Sequence

import httpx

from backend.config import get_settings

logger = logging.getLogger(__name__)

SYSTEM_RAG_PROMPT = """You are an expert AI Assistant and Analyst.

Answer the user's question directly by synthesizing the provided document context into a clear, original, and natural explanation.

RULES:
1. NO VERBATIM COPY-PASTING: Rephrase details in your own clear words.
2. CONVERSATIONAL MEMORY & REASONING: Maintain full continuity with prior conversation history. Reason through the user's prompt directly, and if necessary ask a concise clarification question.
3. ACCURACY: Include relevant facts, metrics, and data from the provided document context accurately.
4. IF UNANSWERED: If the document context does not contain relevant details, answer based on conversational context or state that information was not found in the indexed documents.
"""


def get_active_system_rag_prompt() -> str:
    """Load default system RAG prompt, checking credential/system_prompts.md first."""
    try:
        from backend.prompts import get_credential_dirs, parse_system_prompts_markdown
        for cred_dir in get_credential_dirs():
            sys_file = cred_dir / "system_prompts.md"
            if sys_file.exists():
                parsed = parse_system_prompts_markdown(sys_file.read_text(encoding="utf-8"))
                if "system_rag" in parsed and parsed["system_rag"]:
                    return parsed["system_rag"]
    except Exception:
        pass
    return SYSTEM_RAG_PROMPT


def build_rag_prompt(
    user_query: str,
    context_chunks: Sequence[dict[str, Any]],
    search_enabled: bool = True,
    custom_system_prompt: str | None = None,
) -> tuple[str, str]:
    """Format system prompt and user context prompt based on search state.

    Args:
        user_query: User chat question.
        context_chunks: List of retrieved context chunk dicts.
        search_enabled: Whether vector search RAG mode is active.
        custom_system_prompt: Optional user specified system prompt override.

    Returns:
        tuple[str, str]: Tuple of (system_prompt, user_prompt_with_context).
    """
    persona = (custom_system_prompt or "").strip()
    rag_rules = get_active_system_rag_prompt() if search_enabled else ""

    if persona and rag_rules and persona != rag_rules:
        base_sys_prompt = f"{persona}\n\n### RETRIEVAL GUIDELINES:\n{rag_rules}"
    elif persona:
        base_sys_prompt = persona
    else:
        base_sys_prompt = rag_rules or "You are a helpful AI assistant."

    if not search_enabled:
        return (base_sys_prompt, user_query)

    if not context_chunks:
        user_prompt = (
            f"No document context was found in the database.\n\nUSER QUESTION: {user_query}"
        )
        return (base_sys_prompt, user_prompt)

    context_str_parts: list[str] = []
    for idx, chunk in enumerate(context_chunks, 1):
        meta = chunk.get("metadata", {})
        filename = meta.get("filename", "Document")
        chunk_idx = meta.get("chunk_index", 0)
        text = chunk.get("text", "").strip()
        context_str_parts.append(
            f"[Source {idx}: {filename} (Chunk {chunk_idx})]\n{text}"
        )

    context_block = "\n\n".join(context_str_parts)

    user_prompt = (
        f"### RETRIEVED DOCUMENT CONTEXT:\n{context_block}\n\n"
        f"### USER QUESTION:\n{user_query}"
    )
    return (base_sys_prompt, user_prompt)



async def generate_openrouter_stream(
    system_prompt: str,
    user_prompt: str,
    context_chunks: Sequence[dict[str, Any]],
    history: list[dict[str, str]] | None = None,
    model_tag: str | None = None,
    api_key: str | None = None,
    search_enabled: bool = True,
    reasoning_steps: list[str] | None = None,
    reasoning_enabled: bool = False,
    reasoning_effort: str = "low",
) -> AsyncGenerator[str, None]:
    """Generate Server-Sent Events (SSE) stream for OpenRouter chat completions with dedicated system role, multi-turn history, and reasoning tokens.

    Args:
        system_prompt: System role prompt string.
        user_prompt: User role prompt string with context.
        context_chunks: Retrieved context snippets for citation metadata.
        history: Prior conversation history turns list.
        model_tag: OpenRouter model identifier string.
        api_key: User provided or fallback OpenRouter API key.
        search_enabled: Whether search RAG mode was enabled.
        reasoning_steps: Step-by-step pipeline execution reasoning trace strings.
        reasoning_enabled: Whether reasoning mode is enabled.
        reasoning_effort: OpenRouter reasoning effort ("low", "medium", "high", etc.).

    Yields:
        str: Formatted SSE payload strings (`data: {...}\n\n`).
    """
    settings = get_settings()

    active_api_key = (api_key or "").strip() or settings.openrouter_api_key
    active_model = (model_tag or "").strip() or settings.default_openrouter_model

    # Yield initial citation metadata and reasoning trace packet
    citations: list[dict[str, str]] = []
    if search_enabled and context_chunks:
        for chunk in context_chunks:
            meta = chunk.get("metadata", {})
            snippet_source = chunk.get("match_text", chunk.get("text", ""))
            citations.append(
                {
                    "filename": meta.get("filename", "Doc"),
                    "chunk_index": str(meta.get("chunk_index", 0)),
                    "snippet": snippet_source[:120] + "...",
                }
            )


    steps = reasoning_steps or []
    if not steps:
        if search_enabled:
            steps = [
                f"Searched ChromaDB vector database (found {len(context_chunks)} matching chunks)",
                "Applied semantic prompt augmentation",
            ]
        else:
            steps = ["Search disabled — relying on pre-trained LLM knowledge"]

    meta_packet = {
        "event": "meta",
        "model": active_model,
        "search_enabled": search_enabled,
        "chunk_count": len(context_chunks) if search_enabled else 0,
        "reasoning_steps": steps,
        "citations": citations,
        "reasoning_enabled": reasoning_enabled,
        "reasoning_effort": reasoning_effort,
    }
    yield f"data: {json.dumps(meta_packet)}\n\n"

def format_openrouter_error(status_code: int, raw_body_bytes: bytes) -> str:
    """Format OpenRouter API non-200 errors into clear, direct, indicating diagnostic messages.

    Args:
        status_code: HTTP response status code integer.
        raw_body_bytes: Raw binary body returned by OpenRouter API.

    Returns:
        str: Direct, indicating human-readable error message with resolution steps.
    """
    raw_str = raw_body_bytes.decode("utf-8", errors="ignore").strip()
    clean_detail = ""

    try:
        data = json.loads(raw_str)
        if isinstance(data, dict):
            err_obj = data.get("error")
            if isinstance(err_obj, dict):
                clean_detail = str(err_obj.get("message", "") or "").strip()
            elif isinstance(err_obj, str):
                clean_detail = err_obj.strip()
            elif "message" in data:
                clean_detail = str(data["message"]).strip()
    except Exception:
        clean_detail = raw_str

    if status_code == 401:
        cause_desc = clean_detail or "The OpenRouter API Key is missing, invalid, or expired."
        if "user not found" in clean_detail.lower():
            cause_desc = "The OpenRouter API Key provided is invalid, mistyped, or has been revoked/deleted on OpenRouter ('User not found')."
        return (
            "❌ **OpenRouter API Key Error (401 Unauthorized)**\n\n"
            f"**Cause**: {cause_desc}\n\n"
            "**Action Required**: Click the **Settings** gear icon (bottom left), enter a valid active OpenRouter API Key (`sk-or-v1-...`), and click **Save**."
        )

    if status_code == 402:
        return (
            "❌ **OpenRouter Account Balance Exceeded (402 Payment Required)**\n\n"
            f"**Cause**: {clean_detail or 'Your OpenRouter account has run out of credits or unpaid balance.'}\n\n"
            "**Action Required**: Visit [openrouter.ai/credits](https://openrouter.ai/credits) to top up your account balance."
        )

    if status_code == 429:
        return (
            "❌ **OpenRouter Rate Limit Exceeded (429 Too Many Requests)**\n\n"
            f"**Cause**: {clean_detail or 'You have exceeded your request rate limit or token limit on OpenRouter.'}\n\n"
            "**Action Required**: Please wait a few seconds before retrying your request."
        )

    if status_code == 404:
        return (
            "❌ **OpenRouter Model Not Found (404 Not Found)**\n\n"
            f"**Cause**: {clean_detail or 'The selected model tag is invalid or unavailable.'}\n\n"
            "**Action Required**: Open Settings and switch to a supported model (e.g., `google/gemma-4-31b-it` or `qwen/qwen3.5-flash-02-23`)."
        )

    if status_code == 400:
        return (
            "❌ **OpenRouter Invalid Request (400 Bad Request)**\n\n"
            f"**Cause**: {clean_detail or 'Invalid payload parameters or context window size exceeded.'}\n\n"
            "**Action Required**: Check your request prompt or clear recent conversation history."
        )

    if status_code in (500, 502, 503, 504):
        return (
            f"❌ **OpenRouter Provider Error ({status_code})**\n\n"
            f"**Cause**: {clean_detail or 'The upstream AI model provider is currently down or experiencing temporary service issues.'}\n\n"
            "**Action Required**: Please try again shortly or switch to a different model in Settings."
        )

    return (
        f"❌ **OpenRouter Error (HTTP {status_code})**\n\n"
        f"**Details**: {clean_detail or raw_str or 'Unknown API error'}"
    )


async def generate_openrouter_stream(
    system_prompt: str,
    user_prompt: str,
    context_chunks: Sequence[dict[str, Any]],
    history: list[dict[str, str]] | None = None,
    model_tag: str | None = None,
    api_key: str | None = None,
    search_enabled: bool = True,
    reasoning_steps: list[str] | None = None,
    reasoning_enabled: bool = False,
    reasoning_effort: str = "low",
) -> AsyncGenerator[str, None]:
    """Generate Server-Sent Events (SSE) stream for OpenRouter chat completions with dedicated system role, multi-turn history, and reasoning tokens.

    Args:
        system_prompt: System role prompt string.
        user_prompt: User role prompt string with context.
        context_chunks: Retrieved context snippets for citation metadata.
        history: Prior conversation history turns list.
        model_tag: OpenRouter model identifier string.
        api_key: User provided or fallback OpenRouter API key.
        search_enabled: Whether search RAG mode was enabled.
        reasoning_steps: Step-by-step pipeline execution reasoning trace strings.
        reasoning_enabled: Whether reasoning mode is enabled.
        reasoning_effort: OpenRouter reasoning effort ("low", "medium", "high", etc.).

    Yields:
        str: Formatted SSE payload strings (`data: {...}\n\n`).
    """
    settings = get_settings()

    active_api_key = (api_key or "").strip() or settings.openrouter_api_key
    active_model = (model_tag or "").strip() or settings.default_openrouter_model

    # Yield initial citation metadata and reasoning trace packet
    citations: list[dict[str, str]] = []
    if search_enabled and context_chunks:
        for chunk in context_chunks:
            meta = chunk.get("metadata", {})
            snippet_source = chunk.get("match_text", chunk.get("text", ""))
            citations.append(
                {
                    "filename": meta.get("filename", "Doc"),
                    "chunk_index": str(meta.get("chunk_index", 0)),
                    "snippet": snippet_source[:120] + "...",
                }
            )

    steps = reasoning_steps or []
    if not steps:
        if search_enabled:
            steps = [
                f"Searched ChromaDB vector database (found {len(context_chunks)} matching chunks)",
                "Applied semantic prompt augmentation",
            ]
        else:
            steps = ["Search disabled — relying on pre-trained LLM knowledge"]

    meta_packet = {
        "event": "meta",
        "model": active_model,
        "search_enabled": search_enabled,
        "chunk_count": len(context_chunks) if search_enabled else 0,
        "reasoning_steps": steps,
        "citations": citations,
        "reasoning_enabled": reasoning_enabled,
        "reasoning_effort": reasoning_effort,
    }
    yield f"data: {json.dumps(meta_packet)}\n\n"

    if not active_api_key:
        error_msg = (
            "❌ **OpenRouter API Key Missing**\n\n"
            "**Cause**: No OpenRouter API key was provided in your settings or server environment.\n\n"
            "**Action Required**: Click the **Settings** gear icon in the bottom left, enter your OpenRouter API Key (`sk-or-v1-...`), and click **Save**."
        )
        yield f"data: {json.dumps({'event': 'token', 'token': error_msg})}\n\n"
        yield "data: [DONE]\n\n"
        return

    headers = {
        "Authorization": f"Bearer {active_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "lemmesearch RAG",
    }

    # Pass dedicated system message and full conversation history to LLM
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    if history:
        for turn in history[-10:]:
            r = turn.get("role")
            c = turn.get("content")
            if r in ["user", "assistant"] and c:
                messages.append({"role": r, "content": c})

    messages.append({"role": "user", "content": user_prompt})

    payload: dict[str, Any] = {
        "model": active_model,
        "messages": messages,
        "stream": True,
        "temperature": 0.5 if search_enabled else 0.7,
    }

    # Attach OpenRouter standardized reasoning parameters
    if reasoning_enabled:
        payload["reasoning"] = {
            "effort": (reasoning_effort or "low").lower(),
            "enabled": True,
        }
    else:
        payload["reasoning"] = {
            "effort": "none",
            "exclude": True,
        }

    url = f"{settings.openrouter_base_url}/chat/completions"

    has_reasoning = False
    reasoning_token_count = 0
    in_think_tag = False

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                if response.status_code != 200:
                    raw_err_bytes = await response.aread()
                    formatted_err = format_openrouter_error(response.status_code, raw_err_bytes)
                    logger.error("OpenRouter API error (%d): %s", response.status_code, raw_err_bytes)
                    yield f"data: {json.dumps({'event': 'token', 'token': formatted_err})}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data: "):
                        content_str = line[6:]
                        if content_str == "[DONE]":
                            break
                        try:
                            data_obj = json.loads(content_str)
                            delta = data_obj.get("choices", [{}])[0].get("delta", {})

                            # 1. Extract explicit OpenRouter reasoning field
                            reasoning_chunk = delta.get("reasoning", "") or delta.get("reasoning_content", "")
                            if isinstance(reasoning_chunk, list):
                                reasoning_chunk = "".join([str(item) for item in reasoning_chunk])
                            elif not isinstance(reasoning_chunk, str):
                                reasoning_chunk = str(reasoning_chunk) if reasoning_chunk else ""

                            if reasoning_chunk:
                                if reasoning_enabled:
                                    has_reasoning = True
                                    reasoning_token_count += len(reasoning_chunk)
                                    yield f"data: {json.dumps({'event': 'reasoning_token', 'token': reasoning_chunk})}\n\n"
                                continue

                            # 2. Extract standard content token & check inline <think> tags fallback
                            token = delta.get("content", "")
                            if token:
                                if "<think>" in token:
                                    in_think_tag = True
                                    if reasoning_enabled:
                                        has_reasoning = True
                                        clean_reasoning = token.replace("<think>", "")
                                        if clean_reasoning:
                                            reasoning_token_count += len(clean_reasoning)
                                            yield f"data: {json.dumps({'event': 'reasoning_token', 'token': clean_reasoning})}\n\n"
                                    continue

                                if "</think>" in token:
                                    in_think_tag = False
                                    if reasoning_enabled:
                                        clean_content = token.replace("</think>", "")
                                        if clean_content:
                                            yield f"data: {json.dumps({'event': 'token', 'token': clean_content})}\n\n"
                                    continue

                                if in_think_tag:
                                    if reasoning_enabled:
                                        has_reasoning = True
                                        reasoning_token_count += len(token)
                                        yield f"data: {json.dumps({'event': 'reasoning_token', 'token': token})}\n\n"
                                else:
                                    yield f"data: {json.dumps({'event': 'token', 'token': token})}\n\n"

                        except json.JSONDecodeError:
                            continue

    except httpx.HTTPError as http_err:
        logger.error("HTTP network exception during OpenRouter streaming: %s", http_err)
        err_msg = (
            "❌ **Network Connection Error**\n\n"
            f"**Cause**: Failed to establish connection to OpenRouter servers ({http_err}).\n\n"
            "**Action Required**: Please check your network connection or server connectivity."
        )
        yield f"data: {json.dumps({'event': 'token', 'token': err_msg})}\n\n"
    except Exception as e:
        logger.error("Unexpected exception during OpenRouter streaming: %s", e)
        err_msg = f"❌ **System Execution Error**: {e}"
        yield f"data: {json.dumps({'event': 'token', 'token': err_msg})}\n\n"

    # Emit final Reasoning Validation Event if reasoning was requested
    if reasoning_enabled:
        validation_packet = {
            "event": "reasoning_validation",
            "reasoning_enabled": True,
            "reasoning_effort": reasoning_effort,
            "validated": has_reasoning,
            "reasoning_length": reasoning_token_count,
            "message": (
                f"Reasoning validated: Model returned active reasoning tokens (effort: {reasoning_effort})."
                if has_reasoning
                else f"Reasoning requested (effort: {reasoning_effort}), but model did not output reasoning tokens."
            ),
        }
        yield f"data: {json.dumps(validation_packet)}\n\n"

    yield "data: [DONE]\n\n"

