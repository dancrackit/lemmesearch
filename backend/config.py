"""Configuration settings for lemmesearch backend."""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def get_credential_dir() -> Path:
    """Get the credential directory path, migrating any old misspelled directories first."""
    base = Path.cwd()
    old_dir = base / "crediential"
    if old_dir.exists() and old_dir.is_dir():
        migrate_old_credentials()
    c = base / "credential"
    c.mkdir(parents=True, exist_ok=True)
    return c


def migrate_old_credentials() -> None:
    """Migrate settings from misspelled 'crediential' to 'credential'."""
    base = Path.cwd()
    old_dir = base / "crediential"
    new_dir = base / "credential"
    new_dir.mkdir(parents=True, exist_ok=True)

    import shutil
    for item in old_dir.iterdir():
        if item.is_file():
            dest = new_dir / item.name
            try:
                shutil.copy2(item, dest)
                item.unlink()
            except Exception:
                pass
    try:
        old_dir.rmdir()
    except Exception:
        pass


@dataclass(frozen=True)
class Settings:
    """System settings configuration container."""

    chroma_db_dir: Path = Path("chroma_db")
    collection_name: str = "lemmesearch_docs"
    default_embedding_model_fallback: str = "intfloat/e5-base-v2"
    default_openrouter_model_fallback: str = "google/gemma-4-31b-it"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    @property
    def models_dir(self) -> Path:
        """Get models directory, ensuring it is inside the virtual environment if running inside one."""
        cwd = Path.cwd()
        if ".venv" in cwd.parts:
            idx = cwd.parts.index(".venv")
            return Path(*cwd.parts[:idx + 1]) / "models"
        return cwd / ".venv" / "models"

    @property
    def credentials_dir(self) -> Path:
        """Return active credentials directory."""
        return get_credential_dir()

    @property
    def openrouter_api_key(self) -> str:
        """Fetch OpenRouter API key from environment or credential/openrouter_api_key.md at runtime."""
        env_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if env_key:
            return env_key

        key_file = self.credentials_dir / "openrouter_api_key.md"
        if key_file.exists():
            try:
                content = key_file.read_text(encoding="utf-8")
                match = re.search(r"(sk-or-v1-[A-Za-z0-9_-]+)", content)
                if match:
                    return match.group(1).strip()
                # Also try reading raw Key line if formatted differently
                key_line_match = re.search(r"-\s*\*\*Key:\*\*\s*`?([^\n`]+)`?", content)
                if key_line_match:
                    k = key_line_match.group(1).strip()
                    if k and k != "NOT_SET":
                        return k
            except Exception:
                pass
        return ""

    @property
    def default_openrouter_model(self) -> str:
        """Fetch default openrouter model tag from credential/models.md or fallback."""
        models_file = self.credentials_dir / "models.md"
        if models_file.exists():
            try:
                content = models_file.read_text(encoding="utf-8")
                match = re.search(r"-\s*\*\*Model Tag:\*\*\s*`?([^`\n]+)`?", content)
                if match and match.group(1).strip():
                    return match.group(1).strip()
            except Exception:
                pass
        return self.default_openrouter_model_fallback

    @property
    def default_embedding_model(self) -> str:
        """Fetch default embedding model from credential/models.md or fallback."""
        models_file = self.credentials_dir / "models.md"
        if models_file.exists():
            try:
                content = models_file.read_text(encoding="utf-8")
                match = re.search(r"-\s*\*\*Model Name:\*\*\s*`?([^`\n]+)`?", content)
                if match and match.group(1).strip():
                    return match.group(1).strip()
            except Exception:
                pass
        return self.default_embedding_model_fallback


def save_openrouter_api_key(api_key: str) -> None:
    """Save/update OpenRouter API key into credential file."""
    api_key = api_key.strip()
    if not api_key:
        return
    d = get_credential_dir()
    content = f"""# OpenRouter API Key Configuration

## Environment Variable
- **Variable Name:** `OPENROUTER_API_KEY`
- **Fallback / Local Storage Key:** `openrouter_api_key`
- **Base URL:** `https://openrouter.ai/api/v1`

## API Key Details
- **Key:** `{api_key}`
- **Header Format:** `Authorization: Bearer {api_key}`
- **HTTP Referer:** `http://localhost:7777`
- **Title Header:** `X-Title: lemmesearch RAG`
"""
    d.mkdir(parents=True, exist_ok=True)
    (d / "openrouter_api_key.md").write_text(content, encoding="utf-8")


def get_saved_custom_models_from_file() -> list[dict[str, str]]:
    """Parse saved custom models from credential/models.md file."""
    custom_models = []
    cred_dir = get_credential_dir()
    models_file = cred_dir / "models.md"
    if models_file.exists():
        try:
            content = models_file.read_text(encoding="utf-8")
            match = re.search(r"## Saved Custom Models\n(.*?)(?=\n## |\Z)", content, re.DOTALL)
            if match:
                lines = match.group(1).strip().splitlines()
                for line in lines:
                    m = re.search(r"-\s*`?([^`\n]+)`?", line)
                    if m:
                        tag = m.group(1).strip()
                        parts = tag.split("/")
                        short_name = parts[1] if len(parts) > 1 else tag
                        if tag and not any(cm["tag"] == tag for cm in custom_models):
                            custom_models.append({"tag": tag, "name": short_name})
        except Exception:
            pass
    return custom_models


def save_models_config(
    openrouter_model: str | None = None,
    embedding_model: str | None = None,
    saved_custom_models: list[Any] | None = None,
) -> None:
    """Save/update model settings into credential file."""
    settings = get_settings()
    llm_m = openrouter_model.strip() if openrouter_model else settings.default_openrouter_model
    emb_m = embedding_model.strip() if embedding_model else settings.default_embedding_model

    existing_custom = get_saved_custom_models_from_file()
    custom_tags = [m["tag"] for m in existing_custom]

    if saved_custom_models is not None:
        new_tags = []
        for item in saved_custom_models:
            tag = item.get("tag", "") if isinstance(item, dict) else str(item)
            tag = tag.strip()
            if tag and tag not in new_tags:
                new_tags.append(tag)
        custom_tags = new_tags

    if llm_m and llm_m not in custom_tags and llm_m not in ["google/gemma-4-31b-it", "qwen/qwen3.5-flash-02-23", "deepseek/deepseek-v4-flash"]:
        custom_tags.append(llm_m)

    custom_section = ""
    if custom_tags:
        items_str = "\n".join([f"- `{t}`" for t in custom_tags])
        custom_section = f"\n## Saved Custom Models\n{items_str}\n"

    d = get_credential_dir()
    content = f"""# Configured & Supported Models

## Default LLM Model
- **Model Tag:** `{llm_m}`
- **Provider:** OpenRouter
- **Config Location:** `backend/config.py` (`default_openrouter_model`)

## Additional Supported OpenRouter Models
- `qwen/qwen3.5-flash-02-23`
- `google/gemma-4-31b-it`
{custom_section}
## Default Embedding Model
- **Model Name:** `{emb_m}`
- **Config Location:** `backend/config.py` (`default_embedding_model`)
- **Storage Directory:** `.venv/models`
"""
    d.mkdir(parents=True, exist_ok=True)
    (d / "models.md").write_text(content, encoding="utf-8")


def get_settings() -> Settings:
    """Factory function for retrieving system settings.

    Returns:
        Settings: Instantiated settings object.
    """
    return Settings()


