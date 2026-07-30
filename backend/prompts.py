"""System Persona Prompts loader module reading from credential directory."""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_PROMPTS = {
    "concise": {
        "label": "Concise",
        "prompt": "You are a concise AI assistant. Provide clear, direct, and brief answers focused purely on the core solution without unnecessary filler.",
    },
    "dig_deeper": {
        "label": "Dig Deeper",
        "prompt": "You are a deep-dive analytical AI assistant. Provide comprehensive, structured, and in-depth explanations with thorough reasoning, detailed background, and technical context.",
    },
    "custom": {
        "label": "Custom",
        "prompt": "You are a custom AI Assistant tailored for specialized tasks.",
    },
}


def get_credential_dir() -> Path:
    """Get credential directory path."""
    base = Path.cwd()
    return base / "credential"


def parse_system_prompts_markdown(content: str) -> dict[str, str]:
    """Parse prompt sections from system_prompts.md content.

    Args:
        content: Raw markdown content string.

    Returns:
        dict[str, str]: Mapping of prompt key to prompt string.
    """
    prompts = {}
    sections = re.split(r"##\s+", content)
    for section in sections:
        if not section.strip():
            continue
        lines = section.strip().splitlines()
        header = lines[0].lower()

        code_match = re.search(r"```(?:text)?\n(.*?)\n```", section, re.DOTALL)
        if code_match:
            text = code_match.group(1).strip()
            if "concise" in header:
                prompts["concise"] = text
            elif "dig deeper" in header or "dig_deeper" in header:
                prompts["dig_deeper"] = text
            elif "custom" in header:
                prompts["custom"] = text
            elif "rag" in header or "system rag" in header:
                prompts["system_rag"] = text
    return prompts


def load_persona_prompts() -> dict[str, dict[str, str]]:
    """Load system persona prompts directly from the credential folder.

    Returns:
        dict[str, dict[str, str]]: Dictionary of prompt keys mapped to label and prompt text.
    """
    result = {k: dict(v) for k, v in DEFAULT_PROMPTS.items()}
    cred_dir = get_credential_dir()
    if not cred_dir.exists():
        return result

    # 1. Parse system_prompts.md if exists
    sys_file = cred_dir / "system_prompts.md"
    if sys_file.exists():
        try:
            parsed = parse_system_prompts_markdown(sys_file.read_text(encoding="utf-8"))
            for k, text in parsed.items():
                if k in result and text:
                    result[k]["prompt"] = text
        except Exception as e:
            logger.warning("Could not read system_prompts.md from %s: %s", sys_file, e)

    # 2. Check individual prompt files in credential directory (e.g., concise.md, dig_deeper.md, custom.md)
    for key in result.keys():
        for ext in [".md", ".txt"]:
            f_path = cred_dir / f"{key}{ext}"
            if f_path.exists():
                try:
                    content = f_path.read_text(encoding="utf-8").strip()
                    if content:
                        result[key]["prompt"] = content
                except Exception as e:
                    logger.warning("Could not read prompt file %s: %s", f_path, e)

    return result


def save_persona_prompts(prompts_map: dict[str, str]) -> None:
    """Save/update persona prompts into system_prompts.md and sub-files in credential directories.

    Args:
        prompts_map: Dictionary mapping prompt keys to prompt strings.
    """
    existing = load_persona_prompts()
    concise = prompts_map.get("concise", existing.get("concise", {}).get("prompt", DEFAULT_PROMPTS["concise"]["prompt"]))
    dig_deeper = prompts_map.get("dig_deeper", existing.get("dig_deeper", {}).get("prompt", DEFAULT_PROMPTS["dig_deeper"]["prompt"]))
    custom = prompts_map.get("custom", existing.get("custom", {}).get("prompt", DEFAULT_PROMPTS["custom"]["prompt"]))
    system_rag = prompts_map.get(
        "system_rag",
        "You are an expert AI Assistant and Analyst.\n\nAnswer the user's question directly by synthesizing the provided document context into a clear, original, and natural explanation.",
    )

    content = f"""# System Prompts

## 1. System RAG Prompt (`SYSTEM_RAG_PROMPT`)
**Source:** `backend/llm.py`

```text
{system_rag}
```

---

## 2. Concise Persona Prompt
**Source:** `credential/system_prompts.md`

```text
{concise}
```

---

## 3. Dig Deeper Persona Prompt
**Source:** `credential/system_prompts.md`

```text
{dig_deeper}
```

---

## 4. Custom Persona Prompt
**Source:** `credential/system_prompts.md`

```text
{custom}
```
"""
    cred_dir = get_credential_dir()
    cred_dir.mkdir(parents=True, exist_ok=True)
    (cred_dir / "system_prompts.md").write_text(content, encoding="utf-8")
    (cred_dir / "custom.md").write_text(custom, encoding="utf-8")
    (cred_dir / "concise.md").write_text(concise, encoding="utf-8")
    (cred_dir / "dig_deeper.md").write_text(dig_deeper, encoding="utf-8")


