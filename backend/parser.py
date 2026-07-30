"""Document parser module for text extraction and chunking."""

import logging
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def extract_text_from_file(file_path: Path) -> str:
    """Extract raw text from a document file (PDF, TXT, MD, etc.).

    Args:
        file_path: Path to the target file.

    Returns:
        str: Extracted text content.

    Raises:
        ValueError: If file format is unsupported or parsing fails.
    """
    ext = file_path.suffix.lower()

    if ext == ".pdf":
        try:
            doc = fitz.open(file_path)
            extracted_pages: list[str] = []
            for page in doc:
                text = page.get_text("text")
                if text and text.strip():
                    extracted_pages.append(text.strip())
            doc.close()
            full_text = "\n\n".join(extracted_pages)
            if not full_text.strip():
                raise ValueError(f"PDF file {file_path.name} contains no selectable text (scanned PDF or empty).")
            return full_text
        except Exception as e:
            logger.error("Failed to parse PDF file %s: %s", file_path, e)
            raise ValueError(f"Could not extract text from PDF '{file_path.name}': {e}") from e

    # Text / Code / Markdown files with encoding fallbacks
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "utf-16"]
    for enc in encodings:
        try:
            content = file_path.read_text(encoding=enc)
            if content:
                return content
        except (UnicodeDecodeError, FileNotFoundError, OSError):
            continue

    try:
        # Fallback binary decode with replacement
        raw_bytes = file_path.read_bytes()
        return raw_bytes.decode("utf-8", errors="replace")
    except Exception as e:
        logger.error("Failed to read text file %s: %s", file_path, e)
        raise ValueError(f"Could not read file: {file_path.name}") from e


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    filename: str = "",
) -> list[dict[str, Any]]:
    """Split raw text into structured chunks with chunk metadata.

    Args:
        text: Raw document text string.
        chunk_size: Maximum character count per chunk.
        chunk_overlap: Character overlap between consecutive chunks.
        filename: Original file name.

    Returns:
        list[dict[str, Any]]: List of chunk dictionaries containing content and metadata.
    """
    clean_text = text.strip()
    if not clean_text:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    raw_chunks = splitter.split_text(clean_text)
    total_chunks = len(raw_chunks)

    structured_chunks: list[dict[str, Any]] = []
    for idx, chunk in enumerate(raw_chunks):
        structured_chunks.append(
            {
                "text": chunk,
                "metadata": {
                    "filename": filename,
                    "chunk_index": idx,
                    "total_chunks": total_chunks,
                },
            }
        )

    return structured_chunks
