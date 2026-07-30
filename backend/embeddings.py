"""Embeddings wrapper module for local sentence-transformers."""

import logging
import os
from pathlib import Path
from typing import Callable, Sequence

from sentence_transformers import SentenceTransformer

from backend.config import get_settings

logger = logging.getLogger(__name__)


class LocalEmbedder:
    """Wrapper class around SentenceTransformer for generating embeddings."""

    def __init__(self, model_name: str = "intfloat/e5-base-v2", cache_dir: Path | None = None) -> None:
        """Initialize local SentenceTransformer model.

        Args:
            model_name: HuggingFace model identifier.
            cache_dir: Optional custom model cache directory.
        """
        if cache_dir is None:
            cache_dir = get_settings().models_dir

        cache_dir.mkdir(parents=True, exist_ok=True)
        resolved_cache = str(cache_dir.resolve())

        # Enforce HF and SentenceTransformers caches stay strictly within project directory
        os.environ["HF_HOME"] = resolved_cache
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = resolved_cache

        logger.info("Loading local embedding model '%s' from cache path: %s", model_name, resolved_cache)
        self.model_name = model_name

        try:
            self.model = SentenceTransformer(model_name, cache_folder=resolved_cache)
        except Exception as e:
            logger.warning(
                "Online model check for '%s' failed (%s). Retrying with local_files_only=True...",
                model_name,
                e,
            )
            try:
                self.model = SentenceTransformer(model_name, cache_folder=resolved_cache, local_files_only=True)
            except Exception as local_err:
                logger.error("Failed to load model '%s' locally or online: %s", model_name, local_err)
                raise local_err

    def embed_passages(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[list[float]]:
        """Generate embeddings for text passages in batches with progress feedback.

        Args:
            texts: List of text chunk strings.
            batch_size: Number of chunks per embedding batch.
            progress_callback: Callback function(processed_count, total_count).

        Returns:
            list[list[float]]: Dense float vector embeddings.
        """
        if not texts:
            return []

        total_texts = len(texts)
        prefixed_texts = [f"passage: {t}" for t in texts]
        all_embeddings: list[list[float]] = []

        total_batches = (total_texts + batch_size - 1) // batch_size
        logger.info("Starting embedding for %d passages across %d batches (batch_size=%d)...", total_texts, total_batches, batch_size)

        for i in range(0, total_texts, batch_size):
            batch = prefixed_texts[i : i + batch_size]
            batch_idx = (i // batch_size) + 1

            embeddings = self.model.encode(
                batch, normalize_embeddings=True, show_progress_bar=False
            )
            all_embeddings.extend(embeddings.tolist())

            processed = min(i + batch_size, total_texts)
            pct = int((processed / total_texts) * 100)
            logger.info("Embedding progress: batch %d/%d (%d/%d chunks - %d%%)", batch_idx, total_batches, processed, total_texts, pct)

            if progress_callback:
                try:
                    progress_callback(processed, total_texts)
                except Exception as cb_err:
                    logger.debug("Progress callback exception: %s", cb_err)

        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding vector for a user query.

        Args:
            query: Single text query string.

        Returns:
            list[float]: Dense float query vector.
        """
        prefixed_query = f"query: {query}"
        embedding = self.model.encode(
            prefixed_query, normalize_embeddings=True, show_progress_bar=False
        )
        return embedding.tolist()
