"""ChromaDB vector store manager module."""

import datetime
import logging
from pathlib import Path
from typing import Any, Callable

import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.embeddings import LocalEmbedder

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages document chunk persistence and vector search via ChromaDB."""

    def __init__(self, db_dir: Path, collection_name: str = "lemmesearch_docs") -> None:
        """Initialize persistent ChromaDB client and collection.

        Args:
            db_dir: Path to directory for persistent ChromaDB storage.
            collection_name: Target collection name.
        """
        self.db_dir = db_dir
        self.collection_name = collection_name
        self.db_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.db_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )
        logger.info("ChromaDB vector store ready at %s", self.db_dir)

    def ingest_document(
        self,
        doc_id: str,
        filename: str,
        file_size_str: str,
        chunks: list[dict[str, Any]],
        embedder: LocalEmbedder,
        chunk_size: int,
        chunk_overlap: int,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> int:
        """Embed and store document chunks into ChromaDB with progress callback.

        Args:
            doc_id: Unique identifier for document.
            filename: Name of the uploaded file.
            file_size_str: Formatted human-readable file size.
            chunks: List of chunk dicts from parser.
            embedder: LocalEmbedder instance.
            chunk_size: Token/character chunk size setting.
            chunk_overlap: Token/character chunk overlap setting.
            progress_callback: Optional callback(processed, total) for progress streaming.

        Returns:
            int: Number of inserted chunks.
        """
        if not chunks:
            return 0

        # Remove previous versions of document if exists
        self.delete_document(doc_id=doc_id, filename=filename)

        texts = [chunk["text"] for chunk in chunks]
        embeddings = embedder.embed_passages(
            texts, batch_size=64, progress_callback=progress_callback
        )
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}-chunk-{idx}"
            ids.append(chunk_id)
            metadatas.append(
                {
                    "doc_id": doc_id,
                    "filename": filename,
                    "file_size": file_size_str,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "timestamp": timestamp,
                }
            )

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info("Ingested %d chunks for document %s (%s) into ChromaDB", len(ids), filename, doc_id)
        return len(ids)

    def list_documents(self) -> list[dict[str, Any]]:
        """List all unique ingested documents with metadata.

        Returns:
            list[dict[str, Any]]: Aggregated list of document summary objects.
        """
        try:
            results = self.collection.get(include=["metadatas"])
        except Exception as e:
            logger.error("Failed to list documents from ChromaDB: %s", e)
            return []

        metadatas = results.get("metadatas") or []
        docs_map: dict[str, dict[str, Any]] = {}

        for meta in metadatas:
            if not meta:
                continue
            doc_id = meta.get("doc_id", meta.get("filename", "unknown"))
            if doc_id not in docs_map:
                docs_map[doc_id] = {
                    "id": doc_id,
                    "name": meta.get("filename", "Untitled"),
                    "ext": Path(meta.get("filename", ".txt")).suffix.replace(".", "").upper(),
                    "sizeStr": meta.get("file_size", "Unknown"),
                    "chunks": meta.get("total_chunks", 1),
                    "chunkSize": meta.get("chunk_size", 512),
                    "chunkOverlap": meta.get("chunk_overlap", 50),
                    "timestamp": meta.get("timestamp", "Recent"),
                }

        return list(docs_map.values())

    def delete_document(self, doc_id: str = "", filename: str = "") -> int:
        """Delete all vector chunks belonging to a document from ChromaDB.

        Args:
            doc_id: Unique document ID or filename.
            filename: Filename fallback.

        Returns:
            int: Number of deleted chunks.
        """
        target = (doc_id or filename).strip()
        if not target:
            return 0

        deleted_ids: set[str] = set()

        try:
            # 1. Match by doc_id metadata
            matched_by_id = self.collection.get(where={"doc_id": target}, include=[])
            if matched_by_id and matched_by_id.get("ids"):
                deleted_ids.update(matched_by_id["ids"])

            # 2. Match by filename metadata
            matched_by_name = self.collection.get(where={"filename": target}, include=[])
            if matched_by_name and matched_by_name.get("ids"):
                deleted_ids.update(matched_by_name["ids"])

            # Delete matched IDs
            if deleted_ids:
                id_list = list(deleted_ids)
                self.collection.delete(ids=id_list)
                logger.info("Permanently deleted %d vector chunks for '%s' from ChromaDB", len(id_list), target)
                return len(id_list)

        except Exception as e:
            logger.error("Error deleting document '%s' from ChromaDB: %s", target, e)

        return 0

    def purge_all(self) -> int:
        """Purge all document vector chunks from the collection completely.

        Returns:
            int: Number of deleted chunks.
        """
        try:
            count = self.collection.count()
            if count > 0:
                self.client.delete_collection(name=self.collection_name)
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name, metadata={"hnsw:space": "cosine"}
                )
                logger.info("Purged all %d chunks from ChromaDB collection '%s'", count, self.collection_name)
                return count
        except Exception as e:
            logger.error("Error purging all documents from ChromaDB: %s", e)
        return 0

    def query_similar(
        self, query: str, embedder: LocalEmbedder, top_k: int = 4, max_distance: float = 0.70
    ) -> list[dict[str, Any]]:
        """Retrieve top-K matching document chunks for a query.

        Args:
            query: Natural language query text.
            embedder: LocalEmbedder instance.
            top_k: Number of chunks to retrieve.
            max_distance: Maximum allowed cosine distance (0.0=exact match, 1.0=orthogonal).

        Returns:
            list[dict[str, Any]]: Retrieved chunks with text, metadata, and score.
        """
        if self.collection.count() == 0:
            return []

        query_vector = embedder.embed_query(query)
        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=min(top_k, self.collection.count()),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error("Error executing vector query: %s", e)
            return []

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        hits: list[dict[str, Any]] = []
        for doc_text, meta, dist in zip(documents, metadatas, distances):
            f_dist = float(dist)
            if f_dist <= max_distance:
                hits.append(
                    {
                        "text": doc_text,
                        "metadata": meta,
                        "distance": f_dist,
                    }
                )
            else:
                logger.debug("Filtered out chunk %s due to high cosine distance: %.4f > %.2f", meta.get("chunk_index"), f_dist, max_distance)

        return hits


    def get_chunks_by_indices(
        self, doc_id: str, chunk_indices: list[int]
    ) -> list[dict[str, Any]]:
        """Fetch specific document chunks by doc_id and chunk_indices using metadata filters.

        Args:
            doc_id: Document ID or filename.
            chunk_indices: List of chunk index numbers to retrieve.

        Returns:
            list[dict[str, Any]]: List of retrieved chunk objects.
        """
        if not doc_id or not chunk_indices:
            return []

        target_set = set(chunk_indices)
        target_ids = [f"{doc_id}-chunk-{idx}" for idx in chunk_indices]

        try:
            # 1. Direct ID lookup (Fastest & precise)
            results = self.collection.get(ids=target_ids, include=["documents", "metadatas"])
            documents = results.get("documents") or []
            metadatas = results.get("metadatas") or []

            fetched: list[dict[str, Any]] = []
            for doc_text, meta in zip(documents, metadatas):
                if meta and meta.get("chunk_index") in target_set:
                    fetched.append({"text": doc_text, "metadata": meta, "distance": 0.0})

            if fetched:
                return fetched

            # 2. Fallback match by doc_id metadata
            results = self.collection.get(
                where={"doc_id": doc_id},
                include=["documents", "metadatas"],
            )
            documents = results.get("documents") or []
            metadatas = results.get("metadatas") or []

            for doc_text, meta in zip(documents, metadatas):
                if meta and meta.get("chunk_index") in target_set:
                    fetched.append({"text": doc_text, "metadata": meta, "distance": 0.0})

            # 3. Fallback match by filename metadata
            if not fetched:
                results = self.collection.get(
                    where={"filename": doc_id},
                    include=["documents", "metadatas"],
                )
                documents = results.get("documents") or []
                metadatas = results.get("metadatas") or []
                for doc_text, meta in zip(documents, metadatas):
                    if meta and meta.get("chunk_index") in target_set:
                        fetched.append({"text": doc_text, "metadata": meta, "distance": 0.0})


            return fetched
        except Exception as e:
            logger.warning("Error fetching chunks by indices for %s: %s", doc_id, e)
            return []
