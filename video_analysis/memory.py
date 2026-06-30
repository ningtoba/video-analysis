"""
Conversation Memory module — persistent Q&A memory backed by ChromaDB.

Stores question-answer pairs with video_id metadata and timestamps in a
dedicated ``conversation_memory`` ChromaDB collection, separate from the
video search collection.  Supports eviction of old entries (max 50 entries,
30-day TTL) and graceful fallback to in-memory list storage when ChromaDB
is unavailable.

Uses the same BGE-VL-base embedding pipeline as ``VideoRAG._get_query_embedding``
(or SentenceTransformer fallback) so no additional VRAM is required.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from video_analysis.config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MemoryEntry:
    """A single Q&A pair stored in conversation memory.

    Attributes:
        question: The user's question.
        answer: The assistant's answer.
        timestamp: Unix epoch seconds when the entry was created.
        video_id: Optional video identifier this Q&A relates to.
    """

    question: str
    answer: str
    timestamp: float
    video_id: Optional[str] = None


# ---------------------------------------------------------------------------
# ChromaDB-backed conversation memory
# ---------------------------------------------------------------------------


class ConversationMemory:
    """Persistent conversation memory backed by a dedicated ChromaDB collection.

    Stores question-answer pairs with video metadata and timestamps.
    Automatically evicts old entries when the max count or TTL is exceeded.

    **Fallback behaviour**: If ChromaDB is not available (import error,
    connection error, etc.), the class falls back to a simple in-memory
    list store.  The in-memory store still respects max_entries and TTL
    but loses data on process restart.

    Usage::

        memory = ConversationMemory(config)
        memory.add_entry("What was that scene?", "The car chase at 12:30", video_id="vid1")
        relevant = memory.get_relevant("car scene")
        recent = memory.get_recent(count=5)
        memory.clear_all()
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialise conversation memory.

        Args:
            config: Application configuration.  If ``None``, a default
                ``Config()`` is created.  The memory is only active when
                ``config.conversation_memory_enabled`` is ``True``.
        """
        self.config = config or Config()

        # Tunable limits (from config)
        self.max_entries: int = self.config.conversation_memory_max_entries
        self.ttl_seconds: float = self.config.conversation_memory_ttl_days * 86400.0

        # ChromaDB client and collection (lazy)
        self._chroma_client = None
        self._collection = None

        # Embedding-model references (shared with VideoRAG pattern)
        self._bge_vl_model = None
        self._embedding_model = None  # SentenceTransformer fallback

        # In-memory fallback storage (used when ChromaDB is unavailable)
        self._fallback_store: List[MemoryEntry] = []

        # The fallback embedding function (set once during first embed call)
        self._embed_fn = None  # type: Optional[callable]

        self._initialised = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_entry(self, question: str, answer: str, video_id: Optional[str] = None) -> None:
        """Store a question-answer pair in conversation memory.

        Automatically evicts old entries when the max count or TTL is
        exceeded (eviction runs *before* inserting the new entry).

        Args:
            question: The user's question.
            answer: The assistant's answer.
            video_id: Optional video identifier this Q&A relates to.
        """
        # Evict old entries before adding the new one
        self._evict_old_entries()

        timestamp = time.time()

        if self._collection is not None:
            self._add_to_chroma(question, answer, video_id, timestamp)
        else:
            self._fallback_store.append(
                MemoryEntry(
                    question=question,
                    answer=answer,
                    timestamp=timestamp,
                    video_id=video_id,
                )
            )
        logger.debug("Added conversation memory entry (chroma=%s)", self._collection is not None)

    def get_relevant(self, question: str, top_k: int = 3) -> List[MemoryEntry]:
        """Retrieve top-*k* relevant past Q&A pairs based on semantic similarity.

        Args:
            question: The query to find relevant memories for.
            top_k: Number of results to return (default 3).

        Returns:
            List of ``MemoryEntry`` objects ordered by relevance
            (most relevant first).  Empty list if no entries exist.
        """
        if self._collection is not None:
            return self._get_relevant_chroma(question, top_k)

        # In-memory fallback: simple text overlap scoring
        return self._get_relevant_fallback(question, top_k)

    def get_recent(self, count: int = 5) -> List[MemoryEntry]:
        """Retrieve the most recent conversation entries.

        Args:
            count: Number of recent entries to return (default 5).

        Returns:
            List of ``MemoryEntry`` objects ordered newest first.
        """
        if self._collection is not None:
            return self._get_recent_chroma(count)

        # In-memory fallback
        return sorted(self._fallback_store, key=lambda e: e.timestamp, reverse=True)[:count]

    def clear_all(self) -> None:
        """Remove all conversation memory entries."""
        if self._collection is not None:
            try:
                existing = self._collection.get()
                if existing["ids"]:
                    self._collection.delete(ids=existing["ids"])
                    logger.info("Cleared all conversation memory entries (ChromaDB)")
            except Exception as exc:
                logger.warning("Failed to clear ChromaDB collection: %s", exc)
        self._fallback_store.clear()
        logger.debug("Cleared fallback conversation memory")

    # ------------------------------------------------------------------
    # ChromaDB initialisation
    # ------------------------------------------------------------------

    def _init_chroma(self) -> None:
        """Lazy-initialise the ChromaDB client and conversation memory collection."""
        if self._initialised:
            return
        self._initialised = True

        try:
            import chromadb
        except ImportError:
            logger.warning(
                "chromadb not installed — using in-memory fallback for conversation memory"
            )
            return

        try:
            self._chroma_client = chromadb.PersistentClient(
                path=str(self.config.chroma_path),
            )
            collection_name = "conversation_memory"
            try:
                self._collection = self._chroma_client.get_collection(collection_name)
                logger.info(
                    "Loaded existing conversation memory collection: %s",
                    collection_name,
                )
            except (ValueError, chromadb.errors.NotFoundError):
                self._collection = self._chroma_client.create_collection(
                    collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info("Created conversation memory collection: %s", collection_name)
        except Exception as exc:
            logger.warning(
                "Failed to initialise ChromaDB for conversation memory: %s — "
                "using in-memory fallback",
                exc,
            )
            self._collection = None
            self._chroma_client = None

    # ------------------------------------------------------------------
    # Embedding methods (mirrors VideoRAG._get_query_embedding pattern)
    # ------------------------------------------------------------------

    def _load_bge_vl(self):
        """Lazy-load BGE-VL-base embedding model (shared pattern with VideoRAG)."""
        if self._bge_vl_model is not None:
            return self._bge_vl_model

        try:
            import torch
            from transformers import AutoModel
        except ImportError:
            logger.debug("transformers not available for BGE-VL in conversation memory")
            return None

        model_id = self.config.embedding_model
        try:
            model = AutoModel.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
                device_map="cuda" if torch.cuda.is_available() else "cpu",
            )
            model.set_processor(model_id)
            model.eval()
            self._bge_vl_model = model
            return model
        except Exception as e:
            logger.debug("Failed to load BGE-VL for conversation memory: %s", e)
            return None

    def _get_bge_vl_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding using BGE-VL (text-only)."""
        model = self._load_bge_vl()
        if model is None:
            return None

        import torch

        try:
            with torch.no_grad():
                emb = model.encode(text=text)
            if isinstance(emb, (list, tuple)) and len(emb) == 1:
                emb = emb[0]
            if hasattr(emb, "ndim") and emb.ndim == 2 and emb.shape[0] == 1:
                emb = emb[0]
            return emb.tolist()
        except Exception as e:
            logger.debug("BGE-VL embedding failed in conversation memory: %s", e)
            return None

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for text.

        Primary: BGE-VL-base (same model as VideoRAG).
        Fallback: SentenceTransformer with prefix normalisation.
        """
        # Try BGE-VL first
        try:
            emb = self._get_bge_vl_embedding(text)
            if emb is not None:
                return emb
        except Exception:
            pass

        # Fallback to SentenceTransformer
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers not installed")

        if self._embedding_model is None:
            model_name = self.config.text_embedding_model
            logger.debug("Loading text embedding model for conversation memory: %s", model_name)
            self._embedding_model = SentenceTransformer(
                model_name,
                device="cuda",
                trust_remote_code=True,
            )

        # Apply query prefix (Nomic models need "search_query: ")
        from video_analysis.rag import _apply_embedding_prefix

        prefixed = _apply_embedding_prefix(text, self.config.text_embedding_model, "query")
        emb = self._embedding_model.encode(prefixed, normalize_embeddings=True)
        return emb.tolist()

    # ------------------------------------------------------------------
    # Eviction (max entries & TTL)
    # ------------------------------------------------------------------

    def _evict_old_entries(self) -> None:
        """Evict entries exceeding max_entries or older than TTL.

        Runs before every ``add_entry`` call.
        """
        if self._collection is not None:
            self._evict_chroma()
        else:
            self._evict_fallback()

    def _evict_chroma(self) -> None:
        """Evict old entries from ChromaDB."""
        try:
            # TTL eviction
            cutoff = time.time() - self.ttl_seconds
            all_data = self._collection.get(include=["metadatas"])
            if not all_data["ids"]:
                return

            to_delete = []
            for i, meta in enumerate(all_data["metadatas"]):
                ts = float(meta.get("timestamp", 0))
                if ts < cutoff:
                    to_delete.append(all_data["ids"][i])

            if to_delete:
                self._collection.delete(ids=to_delete)

            # Max-entries eviction (keep the newest N)
            remaining = self._collection.get(include=["metadatas"])
            if len(remaining["ids"]) > self.max_entries:
                # Sort by timestamp descending, keep top max_entries
                indexed = list(zip(remaining["ids"], remaining["metadatas"]))
                indexed.sort(key=lambda x: float(x[1].get("timestamp", 0)), reverse=True)
                keep_ids = {item[0] for item in indexed[: self.max_entries]}
                excess = [item[0] for item in indexed[self.max_entries :]]
                if excess:
                    self._collection.delete(ids=excess)
                    logger.debug(
                        "Evicted %d conversation memory entries (max_entries)",
                        len(excess),
                    )
        except Exception as exc:
            logger.warning("Failed to evict ChromaDB conversation memory: %s", exc)

    def _evict_fallback(self) -> None:
        """Evict old entries from the in-memory fallback store."""
        cutoff = time.time() - self.ttl_seconds
        self._fallback_store = [e for e in self._fallback_store if e.timestamp >= cutoff]
        # Keep newest max_entries
        if len(self._fallback_store) > self.max_entries:
            self._fallback_store.sort(key=lambda e: e.timestamp, reverse=True)
            self._fallback_store = self._fallback_store[: self.max_entries]

    # ------------------------------------------------------------------
    # ChromaDB CRUD helpers
    # ------------------------------------------------------------------

    def _add_to_chroma(
        self,
        question: str,
        answer: str,
        video_id: Optional[str],
        timestamp: float,
    ) -> None:
        """Embed and store a Q&A pair in ChromaDB."""
        # Lazy init on first add
        if self._collection is None:
            self._init_chroma()
            if self._collection is None:
                # Still None = fallback to in-memory
                self._fallback_store.append(
                    MemoryEntry(
                        question=question,
                        answer=answer,
                        timestamp=timestamp,
                        video_id=video_id,
                    )
                )
                return

        combined_text = f"{question} {answer}"
        try:
            embedding = self._get_embedding(combined_text)
        except Exception as exc:
            logger.warning(
                "Failed to compute embedding for conversation memory entry: %s "
                "— skipping ChromaDB add",
                exc,
            )
            return

        entry_id = f"conv_{timestamp:.6f}_{hash(question) & 0xFFFFFFFF:08x}"
        metadata = {
            "question": question,
            "answer": answer,
            "timestamp": timestamp,
        }
        if video_id:
            metadata["video_id"] = video_id

        try:
            self._collection.add(
                ids=[entry_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[combined_text],
            )
        except Exception as exc:
            logger.warning("Failed to add entry to ChromaDB: %s", exc)

    def _get_relevant_chroma(self, question: str, top_k: int) -> List[MemoryEntry]:
        """Query ChromaDB for semantically similar past Q&A entries."""
        try:
            query_emb = self._get_embedding(question)
        except Exception as exc:
            logger.warning("Failed to compute query embedding for conversation memory: %s", exc)
            return []

        try:
            results = self._collection.query(
                query_embeddings=[query_emb],
                n_results=min(top_k, self.max_entries),
                include=["metadatas", "distances"],
            )
        except Exception as exc:
            logger.warning("ChromaDB query failed for conversation memory: %s", exc)
            return []

        entries: List[MemoryEntry] = []
        if results.get("ids") and results["ids"][0]:
            for i, entry_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i]
                entries.append(
                    MemoryEntry(
                        question=meta.get("question", ""),
                        answer=meta.get("answer", ""),
                        timestamp=float(meta.get("timestamp", 0)),
                        video_id=meta.get("video_id"),
                    )
                )
        return entries

    def _get_recent_chroma(self, count: int) -> List[MemoryEntry]:
        """Retrieve the most recent entries from ChromaDB.

        ChromaDB doesn't natively support sort — we fetch all entries
        and sort in-memory.
        """
        try:
            all_data = self._collection.get(
                include=["metadatas"],
            )
        except Exception as exc:
            logger.warning("Failed to fetch recent entries from ChromaDB: %s", exc)
            return []

        if not all_data["ids"]:
            return []

        indexed = []
        for i, meta in enumerate(all_data["metadatas"]):
            indexed.append(
                (
                    float(meta.get("timestamp", 0)),
                    MemoryEntry(
                        question=meta.get("question", ""),
                        answer=meta.get("answer", ""),
                        timestamp=float(meta.get("timestamp", 0)),
                        video_id=meta.get("video_id"),
                    ),
                )
            )

        indexed.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in indexed[:count]]

    # ------------------------------------------------------------------
    # In-memory fallback relevance scoring
    # ------------------------------------------------------------------

    def _get_relevant_fallback(self, question: str, top_k: int) -> List[MemoryEntry]:
        """Simple keyword-overlap scoring for the in-memory fallback store.

        Scores each stored entry by the fraction of query words that appear
        in the combined question+answer text, weighted by word length.
        """
        if not self._fallback_store:
            return []

        query_words = set(question.lower().split())
        if not query_words:
            return self._fallback_store[:top_k]

        scored: List[Tuple[float, MemoryEntry]] = []
        for entry in self._fallback_store:
            combined = f"{entry.question} {entry.answer}".lower()
            entry_words = set(combined.split())
            if not entry_words:
                continue
            # Jaccard-like overlap with length weighting
            overlap = query_words & entry_words
            if not overlap:
                continue
            score = sum(len(w) for w in overlap) / (sum(len(w) for w in query_words) + 1e-9)
            scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]
