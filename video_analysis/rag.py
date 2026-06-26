"""
RAG (Retrieval-Augmented Generation) module for video context.

Indexes processed video content into Chroma vector DB and provides
hybrid retrieval with re-ranking for accurate video Q&A.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

from chromadb.errors import NotFoundError as ChromaNotFoundError

from video_analysis.config import Config
from video_analysis.models import VideoIndex, ChatSource, format_timestamp

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A retrieved and re-ranked chunk of video context."""

    chunk_id: str
    video_id: str
    text: str
    timestamp: float
    scene_id: int
    score: float
    frame_path: Optional[str] = None
    metadata: dict = None


@dataclass
class VideoLibraryInfo:
    """Summary info about an indexed video in the library."""

    video_id: str
    filename: str
    num_scenes: int = 0
    num_chunks: int = 0
    duration: float = 0.0
    has_sprite: bool = False


class VideoRAG:
    """
    Video RAG engine using Chroma vector store.

    Indexes transcript + scene summaries + frame descriptions per video,
    then retrieves relevant context for queries.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._collection = None
        self._embedding_model = None
        self._multimodal_embedder = None
        self._multimodal_tokenizer = None
        self._chroma_client = None

    @property
    def collection(self):
        if self._collection is None:
            self._init_chroma()
        return self._collection

    def _init_chroma(self):
        """Initialize ChromaDB client and collection."""
        try:
            import chromadb
        except ImportError:
            raise ImportError("chromadb not installed")

        self._chroma_client = chromadb.PersistentClient(
            path=str(self.config.chroma_path),
        )
        try:
            self._collection = self._chroma_client.get_collection(
                self.config.chroma_collection,
            )
            logger.info(f"Loaded existing collection: {self.config.chroma_collection}")
        except (ValueError, ChromaNotFoundError):
            self._collection = self._chroma_client.create_collection(
                self.config.chroma_collection,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"Created collection: {self.config.chroma_collection}")

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for text.

        When ``multimodal_embedding_enabled`` is set in config, this method
        delegates to :meth:`_get_multimodal_embedding` without an image path
        (text-only mode of the multimodal model).  Otherwise uses the
        configured SentenceTransformer embedding model.
        """
        # If multimodal embedding is enabled, route through the multimodal
        # model even for text-only queries — this avoids loading two models
        # (SentenceTransformer + Qwen3-VL) and keeps one unified embedding space.
        if self.config.multimodal_embedding_enabled:
            return self._get_multimodal_embedding(text)

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers not installed")

        if self._embedding_model is None:
            logger.info(f"Loading embedding model: {self.config.embedding_model}")
            self._embedding_model = SentenceTransformer(
                self.config.embedding_model,
                device="cuda",
                trust_remote_code=True,
            )

        emb = self._embedding_model.encode(text, normalize_embeddings=True)
        return emb.tolist()

    def _get_multimodal_embedding(
        self, text: str, image_path: Optional[str] = None
    ) -> List[float]:
        """Get multimodal embedding using Qwen3-VL-Embedding (optional).

        When ``image_path`` is provided, the embedding fuses visual + textual
        information in a shared semantic space.  Falls back to text-only
        embedding when the model is not available or when neither image nor
        long text is provided.

        Requirements:
            pip install qwen-vl-utils transformers torch Pillow
        """
        if not self.config.multimodal_embedding_enabled:
            return self._get_embedding(text)

        try:
            import torch
            from PIL import Image
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            logger.debug(
                "transformers or Pillow not available for multimodal embedding"
            )
            return self._get_embedding(text)

        model_id = self.config.multimodal_embedding_model
        try:
            if (
                not hasattr(self, "_multimodal_embedder")
                or self._multimodal_embedder is None
            ):
                logger.info(f"Loading multimodal embedding model: {model_id}")
                self._multimodal_tokenizer = AutoTokenizer.from_pretrained(
                    model_id, trust_remote_code=True
                )
                self._multimodal_embedder = AutoModel.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    device_map="cuda" if torch.cuda.is_available() else "cpu",
                )
                self._multimodal_embedder.eval()

            embedder = self._multimodal_embedder
            tokenizer = self._multimodal_tokenizer

            inputs = [{"text": text}]
            if image_path and Path(image_path).exists():
                inputs[0]["image"] = Image.open(image_path).convert("RGB")

            with torch.no_grad():
                emb = embedder.process(inputs)
            return emb[0].tolist()
        except Exception as e:
            logger.warning(
                f"Multimodal embedding failed ({e}); falling back to text embedding"
            )
            return self._get_embedding(text)

    def index_video(self, video_index: VideoIndex):
        """Index a processed video into Chroma."""
        chunks = []
        metadatas = []
        ids = []

        # Create chunks from scenes
        for scene in video_index.scenes:
            # Build rich text for this scene
            parts = []

            # Transcript
            if scene.transcript:
                parts.append(f"[Transcript]: {scene.transcript}")

            # Frame descriptions and objects
            frame_objects = set()
            for frame in scene.key_frames:
                if frame.description:
                    parts.append(
                        f"[Frame at {format_timestamp(frame.timestamp)}]: {frame.description}"
                    )
                for obj in frame.objects:
                    frame_objects.add(obj["label"])
                if frame.ocr_text and frame.ocr_text.strip():
                    parts.append(
                        f"[Text in frame at {format_timestamp(frame.timestamp)}]: {frame.ocr_text}"
                    )
                if frame.action:
                    conf_str = (
                        f" ({frame.action_confidence:.0%})"
                        if frame.action_confidence
                        else ""
                    )
                    parts.append(
                        f"[Action at {format_timestamp(frame.timestamp)}]: {frame.action}{conf_str}"
                    )

            if frame_objects:
                parts.append(f"[Objects detected]: {', '.join(sorted(frame_objects))}")

            scene_text = "\n".join(parts)
            if not scene_text.strip():
                continue

            chunk_id = f"{video_index.video_id}_scene_{scene.scene_id:04d}"
            chunks.append(scene_text)
            metadatas.append(
                {
                    "video_id": video_index.video_id,
                    "filename": video_index.filename,
                    "scene_id": scene.scene_id,
                    "start_time": scene.start_time,
                    "end_time": scene.end_time,
                    "type": "scene",
                }
            )
            ids.append(chunk_id)

            # Also index each key frame as a separate chunk for finer granularity
            for frame in scene.key_frames:
                frame_parts = []
                if scene.transcript:
                    frame_parts.append(
                        f"[Transcript around this time]: {scene.transcript[:500]}"
                    )
                if frame.description:
                    frame_parts.append(f"[Frame]: {frame.description}")
                if frame.objects:
                    obj_str = ", ".join(o["label"] for o in frame.objects[:10])
                    frame_parts.append(f"[Objects]: {obj_str}")
                if frame.ocr_text and frame.ocr_text.strip():
                    frame_parts.append(f"[Text]: {frame.ocr_text}")
                if frame.action:
                    conf_str = (
                        f" ({frame.action_confidence:.0%})"
                        if frame.action_confidence
                        else ""
                    )
                    frame_parts.append(f"[Action]: {frame.action}{conf_str}")

                frame_text = "\n".join(frame_parts)
                if len(frame_text) < 10:  # skip empty chunks
                    continue

                frame_chunk_id = f"{video_index.video_id}_frame_{frame.timestamp:.2f}"
                chunks.append(frame_text)
                metadatas.append(
                    {
                        "video_id": video_index.video_id,
                        "filename": video_index.filename,
                        "scene_id": scene.scene_id,
                        "start_time": frame.timestamp,
                        "end_time": frame.timestamp + 0.5,
                        "type": "frame",
                        "frame_path": frame.filepath,
                    }
                )
                ids.append(frame_chunk_id)

        # Also add full transcript as chunks
        if video_index.full_transcript:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", ". ", " ", ""],
            )
            transcript_chunks = splitter.split_text(video_index.full_transcript)
            for i, chunk_text in enumerate(transcript_chunks):
                chunk_id = f"{video_index.video_id}_transcript_{i}"
                estimated_time = (
                    i / max(len(transcript_chunks), 1)
                ) * video_index.duration
                chunks.append(f"[Transcript]: {chunk_text}")
                metadatas.append(
                    {
                        "video_id": video_index.video_id,
                        "filename": video_index.filename,
                        "scene_id": -1,
                        "start_time": estimated_time,
                        "end_time": estimated_time
                        + (video_index.duration / max(len(transcript_chunks), 1)),
                        "type": "transcript",
                    }
                )
                ids.append(chunk_id)

        if not chunks:
            logger.warning("No chunks to index")
            return

        # Generate embeddings in batch
        logger.info(f"Indexing {len(chunks)} chunks...")
        embeddings = [self._get_embedding(c) for c in chunks]

        # Batch add to Chroma
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch_end = min(i + batch_size, len(chunks))
            self.collection.add(
                ids=ids[i:batch_end],
                embeddings=embeddings[i:batch_end],
                documents=chunks[i:batch_end],
                metadatas=metadatas[i:batch_end],
            )

        logger.info(f"Indexed {len(chunks)} chunks for {video_index.video_id}")

    def retrieve(
        self, query: str, video_id: Optional[str] = None, top_k: Optional[int] = None
    ) -> List[RetrievedChunk]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Natural language question about the video
            video_id: Optional filter to specific video
            top_k: Number of results to return

        Returns:
            List of RetrievedChunk with scores
        """
        top_k = top_k or self.config.top_k_retrieval

        # Embed the query
        query_embedding = self._get_embedding(query)

        # Build metadata filter
        where = None
        if video_id:
            where = {"video_id": video_id}

        # Query Chroma
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2,  # fetch more for re-ranking
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        chunks = []
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            chunks.append(
                RetrievedChunk(
                    chunk_id=doc_id,
                    video_id=meta.get("video_id", "unknown"),
                    text=results["documents"][0][i],
                    timestamp=meta.get("start_time", 0),
                    scene_id=meta.get("scene_id", -1),
                    score=1.0
                    - (results["distances"][0][i] if results["distances"] else 0),
                    frame_path=meta.get("frame_path"),
                    metadata=meta,
                )
            )

        # Re-rank with cross-encoder if available
        try:
            chunks = self._rerank(query, chunks, top_k)
        except ImportError:
            # Sort by score descending and take top_k
            chunks.sort(key=lambda c: c.score, reverse=True)
            chunks = chunks[:top_k]

        # Optional ColBERTv2 late-interaction re-ranking
        if self.config.colbert_reranker_enabled:
            chunks = self._rerank_colbert(query, chunks, top_k)

        return chunks

    def _rerank(
        self, query: str, chunks: List[RetrievedChunk], top_k: int
    ) -> List[RetrievedChunk]:
        """Re-rank chunks using a cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError("sentence-transformers not installed")

        reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device="cuda",
        )

        pairs = [(query, c.text[:512]) for c in chunks]
        scores = reranker.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk.score = float(score)

        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks[:top_k]

    def _rerank_colbert(
        self, query: str, chunks: List[RetrievedChunk], top_k: int
    ) -> List[RetrievedChunk]:
        """Re-rank using optional ColBERTv2 late-interaction model.

        Falls back to the cross-encoder result if ragatouille is not installed.
        The ColBERTv2 model is loaded lazily, used for re-ranking, then
        unloaded to free VRAM.
        """
        try:
            from video_analysis.colbert_reranker import ColBERTReranker

            reranker = ColBERTReranker()

            if not reranker.available:
                logger.info(
                    "ColBERTv2 re-ranking unavailable (ragatouille not installed)"
                )
                return chunks[:top_k]

            # Extract text from chunks for re-ranking
            texts = [c.text for c in chunks]

            # Re-rank with ColBERTv2
            scored = reranker.rerank(query=query, documents=texts, top_k=top_k)

            # Map scores back to chunks
            score_map = {doc: score for doc, score in scored}
            for chunk in chunks:
                if chunk.text in score_map:
                    chunk.score = float(score_map[chunk.text])

            # Unload ColBERTv2 to free VRAM
            reranker.unload()

            chunks.sort(key=lambda c: c.score, reverse=True)
            logger.info("ColBERTv2 re-ranking complete")
            return chunks[:top_k]
        except ImportError:
            logger.info(
                "ColBERTv2 not available — falling back to cross-encoder results"
            )
            return chunks[:top_k]
        except Exception as e:
            logger.error(f"ColBERTv2 re-ranking failed: {e}")
            return chunks[:top_k]

    def expand_temporal_context(
        self, chunks: List[RetrievedChunk], video_id: str
    ) -> List[RetrievedChunk]:
        """
        Expand retrieved chunks to include temporal neighbors.
        Ensures the LLM gets surrounding context for each hit.
        """
        if not chunks:
            return chunks

        # Get all scene IDs from this video
        all_metas = self.collection.get(
            where={"video_id": video_id},
            include=["metadatas"],
        )

        scene_map = {}
        for i, meta in enumerate(all_metas["metadatas"]):
            sid = meta.get("scene_id")
            if sid >= 0:
                scene_map.setdefault(sid, []).append(i)

        expanded = set()
        result = []
        for chunk in chunks:
            sid = chunk.scene_id
            if sid < 0:
                if chunk.chunk_id not in expanded:
                    expanded.add(chunk.chunk_id)
                    result.append(chunk)
                continue

            # Add temporal neighbors
            for neighbor_sid in range(
                max(0, sid - self.config.temporal_window),
                sid + self.config.temporal_window + 1,
            ):
                neighbor_id = f"{video_id}_scene_{neighbor_sid:04d}"
                if neighbor_id not in expanded:
                    # Fetch from Chroma
                    try:
                        neigh = self.collection.get(
                            ids=[neighbor_id],
                            include=["documents", "metadatas"],
                        )
                        if neigh["ids"]:
                            meta = neigh["metadatas"][0]
                            expanded.add(neighbor_id)
                            result.append(
                                RetrievedChunk(
                                    chunk_id=neighbor_id,
                                    video_id=video_id,
                                    text=neigh["documents"][0],
                                    timestamp=meta.get("start_time", 0),
                                    scene_id=meta.get("scene_id", -1),
                                    score=chunk.score * 0.8,  # slightly discounted
                                    metadata=meta,
                                )
                            )
                    except Exception:
                        pass

        # Sort by timestamp for chronological context
        result.sort(key=lambda c: c.timestamp)
        return result

    def build_context(self, chunks: List[RetrievedChunk]) -> str:
        """Build a structured context string from retrieved chunks."""
        lines = []
        for chunk in chunks:
            header = f"[{format_timestamp(chunk.timestamp)}]"
            if chunk.scene_id >= 0:
                header += f" (Scene {chunk.scene_id})"
            lines.append(header)
            lines.append(chunk.text)
            lines.append("")  # blank line separator
        return "\n".join(lines)

    def get_source_citations(
        self, chunks: List[RetrievedChunk], top_n: int = 3
    ) -> List[ChatSource]:
        """Extract top source citations from retrieved chunks."""
        sources = []
        for chunk in chunks[:top_n]:
            sources.append(
                ChatSource(
                    text=chunk.text[:200],
                    timestamp=chunk.timestamp,
                    frame_path=chunk.frame_path,
                    scene_id=chunk.scene_id,
                    relevance_score=chunk.score,
                )
            )
        return sources

    def list_videos(self) -> List[str]:
        """List all indexed video IDs."""
        all_meta = self.collection.get(include=["metadatas"])
        video_ids = set()
        for meta in all_meta["metadatas"]:
            video_ids.add(meta.get("video_id"))
        return sorted(v for v in video_ids if v)

    def search_videos(self, query: str) -> List[str]:
        """Search indexed videos by filename (case-insensitive substring match)."""
        all_meta = self.collection.get(include=["metadatas"])
        q = query.lower().strip()
        if not q:
            return self.list_videos()
        matched = set()
        for meta in all_meta["metadatas"]:
            vid = meta.get("video_id", "")
            fname = meta.get("filename", "")
            if q in vid.lower() or q in fname.lower():
                matched.add(vid)
        return sorted(m for m in matched if m)

    def search_all(
        self,
        query: str,
        top_k: int = 20,
        image_path: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """Cross-video semantic search — retrieves relevant chunks from ALL
        indexed videos, without a ``video_id`` filter.

        When ``multimodal_embedding_enabled`` is set in config and
        ``image_path`` is provided, the query embedding fuses visual +
        textual information via Qwen3-VL-Embedding.

        Args:
            query: Natural language search query.
            top_k: Number of initial results from ChromaDB.
            image_path: Optional path to an image for multimodal search.

        Returns:
            Re-ranked list of RetrievedChunk sorted by relevance.
        """
        # Use multimodal embedding if configured and image is available
        if self.config.multimodal_embedding_enabled and image_path:
            query_embedding = self._get_multimodal_embedding(query, image_path)
        else:
            query_embedding = self._get_embedding(query)

        # No video_id filter → search all videos
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2,
            where=None,
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        chunks = []
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            chunks.append(
                RetrievedChunk(
                    chunk_id=doc_id,
                    video_id=meta.get("video_id", "unknown"),
                    text=results["documents"][0][i],
                    timestamp=meta.get("start_time", 0),
                    scene_id=meta.get("scene_id", -1),
                    score=1.0
                    - (results["distances"][0][i] if results["distances"] else 0),
                    frame_path=meta.get("frame_path"),
                    metadata=meta,
                )
            )

        # Re-rank with cross-encoder if available
        try:
            chunks = self._rerank(query, chunks, top_k)
        except ImportError:
            chunks.sort(key=lambda c: c.score, reverse=True)
            chunks = chunks[:top_k]

        return chunks

    def get_library_info(self, video_id: str) -> Optional[VideoLibraryInfo]:
        """Get summary info for a single video."""
        try:
            result = self.collection.get(
                where={"video_id": video_id},
                include=["metadatas"],
            )
            if not result["ids"]:
                return None
            metas = result["metadatas"]
            scene_ids = set()
            for m in metas:
                sid = m.get("scene_id", -1)
                if sid >= 0:
                    scene_ids.add(sid)
            # Get duration from the first scene's end_time
            max_end = max(m.get("end_time", 0) for m in metas if "end_time" in m)
            has_sprite = any(
                m.get("sprite_sheet") is not None for m in metas if "sprite_sheet" in m
            )
            return VideoLibraryInfo(
                video_id=video_id,
                filename=metas[0].get("filename", video_id),
                num_scenes=len(scene_ids),
                num_chunks=len(result["ids"]),
                duration=max_end if max_end > 0 else 0.0,
                has_sprite=has_sprite,
            )
        except Exception:
            return None

    def delete_video(self, video_id: str):
        """Remove all chunks for a video."""
        self.collection.delete(where={"video_id": video_id})
        logger.info(f"Deleted index for {video_id}")
