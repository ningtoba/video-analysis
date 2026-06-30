"""
ColBERTv2 late-interaction re-ranker for RAG.

Wraps RAGatouille (AnswerDotAI) to provide token-level late-interaction
re-ranking as an optional enhancement over the default cross-encoder.

The ColBERTv2 model uses token-level matching instead of single-vector
similarity, which can improve retrieval precision for complex queries.

Installation:
    pip install ragatouille>=1.0.0

Usage:
    from video_analysis.colbert_reranker import ColBERTReranker

    reranker = ColBERTReranker()
    if reranker.available:
        reranked = reranker.rerank(query, chunks, top_k=5)
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class ColBERTReranker:
    """Optional ColBERTv2 late-interaction re-ranker.

    Wraps RAGatouille's ``RAGPretrainedModel`` for token-level re-ranking.
    Designed as an optional drop-in enhancement after ChromaDB retrieval
    and before (or instead of) the cross-encoder re-ranker.

    VRAM usage: ~2-3 GB when loaded, 0 when unloaded. Fits on 12 GB RTX 4070
    with sequential loading.
    """

    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        device: str = "cuda",
        index_root: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._index_root = index_root
        self._available = None  # lazily checked

    @property
    def available(self) -> bool:
        """Check if RAGatouille is installed.

        Returns:
            True if ragatouille can be imported, False otherwise.
        """
        if self._available is None:
            try:
                import ragatouille  # noqa: F401

                self._available = True
            except Exception:
                self._available = False
        return self._available

    def _load(self):
        """Lazily load the ColBERTv2 model."""
        if self._model is not None:
            return
        try:
            from ragatouille import RAGPretrainedModel

            self._model = RAGPretrainedModel.from_pretrained(
                self.model_name,
                device=self.device,
            )
            logger.info(f"Loaded ColBERTv2 model: {self.model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load ColBERTv2 model: {e}")
            raise

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
    ) -> List[tuple[str, float]]:
        """Re-rank documents using ColBERTv2 late-interaction scoring.

        Args:
            query: The user's query string.
            documents: List of document texts to re-rank.
            top_k: Number of top results to return.

        Returns:
            List of (document_text, score) tuples sorted by relevance
            (highest first).

        Raises:
            ImportError: If ragatouille is not installed.
        """
        if not self.available:
            raise ImportError(
                "ragatouille is not installed. Install it with: pip install ragatouille>=1.0.0"
            )

        self._load()

        # RAGatouille's search method re-ranks docs against a query
        results = self._model.rerank(
            query=query,
            documents=documents,
            k=top_k,
        )

        # results is a list of dicts with 'content' and 'score' keys
        scored = [(r["content"], float(r["score"])) for r in results]
        return scored

    def unload(self):
        """Free VRAM by releasing the model reference.

        Call this after re-ranking to free GPU memory for other pipeline
        stages.
        """
        self._model = None
        import gc

        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("ColBERTv2 model unloaded from GPU")
