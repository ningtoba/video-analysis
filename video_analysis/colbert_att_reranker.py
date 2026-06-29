"""
ColBERT-Att: Attention-Weighted Late Interaction Re-Ranker.

Implements ColBERT-Att (arXiv:2603.25248, Mar 2026) — a drop-in
enhancement over ColBERTv2 that integrates attention weights into
the late-interaction scoring function.

Standard ColBERTv2 MaxSim:
    score(Q, D) = Σ_{q_i ∈ Q} max_{d_j ∈ D} (E_{q_i} · E_{d_j})

ColBERT-Att (this module):
    score(Q, D) = Σ_{q_i ∈ Q} α_{q_i} · max_{d_j ∈ D} (β_{d_j} · E_{q_i} · E_{d_j})

Where α_{q_i} and β_{d_j} are the normalised attention weights of query
and document tokens from the BERT encoder's last layer.  Tokens the model
"pays more attention to" (higher weight) contribute more to the relevance
score, while low-attention tokens (stop words, filler) are down-weighted.

Per the paper, this yields +1-3% recall on MS-MARCO, BEIR, and LoTTE
benchmarks with zero additional training — the attention weights come
from the frozen ColBERTv2 / BERT checkpoint.

Installation (optional, for GPU):
    pip install transformers>=5.12.0 torch>=2.12.0

Usage:
    from video_analysis.colbert_att_reranker import ColBERTAttReranker

    reranker = ColBERTAttReranker()
    if reranker.available:
        reranked = reranker.rerank(query, documents, top_k=5)
        reranker.unload()  # free VRAM
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ColBERTAttReranker:
    """ColBERT-Att attention-weighted late-interaction re-ranker.

    Loads the full ColBERTv2 model via HuggingFace ``transformers``
    (``ColBertModel`` / ``HF_ColBERT``) with ``output_attentions=True``
    and applies attention-weighted MaxSim scoring.

    VRAM usage: ~2 GB when loaded, 0 when unloaded.
    """

    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        device: str = "cuda",
        use_fp16: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16
        self._model = None
        self._tokenizer = None
        self._available = None

    # ------------------------------------------------------------------
    # Availability & lazy loading
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        if self._available is None:
            try:
                import torch  # noqa: F401
                import transformers  # noqa: F401

                self._available = True
            except Exception:
                self._available = False
        return self._available

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoTokenizer

            # Load the model — ColBERTv2 uses HF_ColBERT / ColBertModel
            # in transformers 5.12+.  We use AutoModel.from_pretrained
            # which dispatches to the correct class.
            self._model = self._load_model()
            self._model.to(self.device)
            self._model.eval()

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            if self.use_fp16 and torch.cuda.is_available():
                self._model.half()

            logger.info(
                f"Loaded {type(self._model).__name__}: {self.model_name} "
                f"on {self.device}"
            )
        except Exception as e:
            logger.error(f"Failed to load ColBERT-Att model: {e}")
            self._available = False
            raise

    def _load_model(self):
        """Load model, trying various import paths for ColBERT."""
        try:
            # transformers 5.12+: ColBertModel
            from transformers import ColBertModel

            return ColBertModel.from_pretrained(self.model_name, output_attentions=True)
        except ImportError:
            pass
        try:
            # transformers 5.12+: HF_ColBERT
            from transformers import HF_ColBERT

            return HF_ColBERT.from_pretrained(self.model_name, output_attentions=True)
        except ImportError:
            pass
        # Fallback: use AutoModel (might be ColBertModel or BERT)
        from transformers import AutoModel

        return AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            output_attentions=True,
        )

    # ------------------------------------------------------------------
    # Token embedding + attention extraction
    # ------------------------------------------------------------------

    def _encode(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Encode texts through ColBERT, return embeddings + attention.

        Returns:
            (token_embeddings, attention_weights, token_counts)
            - token_embeddings: (total_tokens, dim) — all token vectors
            - attention_weights: (total_tokens,) — normalised attention
            - token_counts: list of token counts per input text
        """
        import torch

        self._load()

        # Tokenize
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )

        # Get token embeddings — ColBERT projects BERT's [CLS] token
        # representation through a linear layer.  The output is typically
        # ``last_hidden_state`` (which may already include the projection).
        # ``outputs.last_hidden_state`` shape: (B, T, D)
        embeddings = outputs.last_hidden_state  # (B, T, D)

        # Get attention weights from the last layer
        # attentions[-1] shape: (B, num_heads, T, T)
        attn = outputs.attentions[-1]  # last layer
        # Average over heads -> (B, T, T)
        attn_avg = attn.mean(dim=1)

        # For each token, compute its total attention received across
        # all other tokens in the sequence (token importance).
        # ``attn_avg[i, :, j]`` = attention from token j to token i.
        # Sum over source dimension → total attention *received* by
        # each token (how much other tokens attend to it).
        token_importance = attn_avg.sum(dim=-1)  # (B, T)

        # Zero out padding positions
        token_importance = token_importance * attention_mask.float()

        # Normalise per sequence (softmax over non-padding tokens)
        exp_importance = torch.exp(token_importance * attention_mask.float())
        sum_importance = exp_importance.sum(dim=-1, keepdim=True)
        sum_importance = torch.clamp(sum_importance, min=1e-10)
        token_weights = exp_importance / sum_importance  # (B, T)

        # Flatten
        flat_embeddings = embeddings[attention_mask.bool()]  # (total_tokens, D)
        flat_weights = token_weights[attention_mask.bool()]  # (total_tokens,)
        token_counts = attention_mask.sum(dim=-1).tolist()  # per-text

        return (
            flat_embeddings.cpu().numpy(),
            flat_weights.cpu().numpy(),
            [int(c) for c in token_counts],
        )

    # ------------------------------------------------------------------
    # Attention-weighted MaxSim scoring
    # ------------------------------------------------------------------

    def _attention_weighted_maxsim(
        self,
        query_embs: np.ndarray,  # (Nq, D)
        query_weights: np.ndarray,  # (Nq,)
        doc_embs: np.ndarray,  # (Nd, D)
        doc_weights: np.ndarray,  # (Nd,)
    ) -> float:
        """Compute ColBERT-Att attention-weighted MaxSim score.

        score = Σ_{q_i} α_i · max_{d_j} (β_j · cos_sim(E_q_i, E_d_j))

        where:
          α_i = attention weight of query token i (already normalised)
          β_j = attention weight of document token j (already normalised)

        Returns a single scalar relevance score.
        """
        # Normalise embeddings to unit vectors for cosine similarity
        q_norm = query_embs / (
            np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-10
        )
        d_norm = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-10)

        # Cosine similarity matrix: (Nq, Nd)
        sim = q_norm @ d_norm.T

        # Apply document attention weighting: β_j scales each doc token
        # contribution — reshape to (1, Nd) for broadcasting
        sim_weighted = sim * doc_weights[np.newaxis, :]

        # Max over document tokens per query token: (Nq,)
        max_per_query = sim_weighted.max(axis=1)

        # Apply query attention weighting: α_i scales each query token
        score = float(np.sum(query_weights * max_per_query))

        return score

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Re-rank documents using ColBERT-Att attention-weighted scoring.

        Args:
            query: The user's query string.
            documents: List of document texts to re-rank.
            top_k: Number of top results to return.

        Returns:
            List of (document_text, score) tuples sorted by relevance
            (highest first).
        """
        if not self.available:
            raise RuntimeError(
                "ColBERT-Att is unavailable — transformers not installed"
            )

        if not documents:
            return []

        self._load()

        # Encode query
        q_embs, q_weights, _ = self._encode([query])
        # q_embs: (Tq, D), q_weights: (Tq,)

        # Filter out special tokens from query (CLS, SEP, PAD)
        # We keep all tokens — the attention weights naturally handle
        # down-weighting special tokens.

        # Encode documents — batch to minimise overhead
        doc_emb_list, doc_weight_list, doc_counts = self._encode(documents)

        # Build per-document slices
        scores: List[Tuple[str, float]] = []
        offset = 0
        for doc_idx, (doc_text, n_tokens) in enumerate(zip(documents, doc_counts)):
            d_embs = doc_emb_list[offset : offset + n_tokens]
            d_weights = doc_weight_list[offset : offset + n_tokens]
            offset += n_tokens

            if d_embs.shape[0] == 0:
                continue

            score = self._attention_weighted_maxsim(
                q_embs, q_weights, d_embs, d_weights
            )
            scores.append((doc_text, score))

        # Sort descending by score
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def unload(self):
        """Free VRAM by releasing model references."""
        self._model = None
        self._tokenizer = None
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass
        logger.info("ColBERT-Att model unloaded from GPU")
