"""
Configuration for the video analysis platform.
"""

from pathlib import Path
from dataclasses import dataclass, field
import os


@dataclass
class Config:
    """Central configuration for the platform."""

    # Paths
    data_dir: Path = Path(os.environ.get("VIDEO_ANALYSIS_DATA", "data"))
    video_dir: Path = field(init=False)
    frames_dir: Path = field(init=False)
    audio_dir: Path = field(init=False)
    thumbnails_dir: Path = field(init=False)
    chroma_path: Path = field(init=False)

    # Processing
    whisper_model: str = "large-v3"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "int8_float16"

    # OpenCLIP
    clip_model: str = "ViT-B-32"  # "ViT-B-32" or "ViT-L-14"
    clip_pretrained_dataset: str = (
        "laion2b_s34b_b79k"  # ViT-B-32 default; ViT-L-14 uses laion2b_s32b_b82k
    )
    clip_embed_dim: int = 512  # 512 for ViT-B-32, 768 for ViT-L-14

    # Frame extraction
    frame_rate: float = 0.5  # 1 frame per 2 seconds default
    scene_threshold: float = (
        0.3  # PySceneDetect sensitivity (only used for ffmpeg/content mode)
    )
    scene_detector: str = (
        "adaptive"  # "adaptive", "content", "ffmpeg", "histogram", or "hash"
    )

    # YOLO
    yolo_model: str = "yolo26x.pt"  # Latest YOLO26
    yolo_confidence: float = 0.25

    # RAG — Embedding
    # BGE-VL-base as the primary embedding model (MIT, ~0.8 GB VRAM).
    # Replaces the dual-model approach (SentenceTransformer + Qwen3-VL).
    # Supports text-only, image-only, and composed (image+text) embeddings
    # in a single unified model.
    embedding_model: str = "BAAI/BGE-VL-base"  # 150M params, MIT license
    # Legacy text-only embedding model (used only when BGE-VL is unavailable)
    text_embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    # Multimodal embedding (Qwen3-VL-Embedding — optional, Apache 2.0)
    multimodal_embedding_model: str = "Qwen/Qwen3-VL-Embedding-2B"
    multimodal_embedding_enabled: bool = bool(
        os.environ.get("MULTIMODAL_EMBEDDING", "false").lower() == "true"
    )
    chroma_collection: str = "video_analysis"
    top_k_retrieval: int = 20
    top_k_rerank: int = 5
    temporal_window: int = 1  # neighbors on each side
    temporal_decay_rate: float = 0.1  # TV-RAG time-decay weighting (0 = disabled)
    colbert_reranker_enabled: bool = False  # optional ColBERTv2 late-interaction

    # LLM
    llm_model: str = "deepseek-ai/DeepSeek-V4-Flash"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2048

    # Clip export
    clip_export_dir: Path = field(init=False)

    # OCR (PaddleOCR)
    ocr_enabled: bool = True
    ocr_confidence: float = 0.3

    # Diarization (PyAnnote)
    diarize_enabled: bool = True

    # UI
    ui_host: str = "0.0.0.0"
    ui_port: int = 7860
    ui_share: bool = False
    ui_auth_enabled: bool = bool(os.environ.get("GRADIO_PASSWORD"))
    ui_auth_username: str = os.environ.get("GRADIO_USER", "admin")
    ui_auth_password: str = os.environ.get("GRADIO_PASSWORD", "")
    # Library
    library_max_videos: int = 50

    # Frame sampling
    adaptive_frame_sampling: bool = False  # motion-based adaptive sampling
    adaptive_frame_sampling_sensitivity: float = (
        0.3  # lower = more frames near boundaries
    )
    clip_frame_dedup: bool = False  # CLIP-similarity frame deduplication
    clip_frame_dedup_threshold: float = (
        0.92  # frames above this similarity are deduplicated
    )

    # YouTube / URL import
    yt_dlp_enabled: bool = True
    yt_dlp_format: str = "bestvideo[height<=1080]+bestaudio/best[height<=1080]"
    yt_dlp_output_template: str = "%(id)s.%(ext)s"

    # Batch processing
    batch_concurrent: int = 1  # process 1 video at a time (sequential queue)

    # Action recognition (optional X-CLIP)
    action_recognition_enabled: bool = (
        False  # overridden by ACTION_RECOGNITION_ENABLED env var in __post_init__
    )
    action_model_name: str = "microsoft/xclip-base-patch16-zero-shot"
    action_categories_count: int = 26  # all DEFAULT_ACTION_CATEGORIES

    # Video MLLM (optional VideoChat-Flash 2B / SmolVLM2 — ICLR 2026, MIT, ~5.4 GB VRAM)
    video_mllm_enabled: bool = (
        False  # overridden by VIDEO_MLLM_ENABLED env var in __post_init__
    )
    video_mllm_model: str = "OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448"
    video_mllm_backend: str = "auto"  # "auto", "videochat_flash", "smolvlm2"
    video_mllm_model_size: str = "2.2B"  # "2.2B", "500M", "256M" (SmolVLM2 only)
    video_mllm_as_describer: bool = (
        False  # use MLLM for scene descriptions instead of OpenCLIP
    )
    video_mllm_as_chat_backend: bool = (
        False  # use MLLM as video-native Q&A backend instead of Hermes CLI
    )

    # Scene Graph (VGent/ViG-RAG inspired — graph-based video retrieval)
    scene_graph_enabled: bool = True  # enable scene-graph retrieval layer
    scene_graph_k_hop: int = 2  # K-hop graph expansion (0 = disabled)
    scene_graph_temporal_window: int = 3  # max scene distance for temporal edges
    scene_graph_min_shared_entities: int = (
        1  # min shared objects/actions for entity edge
    )
    scene_graph_semantic_threshold: float = 0.85  # sim threshold for semantic edges

    # Query Routing (text/visual/temporal/multimodal dispatch)
    query_routing_enabled: bool = True  # enable query classification & routing
    query_routing_prefer_llm: bool = (
        True  # use LLM for classification (fast, single-turn)
    )

    # Multi-Hop Query Decomposition
    multi_hop_enabled: bool = True  # enable multi-hop query decomposition
    multi_hop_max_sub_queries: int = 4  # max sub-questions to generate
    multi_hop_rerank_top_k: int = 10  # top-k from each sub-query retrieval

    # Agentic RAG — Iterative Retrieval with Confidence Checking
    agentic_retrieval_enabled: bool = True  # enable iterative agentic retrieval loop
    agentic_max_rounds: int = 3  # max iterative rounds (default: 3)
    agentic_min_confidence: float = 0.5  # min avg top-3 score to stop early

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.video_dir = self.data_dir / "videos"
        self.frames_dir = self.data_dir / "frames"
        self.audio_dir = self.data_dir / "audio"
        self.thumbnails_dir = self.data_dir / "thumbnails"
        self.chroma_path = self.data_dir / "chroma"
        self.clip_export_dir = self.data_dir / "clips"
        # Override action_recognition_enabled from env var (can't read at class body time)
        env_val = os.environ.get("ACTION_RECOGNITION_ENABLED", "").lower()
        if env_val in ("true", "1", "yes"):
            self.action_recognition_enabled = True
        # Override video_mllm_enabled from env var
        mllm_env = os.environ.get("VIDEO_MLLM_ENABLED", "").lower()
        if mllm_env in ("true", "1", "yes"):
            self.video_mllm_enabled = True
        mllm_desc_env = os.environ.get("VIDEO_MLLM_AS_DESCRIBER", "").lower()
        if mllm_desc_env in ("true", "1", "yes"):
            self.video_mllm_as_describer = True
        mllm_chat_env = os.environ.get("VIDEO_MLLM_AS_CHAT_BACKEND", "").lower()
        if mllm_chat_env in ("true", "1", "yes"):
            self.video_mllm_as_chat_backend = True
        # Override video_mllm_backend from env var
        backend_env = os.environ.get("VIDEO_MLLM_BACKEND", "").lower()
        if backend_env in ("auto", "videochat_flash", "smolvlm2"):
            self.video_mllm_backend = backend_env
        # Override video_mllm_model_size from env var
        size_env = os.environ.get("VIDEO_MLLM_MODEL_SIZE", "").upper()
        if size_env in ("2.2B", "500M", "256M"):
            self.video_mllm_model_size = size_env
        for d in [
            self.data_dir,
            self.video_dir,
            self.frames_dir,
            self.audio_dir,
            self.thumbnails_dir,
            self.clip_export_dir,
            self.chroma_path,
        ]:
            d.mkdir(parents=True, exist_ok=True)
