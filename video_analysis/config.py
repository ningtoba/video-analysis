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

    # Entity tracking (ByteTrack via Ultralytics built-in — MIT)
    entity_tracking_enabled: bool = (
        True  # overridden by ENTITY_TRACKING_ENABLED env var
    )
    entity_tracker_type: str = (
        "bytetrack.yaml"  # overridden by ENTITY_TRACKER_TYPE env var  # "bytetrack.yaml" or "botsort.yaml"
    )

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

    # Tiered Frame Storage (v0.21.0)
    frame_storage_mode: str = "tiered"  # full, tiered, compressed
    frame_analysis_size: int = 960  # longest edge for analysis/CLIP frames
    frame_thumbnail_size: int = 320  # longest edge for timeline thumbnails
    frame_compression: str = "jpeg"  # jpeg, webp
    frame_compression_quality: int = 85  # 1-100

    # Video Quality Pre-Screening (v0.21.0)
    quality_screening_enabled: bool = True  # overridden by QUALITY_SCREENING_ENABLED
    quality_min_blur_threshold: float = 100.0  # Laplacian variance threshold
    quality_min_brightness: float = 30.0  # below this = too dark
    quality_max_brightness: float = 225.0  # above this = too bright
    quality_static_threshold: float = 0.98  # similarity threshold for static frames
    quality_skip_ocr_on_blurry: bool = True  # skip OCR on blurry/static frames
    quality_skip_yolo_on_dark: bool = True  # skip YOLO on too dark/bright frames

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

    # Structured JSON Logging (v0.22.0)
    structured_logging_enabled: bool = True  # overridden by STRUCTURED_LOGGING_ENABLED
    structured_logging_format: str = (
        "auto"  # "console", "json", or "auto" — overridden by STRUCTURED_LOGGING_FORMAT
    )
    structured_logging_level: str = "INFO"  # overridden by STRUCTURED_LOGGING_LEVEL

    # UI
    ui_host: str = "0.0.0.0"
    ui_port: int = 7860
    ui_share: bool = False
    ui_auth_enabled: bool = bool(os.environ.get("GRADIO_PASSWORD"))
    ui_auth_username: str = os.environ.get("GRADIO_USER", "admin")
    ui_auth_password: str = os.environ.get("GRADIO_PASSWORD", "")
    # Library
    library_max_videos: int = 50

    # Processing mode (v0.22.0)
    processing_mode: str = "video_full"  # "video_full", "audio_only", or "auto"

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

    # Conversation Memory (ChromaDB-backed Q&A memory, v0.22.0)
    conversation_memory_enabled: bool = bool(
        os.environ.get("CONVERSATION_MEMORY_ENABLED", "true").lower() == "true"
    )
    conversation_memory_max_entries: int = int(
        os.environ.get("CONVERSATION_MEMORY_MAX_ENTRIES", "50")
    )
    conversation_memory_ttl_days: int = int(
        os.environ.get("CONVERSATION_MEMORY_TTL_DAYS", "30")
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
    agentic_max_rounds: int = (
        4  # max iterative rounds (default: 4 — standard, multi-hop, scene-graph, self-check)
    )
    agentic_min_confidence: float = 0.5  # min avg top-3 score to stop early

    # Self-Check + Re-Retrieval (LLM-verified answer-evidence alignment, v0.27.0)
    self_check_enabled: bool = True  # enable LLM-based self-check verification
    self_check_max_rounds: int = 2  # max verification+reretrieval rounds
    self_check_min_confidence: float = 0.7  # min confidence to stop early

    # Face Recognition (InsightFace, v0.26.0)
    face_recognition_enabled: bool = (
        False  # overridden by FACE_RECOGNITION_ENABLED env var in __post_init__
    )
    face_detection_model: str = "buffalo_l"  # InsightFace model pack
    face_match_threshold: float = 0.45  # cosine similarity for identity matching
    face_max_faces: int = 0  # 0 = unlimited
    face_recognition_providers: str = "CUDAExecutionProvider,CPUExecutionProvider"

    # Gradio Workflow (v0.26.0)
    workflow_enabled: bool = True  # enable Gradio Workflow visual pipeline builder UI

    # Prometheus Metrics (v0.28.0)
    prometheus_enabled: bool = True  # overridden by PROMETHEUS_ENABLED env var
    prometheus_metrics_prefix: str = "va_"  # prefix for all metric names

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
        # Override entity_tracking_enabled from env var
        entity_env = os.environ.get("ENTITY_TRACKING_ENABLED", "").lower()
        if entity_env in ("false", "0", "no"):
            self.entity_tracking_enabled = False
        # Override entity_tracker_type from env var
        tracker_env = os.environ.get("ENTITY_TRACKER_TYPE", "").lower()
        if tracker_env in ("bytetrack.yaml", "botsort.yaml"):
            self.entity_tracker_type = tracker_env
        # Override quality_screening_enabled from env var
        quality_env = os.environ.get("QUALITY_SCREENING_ENABLED", "").lower()
        if quality_env in ("false", "0", "no"):
            self.quality_screening_enabled = False
        # Override structured_logging_enabled from env var
        sl_env = os.environ.get("STRUCTURED_LOGGING_ENABLED", "").lower()
        if sl_env in ("false", "0", "no"):
            self.structured_logging_enabled = False
        # Override structured_logging_format from env var
        fmt_env = os.environ.get("STRUCTURED_LOGGING_FORMAT", "").lower()
        if fmt_env in ("console", "json", "auto"):
            self.structured_logging_format = fmt_env
        # Override structured_logging_level from env var
        lvl_env = os.environ.get("STRUCTURED_LOGGING_LEVEL", "").upper()
        if lvl_env in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            self.structured_logging_level = lvl_env
        # Override frame_storage_mode from env var
        storage_env = os.environ.get("FRAME_STORAGE_MODE", "").lower()
        if storage_env in ("full", "tiered", "compressed"):
            self.frame_storage_mode = storage_env
        # Override processing_mode from env var
        processing_env = os.environ.get("PROCESSING_MODE", "").lower()
        if processing_env in ("video_full", "audio_only", "auto"):
            self.processing_mode = processing_env
        # Override face_recognition_enabled from env var
        face_env = os.environ.get("FACE_RECOGNITION_ENABLED", "").lower()
        if face_env in ("true", "1", "yes"):
            self.face_recognition_enabled = True
        # Override workflow_enabled from env var
        workflow_env = os.environ.get("WORKFLOW_ENABLED", "").lower()
        if workflow_env in ("false", "0", "no"):
            self.workflow_enabled = False
        # Override prometheus_enabled from env var
        prom_env = os.environ.get("PROMETHEUS_ENABLED", "").lower()
        if prom_env in ("false", "0", "no"):
            self.prometheus_enabled = False
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
