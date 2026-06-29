"""
Configuration for the video analysis platform.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List
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

    # ASR (Automatic Speech Recognition)
    # Qwen3-ASR is the current SOTA open-source ASR model (Jan 2026):
    #   1.7B: 5.76% avg WER across 52 languages, unified streaming+offline,
    #   word-level timestamps, context biasing, Apache 2.0
    #   0.6B: fast variant (0.064 RTF) for real-time / CPU deployment
    # Alternatives: faster-whisper large-v3 (7.44% WER, MIT, proven),
    #   Moonshine Voice (245M, 6.65% WER, MIT, CPU-only),
    #   Parakeet TDT 0.6B (NVIDIA, 25 languages, 2000x RTF, CPU-fast)
    asr_backend: str = "faster-whisper"  # "faster-whisper" | "qwen3-asr" | "moonshine" | "parakeet"
    whisper_model: str = "large-v3"  # only used when asr_backend=faster-whisper
    whisper_device: str = "cuda"
    whisper_compute_type: str = "int8_float16"
    qwen_asr_model: str = "Qwen/Qwen3-ASR-1.7B"  # 1.7B or 0.6B

    # OpenCLIP — scene classification & visual embeddings
    # ViT-L-14-quickgelu (DFN5B): best overall CLIP, 79.1% ImageNet zero-shot
    # ViT-H-14-quickgelu (DFN5B): top accuracy, needs 16GB+ VRAM
    # ViT-B-32 (laion2b): fastest, lowest VRAM
    # ViT-SO400M-14-SigLIP-384 (webli): strong multilingual + high-res
    clip_model: str = "ViT-L-14-quickgelu"  # "ViT-L-14-quickgelu" | "ViT-H-14-quickgelu" | "ViT-B-32" | "ViT-SO400M-14-SigLIP-384"
    clip_pretrained_dataset: str = "dfn5b"  # "dfn5b" | "laion2b_s34b_b79k" | "webli"
    clip_embed_dim: int = 768  # 768 for ViT-L, 1024 for ViT-H, 512 for ViT-B

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

    # RAG — Embedding (2026 state-of-the-art)
    # BGE-M3 (BAAI, 568M, MIT): production workhorse, dense+sparse+multi-vector,
    #   100+ languages, 8192 ctx. MTEB English ~64-67, strong cross-lingual.
    # BGE-VL-base (BAAI, 150M, MIT): multimodal text+image+composed, ~0.8GB VRAM.
    # Qwen3-Embedding-4B (Apache 2.0): #1 MTEB multilingual at 70.58.
    # Qwen3-Embedding-8B: highest accuracy, needs A100-class GPU.
    # NV-Embed-v2 (NVIDIA, 7.85B): #1 MTEB English at 72.31, research license.
    # nomic-embed-text-v2 (Nomic, 137M): lightweight, fast, good for basic search.
    embedding_model: str = "BAAI/BGE-M3"  # "BAAI/BGE-M3" | "BAAI/BGE-VL-base" | "Qwen/Qwen3-Embedding-4B" | "Qwen/Qwen3-Embedding-8B" | "nvidia/NV-Embed-v2" | "nomic-ai/nomic-embed-text-v2"
    text_embedding_model: str = "nomic-ai/nomic-embed-text-v2"
    multimodal_embedding_model: str = "Qwen/Qwen3-VL-Embedding-2B"  # cross-modal: text+image+video
    multimodal_embedding_enabled: bool = bool(
        os.environ.get("MULTIMODAL_EMBEDDING", "false").lower() == "true"
    )
    # Reranker — bge-reranker-v2-m3 (BAAI, 568M, MIT) is the standard pairing with BGE-M3
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    chroma_collection: str = "video_analysis"
    top_k_retrieval: int = 20
    top_k_rerank: int = 5
    temporal_window: int = 1  # neighbors on each side
    temporal_decay_rate: float = 0.1  # TV-RAG time-decay weighting (0 = disabled)
    colbert_reranker_enabled: bool = False  # optional ColBERTv2 late-interaction
    colbert_att_reranker_enabled: bool = (
        False  # ColBERT-Att attention-weighted (arXiv:2603.25248)
    )

    # MMR (Maximal Marginal Relevance) Diversity Re-Ranking (v0.34.0)
    # Reduces redundancy in retrieved context by balancing relevance and diversity.
    # See Carbonell & Goldstein (SIGIR'98) for the original MMR formulation.
    mmr_diversity_enabled: bool = False  # overridden by MMR_DIVERSITY_ENABLED env var
    mmr_lambda: float = 0.5  # MMR lambda [0,1]; 0 = pure diversity, 1 = pure relevance
    mmr_top_k: int = 15  # number of chunks to re-rank with MMR

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

    # LLM — BYOK (Bring Your Own Key) for cloud inference, or local models
    # Set llm_provider to use cloud APIs instead of local inference.
    # For local: leave api_key empty and ensure hermes/ollama/vllm is running.
    llm_provider: str = "local"  # "local" | "openai" | "anthropic" | "groq" | "deepseek" | "google"
    llm_api_key: str = os.environ.get("LLM_API_KEY", "")  # BYOK — set via env or Settings UI
    llm_api_base: str = ""  # custom API base URL (e.g. http://localhost:1234/v1 for LM Studio)
    llm_model: str = "deepseek-ai/DeepSeek-V4-Flash"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2048

    # Clip export
    clip_export_dir: Path = field(init=False)

    # OCR (PaddleOCR)
    ocr_enabled: bool = True
    ocr_confidence: float = 0.3
    ocr_model_version: str = "PP-OCRv6"  # "PP-OCRv6" or "PP-OCRv5"
    ocr_model_tier: str = "medium"  # "tiny", "small", "medium" (PP-OCRv6 tiers)

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

    # DINOv2 Perceptual Frame Compression (v0.30.0 — LongVU-style)
    # Uses facebook/dinov2-small (21M params, ~85 MB VRAM) to measure
    # perceptual similarity between frames and drop redundant ones.
    dino_frame_compression: bool = False  # enable DINOv2 adaptive frame compression
    dino_frame_compression_threshold: float = (
        0.88  # cosine sim threshold [0,1]; lower = more aggressive compression
    )
    dino_frame_compression_model: str = (
        "facebook/dinov2-small"  # or "facebook/dinov2-base"
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
    video_mllm_backend: str = (
        "auto"  # "auto", "videochat_flash", "smolvlm2", "qwen3_vl"
    )
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

    # Gradio Workflow (v0.29.0 — ui/workflow.py, gr.Workflow visual pipeline builder)
    workflow_enabled: bool = True  # enable Gradio Workflow visual pipeline builder tab

    # Prometheus Metrics (v0.28.0)
    prometheus_enabled: bool = True  # overridden by PROMETHEUS_ENABLED env var
    prometheus_metrics_prefix: str = "va_"  # prefix for all metric names

    # Streaming Pipeline (v0.32.0)
    streaming_enabled: bool = False  # overridden by STREAMING_ENABLED env var
    streaming_chunk_duration: float = (
        30.0  # overridden by STREAMING_CHUNK_DURATION env var
    )
    streaming_overlap: float = 2.0  # overridden by STREAMING_OVERLAP env var
    streaming_incremental_index: bool = (
        True  # overridden by STREAMING_INCREMENTAL_INDEX env var
    )
    streaming_max_chunks: int = (
        0  # overridden by STREAMING_MAX_CHUNKS env var (0 = unlimited)
    )

    # Live Stream Analysis (v0.40.0 — RTMP/RTSP/HLS support)
    live_stream_enabled: bool = False  # overridden by LIVE_STREAM_ENABLED env var
    live_stream_url: str = ""  # overridden by LIVE_STREAM_URL env var
    live_stream_source: str = (
        "rtmp"  # overridden by LIVE_STREAM_SOURCE env var: rtmp, rtsp, hls
    )
    live_stream_chunk_duration: float = (
        30.0  # overridden by LIVE_STREAM_CHUNK_DURATION env var
    )
    live_stream_sliding_window: int = (
        300  # overridden by LIVE_STREAM_SLIDING_WINDOW env var (seconds)
    )
    live_stream_auto_reconnect: bool = (
        True  # overridden by LIVE_STREAM_AUTO_RECONNECT env var
    )
    live_stream_max_retries: int = 3  # overridden by LIVE_STREAM_MAX_RETRIES env var
    live_stream_retry_delay: float = (
        5.0  # overridden by LIVE_STREAM_RETRY_DELAY env var
    )

    # Federated Video Search (v0.33.0 — MCP-based cross-instance query)
    federation_enabled: bool = False  # overridden by FEDERATION_ENABLED env var
    federation_peers: str = ""  # comma-separated URLs, overridden by FEDERATION_PEERS
    federation_timeout: float = 30.0  # overridden by FEDERATION_TIMEOUT env var
    federation_include_local: bool = True  # include local index in federated results

    # Agentic Video Understanding Agent (v0.36.0 — multi-tool agent)
    agent_enabled: bool = False  # overridden by AGENT_ENABLED env var
    agent_max_tools: int = 5  # max tool invocations per query

    # Hierarchical Multi-Agent Orchestrator (v0.51.0 — HiCrew-inspired multi-agent)
    orchestra_enabled: bool = False  # overridden by ORCHESTRA_ENABLED env var
    orchestra_max_agents: int = 5  # max specialist agents per query
    orchestra_confidence_threshold: float = 0.5  # early stopping confidence threshold

    # Camera Tab (v0.41.0 — webcam/live camera capture & analysis)
    camera_enabled: bool = False  # overridden by CAMERA_ENABLED env var

    # Autonomous Video Curator (v0.45.0 — closed-loop MCR exploration agent)
    curator_enabled: bool = False  # overridden by CURATOR_ENABLED env var
    curator_curiosity: float = 0.5  # overridden by CURATOR_CURIOSITY env var (0.0-1.0)
    curator_max_iterations: int = 15  # overridden by CURATOR_MAX_ITERATIONS env var
    curator_output_dir: str = ""  # overridden by CURATOR_OUTPUT_DIR env var

    # OpenTelemetry Tracing (v0.49.0)
    # Configured via standard OTEL_* env vars (OTEL_SERVICE_NAME, OTEL_EXPORTER_OTLP_ENDPOINT, etc.)
    telemetry_enabled: bool = bool(
        os.environ.get("TELEMETRY_ENABLED", "true").lower() == "true"
    )

    # Robust Agent Confidence (v0.50.0 — Robust-TO inspired per-evidence confidence scoring)
    # When enabled, the agent assesses per-frame trustworthiness (blur, brightness, motion,
    # occlusion) and weights evidence accordingly before synthesis. Frames below the
    # minimum trust threshold are skipped, avoiding the "Blind Trust Problem" where
    # the agent treats degraded frames as equally reliable.
    agent_confidence_enabled: bool = bool(
        os.environ.get("AGENT_CONFIDENCE_ENABLED", "false").lower() == "true"
    )
    agent_confidence_min_trust: float = float(
        os.environ.get("AGENT_CONFIDENCE_MIN_TRUST", "0.3")
    )  # frames below this trustworthiness are skipped
    agent_confidence_weight_mode: str = os.environ.get(
        "AGENT_CONFIDENCE_WEIGHT_MODE", "tiered"
    )  # "tiered" (high/medium/low) or "continuous"

    # Event-Causal RAG (v0.57.0 — arXiv:2605.06185, arXiv:2604.05418)
    event_causal_rag_enabled: bool = (
        False  # overridden by EVENT_CAUSAL_RAG_ENABLED env var
    )
    # Automatically run event segmentation + indexing during pipeline processing (v0.58.0)
    event_causal_rag_index_on_process: bool = bool(
        os.environ.get("EVENT_CAUSAL_RAG_INDEX_ON_PROCESS", "true").lower() == "true"
    )
    event_segmentation_strategy: str = "auto"  # "auto", "llm", "transcript", "temporal"
    event_causal_top_k: int = 10  # max events to return from bidirectional retrieval
    event_causal_semantic_weight: float = (
        0.5  # weight for semantic store vs causal store
    )
    event_max_duration_seconds: float = 300.0  # max event duration in seconds
    # Event-Causal RAG in chat retrieval (v0.58.0)
    event_causal_rag_in_chat: bool = bool(
        os.environ.get("EVENT_CAUSAL_RAG_IN_CHAT", "false").lower() == "true"
    )

    # Streaming Thinking (v0.57.0 — arXiv:2603.12262 amortized streaming reasoning)
    streaming_thinking_enabled: bool = (
        False  # overridden by STREAMING_THINKING_ENABLED env var
    )
    streaming_thinking_interval: int = 1  # think on every Nth chunk

    # Rate Limiting (v0.49.0)
    rate_limit_enabled: bool = bool(
        os.environ.get("RATE_LIMIT_ENABLED", "true").lower() == "true"
    )
    rate_limit_capacity: int = int(os.environ.get("RATE_LIMIT_CAPACITY", "100"))
    rate_limit_rate: float = float(
        os.environ.get("RATE_LIMIT_RATE", "1.6667")  # 100/minute
    )

    # Webhook config (v0.59.0)
    webhook_urls: List[str] = field(default_factory=list)
    webhook_timeout: float = 5.0  # seconds

    # Adaptive Pipeline Scaling (v0.60.0)
    # Dynamically adjusts per-stage quality/resolution based on video properties
    # and available GPU VRAM. Overridden by ADAPTIVE_SCALING_POLICY env var.
    adaptive_scaling_enabled: bool = bool(
        os.environ.get("ADAPTIVE_SCALING_ENABLED", "true").lower() == "true"
    )
    adaptive_scaling_policy: str = os.environ.get(
        "ADAPTIVE_SCALING_POLICY", "auto"
    )  # "conservative", "balanced", "performance", "auto"

    # Hugging Face authentication token for gated model access and higher rate limits.
    # Set via HF_TOKEN env var (standard Hugging Face convention).
    hf_token: str = os.environ.get("HF_TOKEN", "")

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
        # Override webhook config from env vars (v0.59.0)
        wh_url_env = os.environ.get("WEBHOOK_URL", "")
        if wh_url_env:
            self.webhook_urls = [u.strip() for u in wh_url_env.split(",") if u.strip()]
        wh_timeout_env = os.environ.get("WEBHOOK_TIMEOUT", "")
        if wh_timeout_env:
            try:
                val = float(wh_timeout_env)
                if val > 0:
                    self.webhook_timeout = val
            except ValueError:
                pass
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
        if backend_env in ("auto", "videochat_flash", "smolvlm2", "qwen3_vl"):
            self.video_mllm_backend = backend_env
        # Override video_mllm_model from env var
        model_env = os.environ.get("VIDEO_MLLM_MODEL", "").strip()
        if model_env:
            self.video_mllm_model = model_env
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
        # Override dino_frame_compression from env var
        dino_env = os.environ.get("DINO_FRAME_COMPRESSION", "").lower()
        if dino_env in ("true", "1", "yes"):
            self.dino_frame_compression = True
        dino_threshold_env = os.environ.get("DINO_FRAME_COMPRESSION_THRESHOLD", "")
        if dino_threshold_env:
            try:
                val = float(dino_threshold_env)
                if 0.0 <= val <= 1.0:
                    self.dino_frame_compression_threshold = val
            except ValueError:
                pass
        dino_model_env = os.environ.get("DINO_FRAME_COMPRESSION_MODEL", "")
        if dino_model_env:
            self.dino_frame_compression_model = dino_model_env
        # Override workflow_enabled from env var
        workflow_env = os.environ.get("WORKFLOW_ENABLED", "").lower()
        if workflow_env in ("false", "0", "no"):
            self.workflow_enabled = False
        # Override prometheus_enabled from env var
        prom_env = os.environ.get("PROMETHEUS_ENABLED", "").lower()
        if prom_env in ("false", "0", "no"):
            self.prometheus_enabled = False
        # Override streaming config from env vars
        stream_env = os.environ.get("STREAMING_ENABLED", "").lower()
        if stream_env in ("true", "1", "yes"):
            self.streaming_enabled = True
        stream_dur = os.environ.get("STREAMING_CHUNK_DURATION", "")
        if stream_dur:
            try:
                val = float(stream_dur)
                if val > 0:
                    self.streaming_chunk_duration = val
            except ValueError:
                pass
        stream_overlap = os.environ.get("STREAMING_OVERLAP", "")
        if stream_overlap:
            try:
                val = float(stream_overlap)
                if val >= 0:
                    self.streaming_overlap = val
            except ValueError:
                pass
        stream_index = os.environ.get("STREAMING_INCREMENTAL_INDEX", "").lower()
        if stream_index in ("false", "0", "no"):
            self.streaming_incremental_index = False
        stream_max = os.environ.get("STREAMING_MAX_CHUNKS", "")
        if stream_max:
            try:
                val = int(stream_max)
                if val >= 0:
                    self.streaming_max_chunks = val
            except ValueError:
                pass
        # Override federation config from env vars
        fed_env = os.environ.get("FEDERATION_ENABLED", "").lower()
        if fed_env in ("true", "1", "yes"):
            self.federation_enabled = True
        fed_peers = os.environ.get("FEDERATION_PEERS", "")
        if fed_peers:
            self.federation_peers = fed_peers
        fed_timeout = os.environ.get("FEDERATION_TIMEOUT", "")
        if fed_timeout:
            try:
                val = float(fed_timeout)
                if val > 0:
                    self.federation_timeout = val
            except ValueError:
                pass
        fed_local = os.environ.get("FEDERATION_INCLUDE_LOCAL", "").lower()
        if fed_local in ("false", "0", "no"):
            self.federation_include_local = False
        # Override agent_enabled from env var
        agent_env = os.environ.get("AGENT_ENABLED", "").lower()
        if agent_env in ("true", "1", "yes"):
            self.agent_enabled = True
        # Override agent_confidence config from env vars (v0.50.0)
        ac_env = os.environ.get("AGENT_CONFIDENCE_ENABLED", "").lower()
        if ac_env in ("true", "1", "yes"):
            self.agent_confidence_enabled = True
        ac_trust_env = os.environ.get("AGENT_CONFIDENCE_MIN_TRUST", "")
        if ac_trust_env:
            try:
                val = float(ac_trust_env)
                if 0.0 <= val <= 1.0:
                    self.agent_confidence_min_trust = val
            except ValueError:
                pass
        ac_weight_env = os.environ.get("AGENT_CONFIDENCE_WEIGHT_MODE", "").lower()
        if ac_weight_env in ("tiered", "continuous"):
            self.agent_confidence_weight_mode = ac_weight_env
        # Override orchestra config from env vars (v0.51.0)
        orch_env = os.environ.get("ORCHESTRA_ENABLED", "").lower()
        if orch_env in ("true", "1", "yes"):
            self.orchestra_enabled = True
        orch_max_env = os.environ.get("ORCHESTRA_MAX_AGENTS", "")
        if orch_max_env:
            try:
                val = int(orch_max_env)
                if 1 <= val <= 20:
                    self.orchestra_max_agents = val
            except ValueError:
                pass
        orch_conf_env = os.environ.get("ORCHESTRA_CONFIDENCE_THRESHOLD", "")
        if orch_conf_env:
            try:
                val = float(orch_conf_env)
                if 0.0 <= val <= 1.0:
                    self.orchestra_confidence_threshold = val
            except ValueError:
                pass
        # Override camera_enabled from env var (v0.41.0)
        camera_env = os.environ.get("CAMERA_ENABLED", "").lower()
        if camera_env in ("true", "1", "yes"):
            self.camera_enabled = True
        # Override curator config from env vars (v0.45.0)
        curator_env = os.environ.get("CURATOR_ENABLED", "").lower()
        if curator_env in ("true", "1", "yes"):
            self.curator_enabled = True
        curiosity_env = os.environ.get("CURATOR_CURIOSITY", "")
        if curiosity_env:
            try:
                val = float(curiosity_env)
                if 0.0 <= val <= 1.0:
                    self.curator_curiosity = val
            except ValueError:
                pass
        max_iter_env = os.environ.get("CURATOR_MAX_ITERATIONS", "")
        if max_iter_env:
            try:
                val = int(max_iter_env)
                if val > 0:
                    self.curator_max_iterations = val
            except ValueError:
                pass
        out_dir_env = os.environ.get("CURATOR_OUTPUT_DIR", "")
        if out_dir_env:
            self.curator_output_dir = out_dir_env
        # Override live stream config from env vars (v0.40.0)
        ls_env = os.environ.get("LIVE_STREAM_ENABLED", "").lower()
        if ls_env in ("true", "1", "yes"):
            self.live_stream_enabled = True
        ls_url = os.environ.get("LIVE_STREAM_URL", "")
        if ls_url:
            self.live_stream_url = ls_url
        ls_src = os.environ.get("LIVE_STREAM_SOURCE", "").lower()
        if ls_src in ("rtmp", "rtsp", "hls"):
            self.live_stream_source = ls_src
        ls_chunk = os.environ.get("LIVE_STREAM_CHUNK_DURATION", "")
        if ls_chunk:
            try:
                val = float(ls_chunk)
                if val > 0:
                    self.live_stream_chunk_duration = val
            except ValueError:
                pass
        ls_win = os.environ.get("LIVE_STREAM_SLIDING_WINDOW", "")
        if ls_win:
            try:
                val = int(ls_win)
                if val > 0:
                    self.live_stream_sliding_window = val
            except ValueError:
                pass
        ls_rec = os.environ.get("LIVE_STREAM_AUTO_RECONNECT", "").lower()
        if ls_rec in ("false", "0", "no"):
            self.live_stream_auto_reconnect = False
        ls_retries = os.environ.get("LIVE_STREAM_MAX_RETRIES", "")
        if ls_retries:
            try:
                val = int(ls_retries)
                if val >= 0:
                    self.live_stream_max_retries = val
            except ValueError:
                pass
        ls_retry_delay = os.environ.get("LIVE_STREAM_RETRY_DELAY", "")
        if ls_retry_delay:
            try:
                val = float(ls_retry_delay)
                if val > 0:
                    self.live_stream_retry_delay = val
            except ValueError:
                pass
        # Override OCR model version from env var
        ocr_ver_env = os.environ.get("OCR_MODEL_VERSION", "").lower()
        if ocr_ver_env in ("pp-ocrv6", "pp-ocrv5"):
            # Preserve the canonical casing: PP-OCRv6 or PP-OCRv5
            self.ocr_model_version = (
                "PP-OCRv5" if ocr_ver_env == "pp-ocrv5" else "PP-OCRv6"
            )
        ocr_tier_env = os.environ.get("OCR_MODEL_TIER", "").lower()
        if ocr_tier_env in ("tiny", "small", "medium"):
            self.ocr_model_tier = ocr_tier_env
        # Override MMR diversity config from env vars
        mmr_env = os.environ.get("MMR_DIVERSITY_ENABLED", "").lower()
        if mmr_env in ("true", "1", "yes"):
            self.mmr_diversity_enabled = True
        mmr_lambda_env = os.environ.get("MMR_LAMBDA", "")
        if mmr_lambda_env:
            try:
                val = float(mmr_lambda_env)
                if 0.0 <= val <= 1.0:
                    self.mmr_lambda = val
            except ValueError:
                pass
        mmr_topk_env = os.environ.get("MMR_TOP_K", "")
        if mmr_topk_env:
            try:
                val = int(mmr_topk_env)
                if val > 0:
                    self.mmr_top_k = val
            except ValueError:
                pass
        # Override event-causal RAG config from env vars (v0.57.0)
        ev_rag_env = os.environ.get("EVENT_CAUSAL_RAG_ENABLED", "").lower()
        if ev_rag_env in ("true", "1", "yes"):
            self.event_causal_rag_enabled = True
        ev_rag_index_env = os.environ.get(
            "EVENT_CAUSAL_RAG_INDEX_ON_PROCESS", ""
        ).lower()
        if ev_rag_index_env in ("false", "0", "no"):
            self.event_causal_rag_index_on_process = False
        ev_rag_chat_env = os.environ.get("EVENT_CAUSAL_RAG_IN_CHAT", "").lower()
        if ev_rag_chat_env in ("true", "1", "yes"):
            self.event_causal_rag_in_chat = True
        ev_strat_env = os.environ.get("EVENT_SEGMENTATION_STRATEGY", "").lower()
        if ev_strat_env in ("auto", "llm", "transcript", "temporal"):
            self.event_segmentation_strategy = ev_strat_env
        # Override streaming thinking config from env vars (v0.57.0)
        st_env = os.environ.get("STREAMING_THINKING_ENABLED", "").lower()
        if st_env in ("true", "1", "yes"):
            self.streaming_thinking_enabled = True
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
