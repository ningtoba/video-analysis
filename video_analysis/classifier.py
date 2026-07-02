"""
File-type based video classifier — determines media type and selects pipeline stages.

Provides three layers of classification with configurable fallback:

  1. **Extension-based** (fastest, zero I/O): Map file suffixes to heuristic types.
  2. **Content sniffing via ffprobe** (fast, <100ms): Parse codec, resolution, duration,
     and stream layout from the actual file header.
  3. **ML-based** (optional, ~1-2 GB VRAM): Lightweight ResNet-18 / mobilenet_v3
     classifier that samples the first frame for a "video scene type" label.

The result is a :class:`VideoClassification` dataclass that drives a configurable
stage-selection map (which pipeline stages to run for which media types).
"""

import json
import logging
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ============================================================================
# 1.  Media type taxonomy
# ============================================================================


class MediaType(str, Enum):
    """High-level media category determined by the classifier."""

    VIDEO = "video"  # Standard video file with visual + audio streams
    AUDIO = "audio"  # Audio-only file (podcast, music, lecture recording)
    IMAGE = "image"  # Static image (not processed by the video pipeline)
    UNKNOWN = "unknown"  # Unrecognised or ambiguous


class VideoSubType(str, Enum):
    """Finer-grained video classification for stage tuning."""

    STANDARD = "standard"  # General video (interviews, vlogs, movies)
    SCREENCAST = "screencast"  # Screen recording / slides / coding
    LECTURE = "lecture"  # Talking-head lecture (mostly static)
    MUSIC_VIDEO = "music_video"  # High-motion, fast cuts, music
    PODCAST = "podcast"  # Talking heads, limited motion
    TALKING_HEAD = "talking_head"  # Single person speaking to camera
    ANIMATION = "animation"  # Cartoon / animated content
    SPORTS = "sports"  # High motion, fast cuts, outdoor
    SECURITY = "security"  # Surveillance / CCTV (low motion, long duration)
    SHORT = "short"  # <30 s clips (TikTok, Reels, memes)
    UNKNOWN = "unknown"


# ============================================================================
# 2.  Classification result
# ============================================================================


@dataclass
class VideoClassification:
    """Result of the file-type classification pipeline.

    Attributes:
        media_type: High-level media category (video / audio / image).
        video_subtype: Finer-grained video subtype (for videos only).
        file_extension: The original file extension (e.g. ``".mp4"``).
        has_video_stream: Whether the file contains a video stream.
        has_audio_stream: Whether the file contains an audio stream.
        codec: Video codec string (e.g. ``"h264"``, ``"hevc"``).
        audio_codec: Audio codec string (e.g. ``"aac"``, ``"opus"``).
        width: Frame width in pixels (0 if no video stream).
        height: Frame height in pixels (0 if no video stream).
        duration_seconds: File duration in seconds.
        fps: Framerate (approximate, 0.0 if unknown).
        ml_label: ML-based scene label (only populated when ML classifier runs).
        ml_confidence: Confidence score for the ML label (0.0 if not used).
        raw_ffprobe: Raw ffprobe JSON for debugging / custom rules.
    """

    media_type: MediaType = MediaType.UNKNOWN
    video_subtype: VideoSubType = VideoSubType.UNKNOWN
    file_extension: str = ""
    has_video_stream: bool = False
    has_audio_stream: bool = False
    codec: str = ""
    audio_codec: str = ""
    width: int = 0
    height: int = 0
    duration_seconds: float = 0.0
    fps: float = 0.0
    ml_label: str = ""
    ml_confidence: float = 0.0
    raw_ffprobe: Dict = field(default_factory=dict)

    @property
    def is_video(self) -> bool:
        """Convenience: is this a video file?"""
        return self.media_type == MediaType.VIDEO

    @property
    def is_audio(self) -> bool:
        """Convenience: is this an audio file?"""
        return self.media_type == MediaType.AUDIO

    @property
    def resolution(self) -> str:
        """Human-readable resolution string (e.g. ``"1920x1080"``)."""
        if self.width and self.height:
            return f"{self.width}x{self.height}"
        return ""

    @property
    def is_high_resolution(self) -> bool:
        """Heuristic: is this at least 1080p?"""
        return self.width >= 1920 or self.height >= 1080


# ============================================================================
# 3.  File-extension heuristic (fastest layer, zero I/O)
# ============================================================================

# Typical video file extensions
VIDEO_EXTENSIONS: Set[str] = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".wmv",
    ".flv",
    ".webm",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".ts",
    ".mts",
    ".3gp",
    ".ogv",
}

# Typical audio file extensions
AUDIO_EXTENSIONS: Set[str] = {
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".aac",
    ".wma",
    ".m4a",
    ".opus",
    ".aiff",
    ".alac",
}

# Typical image file extensions
IMAGE_EXTENSIONS: Set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
}


def classify_by_extension(filepath: Path) -> MediaType:
    """Determine media type purely from file extension (zero I/O).

    This is the first and fastest classification layer.
    """
    ext = filepath.suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        return MediaType.VIDEO
    if ext in AUDIO_EXTENSIONS:
        return MediaType.AUDIO
    if ext in IMAGE_EXTENSIONS:
        return MediaType.IMAGE
    return MediaType.UNKNOWN


# ============================================================================
# 4.  Content sniffing via ffprobe (second layer, <100 ms)
# ============================================================================


def sniff_with_ffprobe(filepath: Path, timeout: int = 15) -> Optional[Dict]:
    """Probe file metadata with ffprobe.

    Returns a parsed JSON dict with ``format`` and ``streams`` keys, or
    ``None`` on failure.  Uses a single fast invocation with targeted
    output selectors.
    """
    if not filepath.exists():
        logger.warning(f"File not found: {filepath}")
        return None

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(filepath),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            logger.debug(f"ffprobe returned {result.returncode} for {filepath}")
            return None
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        logger.warning(f"ffprobe JSON parse error for {filepath}: {e}")
        return None
    except subprocess.TimeoutExpired:
        logger.warning(f"ffprobe timed out ({timeout}s) for {filepath}")
        return None
    except FileNotFoundError:
        logger.error("ffprobe not found — install ffmpeg")
        return None
    except Exception as e:
        logger.warning(f"ffprobe error for {filepath}: {e}")
        return None


def _find_video_stream(streams: List[Dict]) -> Optional[Dict]:
    """Return the first video stream from a list of ffprobe stream dicts."""
    for s in streams:
        if s.get("codec_type") == "video":
            return s
    return None


def _find_audio_stream(streams: List[Dict]) -> Optional[Dict]:
    """Return the first audio stream from a list of ffprobe stream dicts."""
    for s in streams:
        if s.get("codec_type") == "audio":
            return s
    return None


def parse_ffprobe_output(data: Dict) -> Dict:
    """Extract structured classification fields from raw ffprobe JSON.

    Returns a flat dict with keys: ``has_video``, ``has_audio``, ``codec``,
    ``audio_codec``, ``width``, ``height``, ``duration``, ``fps``.
    """
    streams = data.get("streams", [])
    fmt = data.get("format", {})

    video_stream = _find_video_stream(streams)
    audio_stream = _find_audio_stream(streams)

    # Duration: prefer stream-level, fall back to format-level
    duration = 0.0
    if video_stream:
        dur_str = video_stream.get("duration") or fmt.get("duration", "0")
    else:
        dur_str = fmt.get("duration", "0")
    try:
        duration = float(dur_str)
    except (ValueError, TypeError):
        duration = 0.0

    # FPS: parse from avg_frame_rate or r_frame_rate (e.g. "30000/1001")
    fps = 0.0
    if video_stream:
        for key in ("avg_frame_rate", "r_frame_rate"):
            rate_str = video_stream.get(key, "0/1")
            if "/" in rate_str:
                try:
                    num, den = rate_str.split("/")
                    fps = float(num) / float(den)
                    if fps > 0:
                        break
                except (ValueError, ZeroDivisionError):
                    continue

    return {
        "has_video": video_stream is not None,
        "has_audio": audio_stream is not None,
        "codec": (video_stream or {}).get("codec_name", "").lower(),
        "audio_codec": (audio_stream or {}).get("codec_name", "").lower(),
        "width": int((video_stream or {}).get("width", 0)),
        "height": int((video_stream or {}).get("height", 0)),
        "duration": duration,
        "fps": round(fps, 2),
    }


# ============================================================================
# 5.  Heuristic subtype classifier (rule-based, zero ML)
# ============================================================================


def classify_video_subtype(info: Dict, duration: float) -> VideoSubType:
    """Infer video subtype from ffprobe metadata using heuristic rules.

    Uses resolution, aspect ratio, duration, and codec patterns to
    distinguish screencasts, lectures, podcasts, shorts, etc.
    """
    has_audio = info.get("has_audio", False)
    codec = info.get("codec", "")
    width = info.get("width", 0)
    height = info.get("height", 0)
    fps = info.get("fps", 0.0)

    # Short clips (<30 s)
    if 0 < duration < 30:
        return VideoSubType.SHORT

    # Sports: high FPS, high resolution, long duration
    if has_audio and fps >= 29 and height >= 720 and duration > 600:
        return VideoSubType.SPORTS

    # Screencast heuristic: gif/vc1 codec, or exact 1920x1080 or 1366x768 no audio
    # Common screencast codecs: gif, vc1, msmpeg4
    if codec in ("gif", "vc1", "msmpeg4v2", "msmpeg4"):
        return VideoSubType.SCREENCAST

    # Security / surveillance: no audio, ~15 FPS, 4:3 or low-res
    if not has_audio and width > 0 and height > 0:
        if fps < 20 and (width * height) < 640 * 480:
            return VideoSubType.SECURITY
        # Could also be a screencast without audio
        if fps < 16:
            return VideoSubType.SECURITY

    # Music video: has audio, high FPS often, fast cuts
    if has_audio and fps > 30:
        return VideoSubType.MUSIC_VIDEO

    # Podcast heuristic: 16:9, 1080p or lower, 2 audio channels, talking-head duration
    if has_audio:
        aspect = width / height if height > 0 else 0
        if 1.7 < aspect < 1.8 and height <= 1080 and duration > 300:
            # Many podcasts use stereo (2 channels) with static background
            return VideoSubType.PODCAST

    # Lecture: 16:9, less common codec, mostly static
    if has_audio and codec in ("h264", "hevc", "h265") and height >= 720:
        return VideoSubType.LECTURE

    # Animation: matching by resolution patterns (cartoons often 23.976fps)
    if has_audio and 23 < fps < 25 and codec:
        return VideoSubType.ANIMATION

    return VideoSubType.STANDARD


# ============================================================================
# 6.  ML-based scene type classifier (optional third layer, ~1-2 GB VRAM)
# ============================================================================

_ML_CLASSIFIER = None  # Module-level singleton for the ML model


def get_ml_classifier(model_name: str = "mobilenet_v3") -> Optional[object]:
    """Lazy-load a lightweight image classifier for video scene type detection.

    Uses ``torchvision``'s MobileNetV3-Small (~2.5M params, <100 MB VRAM) or
    ResNet-18 (~11M params, ~400 MB VRAM).  The model is loaded once and cached.
    Returns ``None`` if torchvision is not available or the model fails to load.

    Args:
        model_name: One of ``"mobilenet_v3"`` (default, 2.5M params) or
            ``"resnet18"`` (11M params, higher accuracy).

    Reference:
        - MobileNetV3: https://arxiv.org/abs/1905.02244
        - ResNet-18: https://arxiv.org/abs/1512.03385
    """
    global _ML_CLASSIFIER
    if _ML_CLASSIFIER is not None:
        return _ML_CLASSIFIER

    try:
        import torch
        import torchvision.models as models
        from torchvision import transforms
    except ImportError:
        logger.warning("torchvision not installed — ML classifier unavailable")
        return None

    try:
        if model_name == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            input_size = 224
        else:
            model = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            )
            input_size = 224

        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        preprocess = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # ImageNet class indices that map to video-scene-type labels
        scene_mapping = _build_scene_mapping()

        _ML_CLASSIFIER = {
            "model": model,
            "preprocess": preprocess,
            "device": device,
            "scene_mapping": scene_mapping,
            "model_name": model_name,
        }
        logger.info(
            f"ML classifier loaded: {model_name} on {device} "
            f"(~{'100 MB' if model_name == 'mobilenet_v3' else '400 MB'} VRAM)"
        )
        return _ML_CLASSIFIER
    except Exception as e:
        logger.warning(f"Failed to load ML classifier ({model_name}): {e}")
        return None


def _build_scene_mapping() -> Dict[int, str]:
    """Map ImageNet class IDs to human-readable video-scene-type labels.

    Only a subset of ImageNet-1K classes are relevant for video scene
    classification.  This mapping covers the most common cases.
    """
    return {
        404: "airliner",
        409: "airplane",
        436: "beach",
        443: "bicycle",
        466: "bookcase",
        468: "bookshop",
        480: "car_wheel",
        483: "castle",
        497: "church",
        505: "computer_keyboard",
        506: "computer_monitor",
        507: "computer_mouse",
        509: "confectionery",
        510: "conference_room",
        511: "consomme",
        523: "countertop",
        525: "courtroom",
        526: "cowboy_hat",
        534: "crossword_puzzle",
        539: "dais",
        554: "desktop_computer",
        558: "digital_clock",
        567: "dome",
        569: "doormat",
        571: "drawbridge",
        574: "dumbbell",
        575: "dust_bin",
        577: "earthenware",
        579: "electric_fan",
        580: "electric_guitar",
        585: "entertainment_center",
        590: "fence",
        591: "ferris_wheel",
        601: "folding_chair",
        603: "football_helmet",
        604: "forecourt",
        608: "fountain",
        610: "freight_car",
        618: "gas_pump",
        619: "gas_stove",
        627: "golf_ball",
        628: "golf_cart",
        633: "grand_piano",
        634: "greenhouse",
        636: "grille",
        639: "gym",
        640: "hair_spray",
        641: "half_track",
        646: "hand-held_computer",
        647: "handkerchief",
        650: "harmonica",
        654: "head_cabbage",
        656: "headlight",
        660: "hip",
        662: "home_theater",
        664: "horizontal_bar",
        665: "horn",
        668: "hot_pot",
        669: "hot_tub",
        673: "ice_lolly",
        675: "indoor",
        676: "iPod",
        679: "jeep",
        683: "jukebox",
        688: "knee_pad",
        690: "lab_coat",
        691: "ladder",
        694: "lampshade",
        696: "laptop",
        697: "lawn_mower",
        707: "lighthouse",
        708: "limousine",
        710: "lion",
        714: "lotion",
        717: "loudspeaker",
        719: "lumbermill",
        723: "magic_lantern",
        724: "magnetic_compass",
        725: "mailbag",
        726: "mailbox",
        727: "maillot",
        733: "marimba",
        736: "maze",
        737: "measuring_cup",
        739: "medicine_chest",
        740: "megalith",
        741: "microphone",
        743: "microwave",
        745: "military_uniform",
        746: "milk_can",
        749: "minibus",
        751: "miniskirt",
        752: "minivan",
        753: "missile",
        754: "mitten",
        755: "mixing_bowl",
        756: "mobile_home",
        757: "model_t",
        758: "modem",
        759: "monastery",
        760: "monitor",
        762: "monorail",
        763: "mortarboard",
        765: "mosque",
        766: "mosquito_net",
        767: "motor_scooter",
        768: "mountain_bike",
        769: "mountain_tent",
        770: "mouse",
        771: "mousetrap",
        772: "moving_van",
        773: "muzzle",
        774: "nail",
        775: "neck_brace",
        776: "necklace",
        777: "nipple",
        780: "notebook",
        781: "obelisk",
        784: "oil_filter",
        785: "oil_lamp",
        789: "organ",
        790: "oscilloscope",
        791: "overskirt",
        792: "ox",
        794: "packet",
        795: "paddle",
        796: "paddlewheel",
        797: "padlock",
        798: "paintbrush",
        799: "pajama",
        800: "palace",
        801: "panpipe",
        802: "paper_towel",
        804: "park_bench",
        805: "parking_meter",
        806: "parrot",
        807: "passenger_car",
        808: "patio",
        809: "pay_phone",
        810: "pedestal",
        811: "pencil_box",
        812: "pencil_sharpener",
        813: "perfume",
        814: "petri_dish",
        815: "photocopier",
        816: "pick",
        817: "pickelhaube",
        818: "picket_fence",
        819: "pickup",
        820: "pier",
        822: "piggy_bank",
        823: "pill_bottle",
        824: "pillow",
        826: "pineapple",
        827: "ping-pong_ball",
        828: "pinwheel",
        829: "pirate",
        830: "pitcher",
        831: "plane",
        832: "planetarium",
        833: "plastic_bag",
        836: "plate_rack",
        837: "platypus",
        838: "plow",
        839: "plunger",
        840: "polaroid",
        842: "pond",
        843: "pop_bottle",
        844: "popcorn",
        845: "porcupine",
        848: "pot",
        849: "potter's_wheel",
        850: "power_drill",
        851: "prayer_rug",
        852: "printer",
        853: "prison",
        854: "projectile",
        855: "projector",
        856: "puck",
        857: "punching_bag",
        858: "purse",
        859: "quill",
        860: "quilt",
        861: "racer",
        862: "racket",
        863: "radiator",
        864: "radio",
        865: "radio_telescope",
        866: "rain_barrel",
        867: "recreational_vehicle",
        868: "reel",
        869: "reflex_camera",
        870: "refrigerator",
        871: "remote_control",
        872: "restaurant",
        873: "revolver",
        874: "rifle",
        875: "rocking_chair",
        876: "rotisserie",
        877: "rubber_eraser",
        878: "rugby_ball",
        879: "ruler",
        880: "running_shoe",
        881: "safe",
        883: "sandal",
        884: "sarong",
        885: "sax",
        886: "scabbard",
        887: "scale",
        888: "school_bus",
        889: "schooner",
        890: "scoreboard",
        891: "screen",
        892: "screw",
        893: "screwdriver",
        894: "seat_belt",
        895: "sea_lion",
        896: "security",
        897: "sedan",
        898: "sewing_machine",
        901: "shelter",
        902: "shield",
        903: "shoe_shop",
        904: "shoji",
        905: "shopping_basket",
        906: "shopping_cart",
        907: "shovel",
        908: "shower_cap",
        909: "shower_curtain",
        910: "sidetalk",
        912: "sign",
        913: "silencer",
        914: "silk",
        915: "sink",
        916: "skateboard",
        917: "ski",
        919: "skyscraper",
        920: "sliding_door",
        922: "snorkel",
        923: "snow_leopard",
        924: "snowmobile",
        925: "snowplow",
        926: "soap_dispenser",
        928: "soccer_ball",
        929: "sock",
        930: "solar_dish",
        931: "solar_heater",
        932: "solar_panel",
        934: "sombrero",
        935: "sorrel",
        937: "space_bar",
        938: "space_heater",
        940: "spatula",
        941: "speaker",
        942: "spear",
        943: "spectacles",
        944: "speed_boat",
        946: "spider_web",
        947: "spindle",
        948: "spoon",
        949: "sports_car",
        950: "spotlight",
        951: "stage",
        952: "steam_locomotive",
        953: "steel_arch_bridge",
        954: "steel_drum",
        955: "stethoscope",
        956: "stew",
        957: "stirrup",
        958: "stockpot",
        959: "stole",
        960: "stone_wall",
        961: "stopwatch",
        962: "stove",
        963: "strainer",
        964: "strawberry",
        965: "street_sign",
        966: "streetcar",
        967: "stretcher",
        968: "studio_couch",
        969: "stupa",
        970: "submarine",
        971: "suit",
        972: "sundial",
        973: "sunglass",
        974: "sunglasses",
        975: "sunscreen",
        976: "suspension_bridge",
        977: "swab",
        978: "sweatshirt",
        979: "swimming_trunks",
        980: "swing",
        981: "switch",
        982: "syringe",
        983: "table_lamp",
        984: "tank",
        985: "tape_player",
        986: "teapot",
        987: "teddy_bear",
        989: "telephone",
        990: "telescope",
        991: "television",
        992: "tennis_ball",
        993: "thatch",
        994: "theater_curtain",
        995: "thimble",
        996: "thresher",
        997: "throne",
        998: "tile_roof",
        999: "toaster",
        1000: "tobacco_shop",
    }


def classify_frame_with_ml(image_path: Path, classifier: Optional[Dict] = None) -> Dict:
    """Classify a single video frame using the ML classifier.

    Args:
        image_path: Path to a frame image (JPEG/PNG).
        classifier: The result of :func:`get_ml_classifier()`. If ``None``,
            the classifier is lazy-loaded with default settings.

    Returns:
        A dict with ``label`` (str) and ``confidence`` (float). Returns
        ``{"label": "", "confidence": 0.0}`` if classification fails.
    """
    if classifier is None:
        classifier = get_ml_classifier()

    if classifier is None:
        return {"label": "", "confidence": 0.0}

    try:
        import torch
        from PIL import Image

        model = classifier["model"]
        preprocess = classifier["preprocess"]
        device = classifier["device"]
        scene_mapping = classifier["scene_mapping"]

        img = Image.open(image_path).convert("RGB")
        input_tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top_prob, top_idx = torch.topk(probabilities, 1)

        class_id = top_idx.item()
        confidence = top_prob.item()
        label = scene_mapping.get(class_id, f"imagenet_{class_id}")

        return {"label": label, "confidence": round(confidence, 4)}
    except Exception as e:
        logger.warning(f"ML frame classification error: {e}")
        return {"label": "", "confidence": 0.0}


# ============================================================================
# 7.  Stage selection based on classification
# ============================================================================

# Default stage map: for each MediaType and VideoSubType, which pipeline
# stages to run (True = run, False = skip).
#
# Extensible: users can override via config to fine-tune per deployment.
DEFAULT_STAGE_MAP: Dict[str, Dict[str, bool]] = {
    "video": {
        # Baseline for all video files
        "_default": {
            "scene_detection": True,
            "frame_extraction": True,
            "quality_screening": True,
            "transcription": True,
            "speaker_diarization": True,
            "object_detection": True,
            "ocr": True,
            "clip_classification": True,
            "action_recognition": False,
            "video_mllm": False,
            "sprite_sheet": True,
            "rag_indexing": True,
        },
        VideoSubType.SCREENCAST: {
            "object_detection": False,  # Screenshots rarely benefit
            "action_recognition": False,
            "ocr": True,  # OCR is very useful for screencasts
            "scene_detection": True,
            "frame_extraction": True,
            "quality_screening": False,  # Screencasts are clean by nature
        },
        VideoSubType.LECTURE: {
            "action_recognition": False,
            "object_detection": False,
            "clip_classification": False,
            "ocr": True,
            "scene_detection": True,
            "frame_extraction": True,
            "quality_screening": True,
        },
        VideoSubType.PODCAST: {
            "object_detection": False,
            "action_recognition": False,
            "clip_classification": False,
            "ocr": False,
            "scene_detection": True,
            "frame_extraction": True,
            "quality_screening": False,
        },
        VideoSubType.TALKING_HEAD: {
            "object_detection": False,
            "action_recognition": False,
            "clip_classification": False,
            "ocr": False,
            "scene_detection": True,
            "frame_extraction": True,
            "quality_screening": False,
        },
        VideoSubType.MUSIC_VIDEO: {
            "object_detection": True,
            "action_recognition": True,
            "clip_classification": True,
            "ocr": False,
            "scene_detection": True,
            "frame_extraction": True,
        },
        VideoSubType.SPORTS: {
            "object_detection": True,
            "action_recognition": True,
            "clip_classification": True,
            "ocr": False,
            "scene_detection": True,
            "frame_extraction": True,
            "quality_screening": True,
        },
        VideoSubType.SHORT: {
            "scene_detection": False,  # No need for scenes in <30s clips
            "frame_extraction": True,
            "sprite_sheet": False,
            "rag_indexing": True,
            "quality_screening": False,
        },
        VideoSubType.SECURITY: {
            "object_detection": True,
            "action_recognition": False,
            "ocr": False,
            "clip_classification": False,
            "scene_detection": True,
            "frame_extraction": True,
            "quality_screening": False,
            "sprite_sheet": False,
            "rag_indexing": False,
            "transcription": False,  # No speech
            "speaker_diarization": False,
        },
        VideoSubType.ANIMATION: {
            "object_detection": False,  # YOLO doesn't work well on cartoons
            "clip_classification": False,
            "ocr": False,
            "action_recognition": False,
            "scene_detection": True,
            "frame_extraction": True,
        },
    },
    "audio": {
        "_default": {
            "scene_detection": False,
            "frame_extraction": False,
            "quality_screening": False,
            "object_detection": False,
            "ocr": False,
            "clip_classification": False,
            "action_recognition": False,
            "video_mllm": False,
            "sprite_sheet": False,
            "rag_indexing": False,
            "transcription": True,
            "speaker_diarization": True,
        },
    },
    "image": {
        "_default": {
            # Images skip everything — not video pipeline.
            "scene_detection": False,
            "frame_extraction": False,
            "quality_screening": False,
            "object_detection": False,
            "ocr": False,
            "clip_classification": False,
            "action_recognition": False,
            "video_mllm": False,
            "sprite_sheet": False,
            "rag_indexing": False,
            "transcription": False,
            "speaker_diarization": False,
        },
    },
    "unknown": {
        "_default": {
            # Unknown — run only audio stages as a safe default.
            "scene_detection": False,
            "frame_extraction": False,
            "quality_screening": False,
            "object_detection": False,
            "ocr": False,
            "clip_classification": False,
            "action_recognition": False,
            "video_mllm": False,
            "sprite_sheet": False,
            "rag_indexing": False,
            "transcription": True,
            "speaker_diarization": True,
        },
    },
}

# All known stage names (for validation)
ALL_STAGE_NAMES: Set[str] = {
    "scene_detection",
    "frame_extraction",
    "quality_screening",
    "transcription",
    "speaker_diarization",
    "object_detection",
    "ocr",
    "clip_classification",
    "action_recognition",
    "video_mllm",
    "sprite_sheet",
    "rag_indexing",
}


def get_active_stages(
    classification: VideoClassification,
    stage_map: Optional[Dict[str, Dict[str, bool]]] = None,
) -> Set[str]:
    """Determine which pipeline stages to run based on video classification.

    Merges the ``_default`` stage map for the media type with any subtype-specific
    overrides.  Subtype overrides take precedence over defaults.

    Args:
        classification: Result of the file-type classification pipeline.
        stage_map: Optional custom stage map. Falls back to
            :data:`DEFAULT_STAGE_MAP`.

    Returns:
        A set of stage names that should be **skipped** (matching the existing
        ``_get_skipped_stages()`` interface in ``VideoPipeline``).
    """
    if stage_map is None:
        stage_map = DEFAULT_STAGE_MAP

    media_key = classification.media_type.value
    subtype_key = classification.video_subtype.value

    # Get the base stage map for this media type
    media_map = stage_map.get(media_key, stage_map.get("unknown", {}))
    defaults: Dict[str, bool] = media_map.get("_default", {})

    # Apply subtype overrides
    subtype_overrides: Dict[str, bool] = media_map.get(subtype_key, {})

    merged: Dict[str, bool] = dict(defaults)
    merged.update(subtype_overrides)

    # Return stages that should be SKIPPED (matching the existing interface)
    skipped = {name for name, enabled in merged.items() if not enabled}
    return skipped


# ============================================================================
# 8.  Orchestrator: run all three layers and merge results
# ============================================================================


def classify_file(
    filepath: Path,
    use_ml: bool = False,
    ml_model_name: str = "mobilenet_v3",
    stage_map: Optional[Dict] = None,  # type: ignore[type-arg]
) -> VideoClassification:
    """Run the full file-type classification pipeline.

    Classification proceeds in layers, each falling back to the next:

    1. **Extension heuristic** — zero I/O, pure suffix mapping.
    2. **Content sniffing (ffprobe)** — reads stream metadata from the file header.
    3. **ML classification** (optional) — classifies the first video frame.

    Args:
        filepath: Path to the input file.
        use_ml: Whether to run the ML-based frame classifier (requires
            torchvision and ~1-2 GB VRAM).
        ml_model_name: Which ML model to use (``"mobilenet_v3"`` or
            ``"resnet18"``).
        stage_map: Optional custom stage selection map.

    Returns:
        A :class:`VideoClassification` dataclass with all results.
    """
    filepath = Path(filepath)
    result = VideoClassification(file_extension=filepath.suffix.lower())

    # --- Layer 1: Extension heuristic ---
    result.media_type = classify_by_extension(filepath)

    # --- Layer 2: Content sniffing (ffprobe) ---
    raw = sniff_with_ffprobe(filepath)
    if raw is not None:
        result.raw_ffprobe = raw
        info = parse_ffprobe_output(raw)
        result.has_video_stream = info["has_video"]
        result.has_audio_stream = info["has_audio"]
        result.codec = info["codec"]
        result.audio_codec = info["audio_codec"]
        result.width = info["width"]
        result.height = info["height"]
        result.duration_seconds = info["duration"]
        result.fps = info["fps"]

        # Override extension-based media type with ground truth from ffprobe
        if info["has_video"]:
            result.media_type = MediaType.VIDEO
        elif info["has_audio"] and not info["has_video"]:
            result.media_type = MediaType.AUDIO

        # Subtype heuristic (rule-based, video only)
        if result.media_type == MediaType.VIDEO and info["has_video"]:
            result.video_subtype = classify_video_subtype(info, result.duration_seconds)

    # --- Layer 3: ML-based frame classification (optional) ---
    if use_ml and result.media_type == MediaType.VIDEO and result.has_video_stream:
        # Extract a single frame and classify it
        classifier = get_ml_classifier(ml_model_name)
        if classifier is not None:
            ml_result = _classify_first_frame(filepath, classifier)
            result.ml_label = ml_result["label"]
            result.ml_confidence = ml_result["confidence"]

    return result


def _classify_first_frame(filepath: Path, classifier: Dict) -> Dict:
    """Extract the first frame of a video and classify it with the ML model.

    Uses ffmpeg to grab the very first frame to avoid costly seeking.
    """
    import tempfile

    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(filepath),
                "-vframes",
                "1",
                "-qscale:v",
                "2",
                "-f",
                "image2",
                str(tmp_path),
            ],
            capture_output=True,
            timeout=30,
            check=True,
        )

        ml_result = classify_frame_with_ml(tmp_path, classifier)
        tmp_path.unlink(missing_ok=True)
        return ml_result
    except Exception as e:
        logger.warning(f"Failed to classify first frame: {e}")
        return {"label": "", "confidence": 0.0}


# ============================================================================
# 9.  Integration helper: wrap existing _get_skipped_stages interface
# ============================================================================


def pipeline_skipped_stages(
    filepath: Path,
    processing_mode: str = "auto",
    stage_map: Optional[Dict] = None,
    use_ml: bool = False,
) -> Set[str]:
    """Compute pipeline skipped stages using auto-detection or explicit mode.

    This is the main integration point for ``VideoPipeline._get_skipped_stages()``.

    Args:
        filepath: Path to the input file.
        processing_mode: ``"video_full"``, ``"audio_only"``, or ``"auto"``.
            In ``"auto"`` mode, the file is classified to determine the
            appropriate stages.
        stage_map: Optional custom stage selection map.
        use_ml: Whether to use ML-based frame classification for ``auto`` mode.

    Returns:
        Set of stage names to **skip** (same interface as
        ``VideoPipeline._get_skipped_stages()``).
    """
    if processing_mode == "video_full":
        return set()  # Run everything
    if processing_mode == "audio_only":
        # Skip all visual stages
        return {
            "scene_detection",
            "frame_extraction",
            "quality_screening",
            "object_detection",
            "ocr",
            "clip_classification",
            "video_mllm",
            "action_recognition",
            "sprite_sheet",
            "rag_indexing",
        }

    # Auto mode: classify and select
    classification = classify_file(filepath, use_ml=use_ml)
    return get_active_stages(classification, stage_map=stage_map)
