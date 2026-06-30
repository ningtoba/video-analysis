"""Real-time video stream analysis engine.

Processes live RTSP/webcam feeds and uploaded video files through a
frame sampler, motion detector, and LLM Vision scheduler.
"""

from video_analysis.stream.engine import StreamEngine as StreamEngine
from video_analysis.stream.source import FileSource as FileSource
from video_analysis.stream.source import open_source as open_source
from video_analysis.stream.sampler import CircularFrameBuffer as CircularFrameBuffer
from video_analysis.stream.sampler import FrameSampler as FrameSampler
from video_analysis.stream.sampler import SampledFrame as SampledFrame
from video_analysis.stream.motion import MotionDetector as MotionDetector
from video_analysis.stream.analyzer import LLMAnalyzer as LLMAnalyzer
from video_analysis.stream.store import EventStore as EventStore
from video_analysis.stream.store import TimelineEvent as TimelineEvent
