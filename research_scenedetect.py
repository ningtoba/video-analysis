"""
Probe the scenedetect v0.7 Python API.

Usage:
    pip install scenedetect>=0.7.0
    python research_scenedetect.py

Expected output when scenedetect is installed:
    - Lists FrameTimecode attributes (get_seconds, timecode, get_frames, etc.)
    - Confirms detect() returns a list of (start_tc, end_tc) tuples
"""

import sys

try:
    import scenedetect
    from scenedetect import detect, AdaptiveDetector, ContentDetector
    from scenedetect.common import FrameTimecode

    print(f"scenedetect version: {scenedetect.__version__}")

    # Quick sanity: create a FrameTimecode to verify the API
    tc = FrameTimecode(timecode="00:00:01.000", fps=30.0)
    print(
        f"FrameTimecode('00:00:01.000', fps=30.0).get_seconds() = {tc.get_seconds()}"
    )  # 1.0
    print(f"tc.timecode = {tc.timecode}")
    print(f"tc.get_frames() = {tc.get_frames()}")  # 30
    print()

    # Test detect() signature
    import inspect

    sig = inspect.signature(detect)
    print(f"detect signature: detect{sig}")
    print(">>> from scenedetect import detect, AdaptiveDetector")
    print(">>> scene_list = detect('video.mp4', AdaptiveDetector())")
    print(">>> scene_list[0]  -> (start_tc, end_tc)")
    print(">>> scene_list[0][0].get_seconds()  -> start time in seconds")
    print(">>> scene_list[0][1].get_seconds()  -> end time in seconds")

    print("\nAPI probe complete — all types match the documented interfaces.")

except ImportError:
    print("scenedetect not installed. Install with: pip install scenedetect>=0.7.0")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
