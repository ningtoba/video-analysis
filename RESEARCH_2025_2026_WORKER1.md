# Research: Real-Time Communication (RTC) & WebRTC for Video Analysis (2025-2026)

Worker 1 report - WebRTC frame streaming, MLLM token streaming, Gradio 6 real-time, FastAPI SSE, and open-source RTC+ML projects.
Compiled: 2026-06-27

---

## 1. WebRTC-Based Frame Streaming: Browser to ML Pipeline

### FastRTC (gradio-app/fastrtc) - 4.6k stars - RECOMMENDED

The official Gradio companion library for WebRTC streaming. Turns any Python function that accepts a numpy array (video frame) into a real-time WebRTC stream endpoint.

- Repo: https://github.com/gradio-app/fastrtc
- Docs: https://fastrtc.org
- pip install fastrtc
- Key features:
  - Stream(handler=fn, modality="video", mode="send-receive") - handler receives frames as numpy arrays
  - Automatic Gradio UI via .ui.launch() - zero-effort web frontend
  - FastAPI .mount(app) integration - mount on existing FastAPI server
  - WebSocket fallback mode
  - Gradio 6 compatible

Example:
    from fastrtc import Stream
    import gradio as gr

    def process_frame(frame, conf_threshold=0.3):
        result = model.detect(frame, conf_threshold)
        return result

    stream = Stream(
        handler=process_frame,
        modality="video",
        mode="send-receive",
        additional_inputs=[gr.Slider(minimum=0, maximum=1, step=0.01, value=0.3)]
    )
    stream.ui.launch()

### aiortc (aiortc/aiortc) - 5.1k stars - For custom WebRTC servers

Full SDP negotiation, ICE/STUN/TURN, VP8/H.264 encode/decode, data channels in Python.

- Repo: https://github.com/aiortc/aiortc
- pip install aiortc
- Webcam example: aiortc/examples/webcam - serves webcam video via WebRTC
- Subclass MediaStreamTrack and override recv() to run inference before forwarding
- Pros: Complete control, no Gradio, works with any frontend
- Cons: More boilerplate, manual signaling server

### Juturna (Meetecho) - Newest entrant (Oct 2025)

From creators of Janus WebRTC Server. Pipeline-based RTP media processor in Python.
- Define pipelines as JSON nodes (source -> processing -> sink)
- RTP-native (works with Janus, GStreamer), plugin architecture
- Very new, requires Janus server

### Modal's WebRTC + YOLO Example

- aiortc on Modal's serverless infra with YOLO real-time detection
- Key: "2-4ms inference, RTT below video frame rates (~30ms)"

---

## 2. MLLM Inference Streaming (Token-by-Token SSE)

### vLLM - OpenAI-Compatible Streaming

OpenAI-compatible API server supports stream=True for vision models (llava, qwen2-vl, pixtral).

- SSE format: text/event-stream
- Vision: pass image_url (base64) in content array
- RTX 4070: ~20-40 tok/s with Qwen2-VL-7B FP8
- Key: Vision prefill is non-streaming, only decode phase streams

### Ollama - Simpler Setup

- Supports stream=True with vision models (llava, llama3.2-vision, qwen2-vl)
- Simpler but less performant than vLLM

### Summary

| Engine    | Vision Support           | SSE   | RTX 4070 | Complexity |
|-----------|-------------------------|-------|----------|------------|
| vLLM      | Qwen2-VL, Llava, Pixtral| Yes   | FP8      | Medium     |
| Ollama    | Llava, LLaMA-Vision     | Yes   | 4-bit    | Easy       |
| TGI       | Llava-Next, Idefics3    | Yes   | Larger   | Medium     |
| llama.cpp | Llava via server        | Yes   | Q4       | Medium     |

---

## 3. Gradio 6 Real-Time Components

### gr.Image with streaming=True

- Available since Gradio 4.x, still supported in Gradio 6
- Pattern: gr.Image(source="webcam", streaming=True)
- Sends webcam frames as numpy arrays via .stream() event
- Limitation: JPEG-polling over HTTP, not WebRTC (~100-500ms vs ~10-50ms)

### gr.WebcamOptions (NEW in Gradio 6)

- Dedicated class for custom media constraints
- Controls: resolution, frame rate, front/rear camera
- Example: gr.Image(webcam_options=gr.WebcamOptions(width=640, height=480))

### FastRTC vs gr.Image(streaming=True)

| Feature       | gr.Image(streaming=True)   | fastrtc.Stream     |
|---------------|---------------------------|-------------------|
| Protocol      | HTTP multipart poll       | WebRTC (UDP)      |
| Latency       | 100-500ms                 | 10-50ms           |
| Frame rate    | Network dependent         | 30+ FPS           |
| Bidirectional | No                        | Yes               |
| Frontend      | Gradio only               | Gradio + FastAPI  |

---

## 4. FastAPI + SSE for Streaming MLLM Responses

### Best Practices

1. X-Accel-Buffering: no - critical behind nginx
2. StreamingResponse with async generator - don't block event loop
3. Run frame encoding in thread pool (run_in_executor)
4. Rate limit: 1-2 FPS for 7B MLLM on RTX 4070
5. Check await request.is_disconnected() for client disconnect
6. Add CORS middleware for cross-origin frontends

### Frame Rate Strategy

MLLMs cannot process 30 FPS. Use frame selection (motion-based, key frame detection, temporal sampling) to reduce to 1-2 FPS. YOLO/object detection runs at full rate (~2-4ms per frame) in parallel.

---

## 5. Open-Source: WebRTC + Local ML for Video Analysis

### FastRTC + YOLOv10 (4.6k stars) - Most relevant
- WebRTC + YOLOv10 ONNX + Gradio 6
- ~30 FPS on consumer GPU
- Swap YOLO for full video analysis pipeline

### Pipecat (pipecat-ai/pipecat, 13.1k stars)
- Real-time voice and multimodal agent framework
- WebRTC transports: Daily, LiveKit, SmallWebRTCTransport
- No generic video frame -> MLLM pipeline
- Voice agent focus

### LiveKit Agents
- Server-side agent framework for LiveKit WebRTC
- Can process video tracks
- Overkill for single RTX 4070 self-hosted

### Juturna (Meetecho) - Watch for 2026
- RTP media processing pipelines in Python
- Ideal for IP camera / RTSP -> ML
- Too new (Oct 2025)

### aiortc + Custom ML (5.1k stars)
- Many community projects: YOLO+aiortc, WebRTC+TensorFlow
- Pattern: Subclass MediaStreamTrack, override recv(), call ML

---

## Architecture Recommendations

### Option A: FastRTC + Existing VideoPipeline (Recommended for Webcam)

Browser Webcam -> FastRTC Stream -> Frame Handler
                                       -> YOLO/CLIP at full rate -> annotated frame via WebRTC
                                       -> MLLM at 1-2 FPS via SSE -> streaming description

Pros:
- FastRTC from Gradio team, maintained, works with Gradio 6
- Reuses existing YOLO/CLIP models
- .mount(app) integrates with existing FastAPI
- Zero frontend work

### Option B: aiortc for Custom WebRTC Server

Subclass MediaStreamTrack, override recv(), call ML inference, forward annotated frame.

### MLLM Streaming Integration

Existing project already has:
- video_mllm.py - SmolVLM2, VideoChat-Flash, Qwen3-VL backends
- llm_provider.py - OpenAI-compatible provider

New class:
1. Receives frame at 1-2 FPS from FastRTC handler
2. Encodes frame to base64 JPEG
3. Calls vLLM/Ollama with stream=True
4. Yields tokens via SSE to browser

### Key Dependencies to Add

fastrtc>=0.1.0    # WebRTC streaming for Gradio
openai>=1.0.0      # OpenAI-compatible API client

No additional GPU models needed.
