# Peter Pan, Where did the Shadow Go?

Interactive installation framework exploring **Generative Experience** in physical space: by severing the optical link between an object and its shadow, the shadow becomes an autonomous digital agent. The stack uses Visual Language Models (VLM), real-time perception, and projection mapping to drive a screen-less, material-aware experience.

This repository provides a **modular, hardware-agnostic** base framework and clear interfaces so you can plug in different sensors, engines, and AI backends without rewriting the pipeline.

---

## Design Principles

- **Modularity** — Perception, World Modeling, Agent, and Render are separate modules; they communicate via standardized data contracts (see `peter_pan.protocols`).
- **Hardware agnostic** — No binding to specific camera or projector models; only input/output data formats are fixed.
- **Screen-less immersion** — Output is defined for physical projection (e.g. OSC/WebSocket to Unreal nDisplay or Unity), not traditional UI.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Peter Pan — Data Flow                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  Layer 0 (Hardware)                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────────────────────┐│
│  │ RGB Camera   │  │ Depth (opt.) │  │ Projector(s) ← Render Engine (UE/…) ││
│  └──────┬───────┘  └──────┬───────┘  └─────────────────────────────────────┘│
│         │                 │                           ▲                     │
│         ▼                 ▼                           │                     │
│  ┌─────────────────────────────────────┐    ┌─────────┴─────────┐           │
│  │ Layer 1: Perception Hub             │    │ Layer 4:          │           │
│  │ • Shadow mask (2D contour)          │    │ Render & Mapping  │           │
│  │ • Semantic label (VLM)              │───▶│ • OSC / WebSocket │          │
│  │ • Optional: 3D point cloud          │    │ • Animation/pose  │           │
│  └──────────────┬──────────────────────┘    └─────────▲─────────┘           │
│                 │                                     │                     │
│                 ▼                                     │                     │
│  ┌─────────────────────────────────────┐    ┌─────────┴─────────┐           │
│  │ Layer 2: World Engine               │    │ Layer 3:          │           │
│  │ • Global 3D coords                  │───▶| Agent Brain       │          │
│  │ • Surfaces (ground, climbable)      │    │ • Action intents  │           │
│  └─────────────────────────────────────┘    │ • awaken / climb  │           │
│                                             └───────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Four abstract modules** (see TRD):

| Module             | Responsibility                    | Input                    | Output                          |
|--------------------|-----------------------------------|--------------------------|---------------------------------|
| **Perception Hub** | Capture + shadow + semantics      | RGB stream, depth (opt.) | Mask, semantic JSON, point cloud|
| **World Engine**   | Digital twin of physical space    | Mesh, calibration        | Global 3D, surfaces             |
| **Agent Brain**    | Decide shadow’s next behavior     | State, obstacles         | Action intents (e.g. climb)     |
| **Render & Mapping** | Drive animation, send to projectors | Action intents, 3D   | OSC/WS → engine                 |

Communication between Python and the render engine is via **OSC** (default) or **WebSocket**; the engine (Unreal, Unity, TouchDesigner) listens and drives the animated shadow.

---

## Project Layout

```
Hypersense/
├── README.md
├── requirements.txt
├── config.yaml                 # Hardware-agnostic config (override here)
├── peter_pan/
│   ├── __init__.py
│   ├── config/                 # Config loader
│   ├── protocols/              # I/O schemas (PerceptionOutput, ActionIntent, …)
│   ├── perception/             # Perception Hub (camera, shadow, VLM)
│   ├── world_engine/           # World Engine (stub + interface)
│   ├── agent_brain/            # Agent Brain (simple + interface)
│   └── render_bridge/          # OSC / WebSocket client
└── scripts/
    ├── run_mvp_bridge.py       # Phase 4: perception → agent → OSC
    └── …
```

---

## MVP Setup (Phases 1–4)

### Phase 1: Hardware

- One projector (vertical or angled onto a table).
- One webcam or HD camera (above or beside projector).
- One workstation with a GPU (for future VLM/rendering).

### Phase 2: Perception (Layer 1)

- **Image capture**: `peter_pan.perception.camera_shadow.CameraShadowPerception` supports `shadow.method=background_subtraction` (default), `shadow.method=detector_yolo` (fast YOLO-only multi-object boxes), and `shadow.method=detector_sam2` (YOLO box detection + SAM2 mask refinement).
- **VLM**: Call your preferred API (OpenAI / Hugging Face) with a single frame and store a `SemanticLabel`; set it via `set_semantic_from_vlm()` for the agent.
- **Tune + OSC check**: With the table empty, run `python -m scripts.debug_phase2_perception --preview` — a small Tk panel offers **「重設背景（基準畫面）」** and **「VLM 多物體辨識」**（對每個有效 `track_id` 各送一次 OpenAI，與預覽視窗鍵 `b` / `v` 相同）。`perception.vlm.max_objects_per_batch` 可限制單次最多幾個物體（費用 ≈ 物件數 × API 次數）。Use `python -m scripts.debug_phase2_perception --preview --no-tk` for OpenCV-only + keys `b`/`v`/`q`. For console-only + buttons: `--control-panel`. Then use `python -m scripts.osc_listen` in one terminal and `python -m scripts.run_mvp_bridge` in another; place then remove the object — you should see `/peter_pan/event/awaken` on the listener.
- **Recognizing *what* the object is** (not just motion vs background): see **Object recognition** below.

### Phase 3: Render engine “Hello World”

- In Unreal (or Unity) create a project with a virtual camera aligned to the table.
- Add a black, unlit shadow-like character (e.g. cat); output to the projector.
- No Python link yet.

### Phase 4: Bridge

- Start the render engine and an **OSC receiver** on host/port (e.g. `127.0.0.1:7000`).
- Run:

  ```bash
  pip install -r requirements.txt
  python -m scripts.run_mvp_bridge
  ```

- When the physical object is removed from the table, the script detects `object_visible -> False`, the agent outputs **awaken**, and the bridge sends an OSC message (e.g. `/peter_pan/event/awaken`). Map this in your engine to start the “standing” or “walking” animation.

---

## Configuration

Edit `config.yaml` (or pass a path to `load_config()`). Important sections:

- **perception**: `camera_index`, image size, `shadow.method` (`background_subtraction`, `detector_yolo`, or `detector_sam2`), `vlm` provider/model/prompt, `vlm.max_objects_per_batch` (Phase 2 多物體 VLM 上限)。
- **render_bridge**: `transport` (`osc` or `websocket`), `osc.host` / `osc.port` or `websocket.url`.

Defaults live in `peter_pan.config.loader`; only override what you need.

---

## Interfaces for Extensions

- **Perception**: Implement `PerceptionHubInterface` (e.g. add SAM, depth, or another camera).
- **World**: Implement `WorldEngineInterface` (e.g. Open3D point cloud → surfaces).
- **Agent**: Implement `AgentBrainInterface` (e.g. LLM-based planner with full state).
- **Render**: Implement `RenderBridgeInterface` (e.g. WebSocket instead of OSC).

Data types are in `peter_pan.protocols`: `PerceptionOutput`, `WorldState`, `ActionIntent`, `RenderCommand`. Serialization (e.g. JSON/OSC) is defined on these classes.

### Object recognition (what is in the image?)

Background subtraction only detects **change** vs a stored plate; it does not label “cat vs cup”. To attach **semantic identity** to the pipeline:

1. **VLM (vision-language model, built-in path)** — Set `perception.vlm.enabled: true` in `config.yaml`, install `openai` (`pip install -r requirements.txt`), and export **`OPENAI_API_KEY`**. In **auto** mode, `CameraShadowPerception` calls OpenAI on a **throttled interval** (`interval_sec`, default 3s) while `object_visible` is true, optionally cropping to the **largest contour** (`use_largest_contour_crop`). In **manual** mode, Phase 2 debug can run **multi-object VLM**: one API call per tracked detection (up to `max_objects_per_batch`); descriptions are stored per `track_id` (`get_vlm_semantic_for_track` / `snapshot_vlm_semantics_by_track`) and drawn on each box. Global `PerceptionOutput.semantic` still reflects the last completed VLM result. You can still override manually with `set_semantic_from_vlm()`.
2. **Detection / classification (on-device or API)** — e.g. Ultralytics YOLO, Roboflow, or Hugging Face `transformers` for DETR/ViT: run on each frame or on a slow cadence, map top class + box into your own fields or into `SemanticLabel.description` / `PerceptionOutput.extra`.
3. **Segmentation** — SAM (or SAM2) for a tight mask instead of diff contours; you’d extend or replace the shadow step while keeping `PerceptionHubInterface`.

Pick one path based on latency (real-time vs every N seconds), GPU, and whether you need arbitrary language (“a toy dinosaur”) vs fixed categories.

### Detector + SAM2 contour pipeline (implemented)

Use this when you want robust contours in cluttered scenes:

1. Set `config.yaml`:
   - `perception.shadow.method: detector_sam2`
   - `perception.shadow.detector.model: yolov8n.pt` (or your YOLO checkpoint)
   - `perception.shadow.sam2.model_cfg`: SAM2 config file path/name
   - `perception.shadow.sam2.checkpoint`: SAM2 checkpoint path
2. Install dependencies:
   - `pip install ultralytics`
   - `pip install git+https://github.com/facebookresearch/sam2.git`
3. Run debug script as usual (`python -m scripts.debug_phase2_perception --preview`).

The detector finds a candidate box, SAM2 refines the mask, and the largest refined contour is exported as `ShadowMaskOutput.polygon_xy / bounds_xyxy`.

### YOLO-only fast detection (implemented)

If SAM2 is too heavy, use YOLO-only boxes:

1. Set `perception.shadow.method: detector_yolo`
2. Configure detector options:
   - `perception.shadow.detector.model` (e.g. `yolov8n.pt`)
   - `perception.shadow.detector.min_conf`
   - `perception.shadow.detector.max_objects`
   - `perception.shadow.tracker.backend: bytetrack` (recommended) or `simple`
   - `perception.shadow.tracker.bytetrack_cfg` (default `bytetrack.yaml`)
   - `perception.shadow.tracker.*` for `simple` fallback (`iou_threshold`, `center_dist_px`, `max_age_frames`)
3. Run `python -m scripts.debug_phase2_perception --preview`

Debug log prints `detected=<N>`, class names, and tracked IDs. The top-confidence box is mapped to `ShadowMaskOutput.bounds_xyxy`, while all detections (including `track_id`) are in `ShadowMaskOutput.extra["detections"]`.

---

## License and References

- Concept: *Peter Pan, Where did the Shadow Go?* — generative experience in physical space, shadow as agent.
- TRD: Modularity, hardware agnosticism, screen-less immersion; MVP from single-camera + single projector.
