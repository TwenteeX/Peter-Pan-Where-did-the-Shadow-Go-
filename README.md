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
│                        Peter Pan — Data Flow                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Layer 0 (Hardware)                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────────────┐│
│  │ RGB Camera   │  │ Depth (opt.) │  │ Projector(s) ← Render Engine (UE/…)  ││
│  └──────┬───────┘  └──────┬───────┘  └──────────────────────────────────────┘│
│         │                 │                            ▲                     │
│         ▼                 ▼                            │                     │
│  ┌─────────────────────────────────────┐    ┌─────────┴─────────┐           │
│  │ Layer 1: Perception Hub             │    │ Layer 4:          │           │
│  │ • Shadow mask (2D contour)          │    │ Render & Mapping  │           │
│  │ • Semantic label (VLM)              │───▶│ • OSC / WebSocket  │           │
│  │ • Optional: 3D point cloud          │    │ • Animation/pose  │           │
│  └──────────────┬──────────────────────┘    └─────────▲─────────┘           │
│                 │                                      │                     │
│                 ▼                                      │                     │
│  ┌─────────────────────────────────────┐    ┌─────────┴─────────┐           │
│  │ Layer 2: World Engine               │    │ Layer 3:          │           │
│  │ • Global 3D coords                  │───▶│ Agent Brain       │           │
│  │ • Surfaces (ground, climbable)      │    │ • Action intents   │           │
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
├── examples/                   # Sample environment.json / reasoning_output.json
├── docs/
│   └── SPRINT_MVP.md           # Weekly sprint: shadow + reasoning chain
└── scripts/
    ├── run_mvp_bridge.py       # Phase 4: perception → agent → OSC
    ├── run_shadow_demo.py      # Sprint: camera → shadow → env → reasoning + viz
    └── run_reasoning_offline.py
```

---

## Weekly Sprint — Shadow contour + reasoning agent (2D MVP)

This iteration implements the PRD chain:

**Camera frame → `shadow_detector` → `environment.json` → `RuleBasedReasoningAgent` → `reasoning_output.json` → OpenCV visualization.**

- **Goal A (shadow)**: Lab luminance vs background, ROI mask, morphology, contours → polygon / bbox / centroid in **table coordinates**, JSON-serializable; debug panels (raw, mask, overlay, path).
- **Goal B (agent)**: Deterministic **rule-based** planner (no LLM in the loop): priorities — edge safety → climbable books → shadow-biased A* paths. Behaviors include `walk_to_waypoint`, `avoid_edge`, `approach_book_edge`, `climb_book`, `wait_in_shadow`. Each decision logs to `outputs/reasoning_logs/`.

```bash
pip install -r requirements.txt
# Live demo (press b = background, s = save JSON, q = quit)
python -m scripts.run_shadow_demo
# Reasoning only from examples
python -m scripts.run_reasoning_offline examples/environment.json
```

Tune `shadow_detector` and `environment_builder` in `config.yaml` (table ROI, book polygons, agent start, goal). See `docs/SPRINT_MVP.md` for the flow diagram and checklist.

---

## MVP Setup (Phases 1–4)

### Phase 1: Hardware

- One projector (vertical or angled onto a table).
- One webcam or HD camera (above or beside projector).
- One workstation with a GPU (for future VLM/rendering).

### Phase 2: Perception (Layer 1)

- **Image capture**: `peter_pan.perception.camera_shadow.CameraShadowPerception` uses OpenCV to read the camera and optional background subtraction for shadow contour.
- **VLM**: Call your preferred API (OpenAI / Hugging Face) with a single frame and store a `SemanticLabel`; set it via `set_semantic_from_vlm()` for the agent.

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

- **perception**: `camera_index`, image size, `shadow.method` (e.g. `background_subtraction`), `vlm` provider/model/prompt.
- **shadow_detector** / **environment_builder** / **reasoning_agent**: sprint 2D pipeline (ROI, thresholds, manual objects, planner margins).
- **render_bridge**: `transport` (`osc` or `websocket`), `osc.host` / `osc.port` or `websocket.url`.

Defaults live in `peter_pan.config.loader`; only override what you need.

---

## Interfaces for Extensions

- **Perception**: Implement `PerceptionHubInterface` (e.g. add SAM, depth, or another camera).
- **World**: Implement `WorldEngineInterface` (e.g. Open3D point cloud → surfaces).
- **Agent**: Implement `AgentBrainInterface` (e.g. LLM-based planner with full state).
- **Render**: Implement `RenderBridgeInterface` (e.g. WebSocket instead of OSC).

Data types are in `peter_pan.protocols`: `PerceptionOutput`, `WorldState`, `ActionIntent`, `RenderCommand`. Serialization (e.g. JSON/OSC) is defined on these classes.

---

## License and References

- Concept: *Peter Pan, Where did the Shadow Go?* — generative experience in physical space, shadow as agent.
- TRD: Modularity, hardware agnosticism, screen-less immersion; MVP from single-camera + single projector.
