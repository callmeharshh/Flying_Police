# Flying Police — Code Documentation

**Version:** 1.0 | **Python:** 3.14 | **Framework:** LangChain 0.3 + GPT-4o-mini

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [End-to-End Pipeline Flow](#3-end-to-end-pipeline-flow)
   - [3.1 Full System](#31-full-system)
   - [3.2 Single Frame Decision Tree](#32-single-frame-decision-tree)
   - [3.3 ReAct Agent Loop](#33-react-agent-loop)
4. [Module Reference](#4-module-reference)
   - [config.py](#41-configpy)
   - [main.py](#42-mainpy)
   - [pipeline/](#43-pipeline)
   - [vlm/](#44-vlm)
   - [agent/](#45-agent)
   - [storage/](#46-storage)
   - [query/](#47-query)
   - [notifications/](#48-notifications)
   - [data/](#49-data)
   - [ui/](#410-ui)
5. [Alert Rules Reference](#5-alert-rules-reference)
6. [Storage Schema](#6-storage-schema)
   - [6.1 SQLite](#61-sqlite-dataeventsdb)
   - [6.2 ChromaDB](#62-chromadb-datachroma)
7. [Configuration Reference](#7-configuration-reference)
8. [Test Suite](#8-test-suite)
9. [Reference Videos](#9-reference-videos)
10. [Design Decisions](#10-design-decisions)
11. [Troubleshooting & Debugging](#11-troubleshooting-debugging)
12. [Extension Guide](#12-extension-guide)
13. [Scripts Reference](#13-scripts-reference)
14. [Environment Variables Reference](#14-environment-variables-reference)
15. [Module Import Graph](#15-module-import-graph)
16. [Known Limitations](#16-known-limitations)
17. [Performance Expectations](#17-performance-expectations)
18. [Query Result Interpretation](#18-query-result-interpretation)

---

<a id="1-project-overview"></a>
## 1. Project Overview

Flying Police processes surveillance video frame-by-frame to detect, classify, and alert on security events. It combines classical computer vision (OpenCV MOG2), a vision-language model (BLIP), and a reasoning agent (LangChain + GPT-4o-mini) with dual-database storage (SQLite + ChromaDB) and real-time Telegram notifications.

**Key properties:**
- No cloud vision API per frame — BLIP runs fully locally
- Alerts are deterministic (rules engine) — the LLM cannot suppress them
- Two databases answer different query types: SQL for structured lookups, ChromaDB for semantic search
- Stateless runs — each execution clears and rebuilds SQLite + ChromaDB from scratch

---

<a id="2-repository-structure"></a>
## 2. Repository Structure

```
```
Flying-Police/
├── config.py                          # All constants + env-var loading
├── main.py                            # CLI entry point
│
├── pipeline/
│   ├── video_processor.py             # Master pipeline: video → PipelineEvent stream
│   ├── frame_preview.py               # BGR frame → annotated RGB for UI
│   └── session.py                     # Per-run session directory management (UI)
│
├── vlm/
│   ├── background_subtractor.py       # MOG2 wrapper → ObjectCrop list
│   ├── blip_analyzer.py               # BLIP captioning + CropAnalysis
│   ├── lighting_detector.py           # Frame brightness → day/night classification
│   └── constants.py                   # Keyword taxonomies (object types, colors)
│
├── agent/
│   ├── alert_rules.py                 # Deterministic rules engine (RULE-01 to RULE-05)
│   ├── security_agent.py              # LangChain ReAct agent
│   ├── tools.py                       # @tool functions used by the agent
│   ├── vehicle_tracker.py             # Spatial motion tracker (cross-frame identity)
│   ├── vehicle_context.py             # TrackUpdate → agent context dict
│   └── callbacks.py                   # Patched LangChain callback handler
│
├── storage/
│   ├── event_store.py                 # SQLite wrapper (events + alerts)
│   └── frame_index.py                 # ChromaDB wrapper (vector index)
│
├── query/
│   ├── query_engine.py                # NL query router (SQL + ChromaDB)
│   └── track_merge.py                 # Union-find dedup of fragmented tracks
│
├── notifications/
│   └── telegram_notifier.py           # Telegram Bot API (fire-and-forget)
│
├── data/
│   ├── simulated_frames.py            # 7 hand-crafted test scenarios + FrameAnalysis dataclass
│   ├── telemetry.py                   # Drone telemetry (GPS, battery, ground coverage)
│   ├── validation_capture.py          # Golden fixture capture + manifest management
│   └── sample_video/                  # Input MP4 files (see §9)
│
├── ui/
│   └── app.py                         # Streamlit web interface
│
├── scripts/
│   ├── capture_validation_set.py      # CLI to generate validation fixtures
│   └── generate_architecture_ppt.py   # python-pptx slide generator
│
├── tests/
│   ├── conftest.py                    # pytest markers + fixture gate
│   └── unit/                          # 64 unit tests (no model/API calls)
│
├── requirements.txt                   # Core deps (opencv, torch, BLIP, chromadb, streamlit)
├── requirements-langchain.txt         # LangChain deps (installed with --no-deps)
├── .env                               # OPENAI_API_KEY, TELEGRAM_* (not committed)
└── README.md
```

---

<a id="3-end-to-end-pipeline-flow"></a>
## 3. End-to-End Pipeline Flow

<a id="31-full-system"></a>
### 3.1 Full System

```
┌──────────────────────────────────────────────────────────────────┐
│                         INPUT                                    │
│   MP4 video file  (any resolution, any fps)                      │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼  sample every Nth raw frame
┌──────────────────────────────────────────────────────────────────┐
│                    FRAME SAMPLING                                │
│   SAMPLE_EVERY_N_FRAMES = 2  →  ~50% of raw frames processed    │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                  ┌──────────┴──────────┐
                  │ first MOG2_WARMUP   │
                  │ frames (default: 5) │
                  │  → calibrate BG     │
                  │  → no detection     │
                  └──────────┬──────────┘
                             │ warmed up
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                BACKGROUND SUBTRACTION  (MOG2)                    │
│   • Foreground mask → morphological close                        │
│   • findContours → filter by area (2000px² – 40% of frame)      │
│   • Each surviving contour → ObjectCrop (PIL + bbox)             │
│   • Empty result → frame skipped (no BLIP, no rules, no agent)  │
└────────────────────────────┬─────────────────────────────────────┘
                             │ List[ObjectCrop]
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                    LIGHTING DETECTION                            │
│   • Central 65% crop of V-channel (HSV)                         │
│   • is_night = mean < 90 OR dark_pixel_ratio ≥ 0.55             │
└────────────────────────────┬─────────────────────────────────────┘
                             │ LightingAnalysis
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                    VLM LAYER  (BLIP)                             │
│   • analyze_crop() per ObjectCrop → CropAnalysis                │
│     – caption (free-form sentence)                               │
│     – object_type: person | vehicle | police_car | bicycle |     │
│                    animal | unknown                              │
│     – color (first matched COLOR_KEYWORD)                        │
│   • merge all CropAnalysis → FrameVLMResult                     │
│     – objects: deduplicated type list                            │
│     – raw_description: captions joined with "; "                 │
└────────────────────────────┬─────────────────────────────────────┘
                             │ FrameVLMResult
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                  MOTION TRACKER  (VehicleTracker)               │
│   • Match crop bbox center to active tracks by proximity        │
│   • max_match_distance = MAX_VELOCITY * gap * MARGIN            │
│   • Hit → continue track, update trajectory                     │
│   • Miss → create new track (new_entry = True)                  │
│   • Returns TrackUpdate (track_id, entry_count, trajectory, …)  │
└────────────────────────────┬─────────────────────────────────────┘
                             │ TrackUpdate
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│              ALERT RULES ENGINE  (deterministic)                 │
│   RULE-01  person  + is_night              → HIGH                │
│   RULE-02  unknown vehicle                 → MEDIUM              │
│   RULE-03  vehicle entry_count > limit     → MEDIUM              │
│   RULE-04  person loitering / stationary   → HIGH (×2 stages)   │
│   RULE-05  perimeter + is_night            → HIGH                │
│   Returns List[Alert]  — fires before LLM, cannot be overridden │
└────────────────────────────┬─────────────────────────────────────┘
                             │ List[Alert]
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                  LANGCHAIN ReAct AGENT                           │
│   Receives: frame context + pre_alerts + vehicle_context dict   │
│   Tools: log_event | trigger_alert | query_history |            │
│          query_track_positions                                   │
│   Memory: ConversationBufferWindowMemory(k=10)                  │
│   Returns: one-sentence summary                                  │
└────────┬───────────────────────────────────────────┬────────────┘
         │                                           │
         ▼                                           ▼
┌─────────────────────┐                 ┌───────────────────────────┐
│   SQLite            │                 │   ChromaDB                │
│   events table      │                 │   frames collection       │
│   alerts table      │                 │   384-dim HNSW cosine     │
└─────────────────────┘                 └───────────────────────────┘
         │                                           │
         └──────────────────┬────────────────────────┘
                            │  HIGH alerts only
                            ▼
              ┌─────────────────────────┐
              │   Telegram Bot API      │
              │   annotated JPEG photo  │
              │   fire-and-forget thread│
              └─────────────────────────┘
```

<a id="32-single-frame-decision-tree"></a>
### 3.2 Single Frame Decision Tree

```
sampled frame N
      │
      ├─► MOG2.apply() ──── empty? ──► yield skip event
      │                        │
      │                        NO
      │                        │
      ├─► lighting_detector ──► is_night flag
      │
      ├─► BLIP.analyze_frame() ──► FrameVLMResult
      │
      ├─► infer_activity(objects, caption) ──► "entering" | "loitering" | "stationary" | "clear"
      │
      ├─► VehicleTracker.update() ──► TrackUpdate
      │
      ├─► AlertRulesEngine.evaluate()
      │         │
      │         ├── HIGH alerts? ──► telegram_alert() in daemon thread
      │         └── any alerts? ──► log_alert() to SQLite
      │
      ├─► SecurityAgent.process() (if agent enabled)
      │         └── calls tools → log_event / trigger_alert / query_history
      │
      ├─► FrameIndex.add_frame() ──► ChromaDB
      │
      └─► yield PipelineEvent(kind="frame", …)
```

<a id="33-react-agent-loop"></a>
### 3.3 ReAct Agent Loop

```
Agent receives query string:
  "Frame 12 | main_gate | 2026-01-01T23:05:00
   Objects: ['person'] | Activity: loitering
   Caption: a man in a dark jacket standing near the gate
   Pre-fired alerts: [RULE-01 person at night HIGH]
   Vehicle context: {track_id: T-3, entry_count: 1, …}"
          │
          ▼
  Thought → Action (tool call) → Observation
          │
          ├── Usually: log_event(message) → "Event logged: evt_xxx"
          ├── Sometimes: query_history("person at gate") → top-3 past frames
          └── Rarely: trigger_alert(message, severity) → "Alert stored: alt_xxx"
          │
          ▼  (up to 6 iterations)
  Final Answer: one-sentence summary
```

---

<a id="4-module-reference"></a>
## 4. Module Reference

<a id="41-configpy"></a>
### 4.1 `config.py`

Central configuration — all tunable constants and environment-variable bindings. Calls `load_dotenv()` at import time so `.env` values are available to every downstream module.

| Constant | Type | Default | Purpose |
|---|---|---|---|
| `BASE_DIR` | `str` | `__file__` dir | Anchor for relative paths |
| `DATA_DIR` | `str` | `BASE_DIR/data` | Root for all persistent data |
| `SAMPLE_VIDEO_DIR` | `str` | `DATA_DIR/sample_video` | Input video files |
| `DEMO_VIDEO` | `str` | `entrance_area_720p.mp4` | Default CLI video |
| `EVENTS_DB_PATH` | `str` | `DATA_DIR/events.db` | SQLite file |
| `CHROMA_DIR` | `str` | `DATA_DIR/chroma` | ChromaDB persistence dir |
| `UI_SESSIONS_DIR` | `str` | `DATA_DIR/sessions` | Per-session data (UI) |
| `UI_UPLOADS_DIR` | `str` | `DATA_DIR/uploads` | Temporary upload storage (UI) |
| `SAMPLE_EVERY_N_FRAMES` | `int` | `2` | Frame sampling rate |
| `VALIDATION_FIXTURES_DIR` | `str` | `DATA_DIR/validation_fixtures` | Golden test captures |
| `VALIDATION_CAPTURE_EVERY_N` | `int` | `30` | Save every Nth sampled frame |
| `VALIDATION_MAX_CAPTURES` | `int` | `15` | Max saved captures per run |
| `BLIP_MODEL` | `str` | `Salesforce/blip-image-captioning-base` | HuggingFace model ID |
| `EMBEDDING_MODEL` | `str` | `sentence-transformers/all-MiniLM-L6-v2` | ChromaDB embedding model |
| `MIN_CONTOUR_AREA` | `int` | `2000` | Minimum foreground blob (px²) |
| `MAX_CONTOUR_AREA_RATIO` | `float` | `0.40` | Max blob as fraction of frame |
| `MOG2_VAR_THRESHOLD` | `int` | `120` | MOG2 variance sensitivity |
| `MOG2_WARMUP_FRAMES` | `int` | `5` | Silent calibration frames |
| `OPENAI_MODEL` | `str` | `gpt-4o-mini` | LLM for the agent |
| `AGENT_MEMORY_K` | `int` | `10` | Sliding window memory depth |
| `AGENT_VERBOSE` | `bool` | `False` | Print ReAct chain to stdout |
| `LOCATIONS` | `list` | `["main_gate", "garage", "perimeter"]` | Valid location names |
| `KNOWN_VEHICLES` | `list` | `["blue ford f150", "blue truck"]` | RULE-02 allowlist |
| `TELEGRAM_BOT_TOKEN` | `str` | `""` | From `.env` |
| `TELEGRAM_CHAT_ID` | `str` | `""` | From `.env` |
| `LOITER_THRESHOLD_SECONDS` | `int` | `15` | RULE-04 dwell before 2nd alert |
| `REPEAT_ENTRY_LIMIT` | `int` | `1` | RULE-03 entry count threshold |
| `LIGHTING_CENTER_SAMPLE_RATIO` | `float` | `0.65` | Central crop fraction for brightness |
| `LIGHTING_NIGHT_MEAN_THRESHOLD` | `int` | `90` | V-channel mean below → night |
| `LIGHTING_DARK_PIXEL_VALUE` | `int` | `50` | Pixel V below → dark |
| `LIGHTING_NIGHT_DARK_PIXEL_RATIO` | `float` | `0.55` | Dark-pixel fraction → night |
| `VEHICLE_TRACK_MAX_FRAME_GAP` | `int` | `45` | Max gap to continue a track |
| `VEHICLE_TRACK_MAX_VELOCITY_PX_PER_FRAME` | `int` | `25` | Expected max blob speed |
| `VEHICLE_TRACK_VELOCITY_MARGIN` | `float` | `1.5` | Tolerance on predicted distance |
| `VEHICLE_TRACK_MAX_TRAJECTORY_POINTS` | `int` | `8` | History positions kept |
| `MIN_DISTINCT_PERSON_SEPARATION_PX` | `int` | `250` | Center distance → 2 persons |

---

<a id="42-mainpy"></a>
### 4.2 `main.py`

CLI entry point. Clears state, wires all components, runs the pipeline, prints results, and opens the interactive query REPL.

**Startup sequence:**
1. Delete `EVENTS_DB_PATH` and `CHROMA_DIR` (fresh run)
2. Instantiate `EventStore`, `FrameIndex`, `AlertRulesEngine`, `SecurityAgent`, `VideoProcessor`
3. Iterate `processor.iter_events(VIDEO_PATH)` — print each non-progress event
4. Print alert summary from `store.get_alerts()`
5. Print full event log from `store.get_all_events()`
6. Run sample ChromaDB queries
7. Launch `QueryEngine.run_interactive()` — REPL until `quit`

```
Functions
─────────
print_section(title)         Print a padded section header
_print_event(event)          Dispatch PipelineEvent to console output
main()                       Entry point — full run + REPL
```

---

<a id="43-pipeline"></a>
### 4.3 `pipeline/`

#### `pipeline/video_processor.py`

The master orchestrator. Opens the video file and drives every downstream component in a generator loop.

**`PipelineEvent` dataclass**

```
Fields:
  kind      str           Event type (see below)
  message   str           Human-readable description
  frame_id  Optional[int] Sampled frame counter
  raw_idx   Optional[int] Raw frame index in video file
  severity  Optional[str] Alert severity if applicable
  rule_id   Optional[str] Alert rule ID if applicable
  extra     dict          Kind-specific payload (see below)
```

| `kind` | Emitted when | `extra` keys |
|---|---|---|
| `startup` | Video opened, session info logged | — |
| `warmup` | MOG2 calibration frame | `warmup_count`, `total_warmup` |
| `info` | General status messages | — |
| `progress` | Every sampled frame (UI progress bar) | `current`, `total` |
| `skip` | Frame with no foreground detected | — |
| `detection` | Foreground found, BLIP ran | `objects`, `caption`, `activity` |
| `alert` | Rule fired | `rule_id`, `severity`, `message` |
| `track` | VehicleTracker updated | `track_id`, `entry_count`, `is_new` |
| `agent` | Agent produced a response | `response` |
| `frame` | Full frame bundle (used by UI) | `status`, `logs`, `alerts`, `preview_rgb`, `blip`, `telemetry`, `track` |
| `error` | Exception during frame processing | — |
| `done` | All frames processed | `total_detections`, `total_alerts` |

**`infer_activity(objects, caption) → str`**

Maps BLIP caption keywords to activity labels:

```
walking / approaching / entering / moving  →  "entering"
standing / loitering / waiting / stationary →  "loitering"
parked / stopped / sitting                 →  "stationary"
(no objects)                               →  "clear"
(default)                                  →  "entering"
```

**`VideoProcessor` class**

```python
VideoProcessor(store, index, rules, agent=None, location="main_gate", sample_every_n=2)

iter_events(video_path: str) -> Iterator[PipelineEvent]
```

Internal flow per sampled frame:

```
open video with cv2.VideoCapture
  │
  ├─ yield startup events (resolution, fps, duration, sampled count)
  ├─ warm up BackgroundSubtractor on first MOG2_WARMUP_FRAMES
  │
  └─ for each sampled frame:
       ├─ subtractor.apply(frame) → crops
       ├─ if empty → yield skip, continue
       ├─ lighting_detector.analyze_frame_lighting(frame) → LightingAnalysis
       ├─ blip.analyze_frame(crops) → FrameVLMResult
       ├─ infer_activity(objects, caption)
       ├─ tracker.update(…) → TrackUpdate
       ├─ rules.evaluate(…) → alerts
       ├─ for HIGH alerts: telegram_alert(…, frame=frame, bbox=crop_bbox)
       ├─ store.log_alert(…) for each alert
       ├─ agent.process(…) if agent enabled → response string
       ├─ index.add_frame(…)
       └─ yield PipelineEvent(kind="frame", extra={…full bundle…})
```

---

#### `pipeline/frame_preview.py`

Prepares frames for the Streamlit live preview panel.

```python
prepare_frame_preview(
    bgr_frame: np.ndarray,
    bbox: Optional[Tuple[int,int,int,int]] = None,
    max_width: int = 960
) -> np.ndarray   # RGB uint8
```

Steps:
1. If `bbox` provided: draw green rectangle (`(0,220,80)`, thickness 2)
2. Convert BGR → RGB
3. If width > `max_width`: downscale proportionally (preserves aspect ratio)

Constants: `MAX_PREVIEW_WIDTH = 960`, `BBOX_COLOR_BGR = (0, 220, 80)`

---

#### `pipeline/session.py`

Manages isolated per-run session directories for the UI so concurrent users don't share state.

```python
create_session_id() -> str                        # uuid4 hex[:12]
session_paths(session_id) -> dict                 # {root, events_db, chroma_dir}
ensure_session_dirs(session_id) -> dict           # mkdir + return paths
reset_session(session_id) -> dict                 # rmtree root + recreate
save_upload(uploaded_bytes, filename) -> str      # write to UI_UPLOADS_DIR, return path
```

Each session gets its own SQLite file and ChromaDB directory under `data/sessions/<id>/`.

---

<a id="44-vlm"></a>
### 4.4 `vlm/`

#### `vlm/constants.py`

Keyword taxonomies used by `BLIPAnalyzer` and `QueryEngine`.

| Taxonomy | Members (examples) |
|---|---|
| `OBJECT_TYPES` | `person`, `police_car`, `bicycle`, `animal`, `vehicle`, `unknown` |
| `POLICE_CAR_KEYWORDS` | `police car`, `patrol car`, `sheriff`, `cop car`, … |
| `BICYCLE_KEYWORDS` | `bicycle`, `bike`, `cyclist`, `ebike`, … |
| `ANIMAL_KEYWORDS` | `dog`, `cat`, `bird`, `deer`, `raccoon`, … |
| `PERSON_KEYWORDS` | `person`, `people`, `man`, `woman`, `pedestrian`, … |
| `VEHICLE_KEYWORDS` | `car`, `truck`, `van`, `suv`, brand names (toyota, ford, bmw, …) |
| `COLOR_KEYWORDS` | `red`, `blue`, `white`, `black`, `grey`, `silver`, … |
| `MOTOR_VEHICLE_TYPES` | `vehicle`, `police_car` (used for tracker + RULE-02/03) |

Priority order for `extract_object_type(caption)`:
`police_car > person > bicycle > animal > vehicle > unknown`

```python
extract_object_type(caption: str) -> str
extract_color(caption: str) -> str
```

---

#### `vlm/background_subtractor.py`

Wraps OpenCV MOG2 with contour filtering and crop extraction.

**`ObjectCrop` dataclass**

```
image          PIL.Image     Cropped object region (colour)
bbox           (x, y, w, h)  Absolute pixel coordinates
bbox_relative  (x, y, w, h)  Normalized 0–1 relative to frame
```

**`BackgroundSubtractor` class**

```python
BackgroundSubtractor()
# Internally: cv2.createBackgroundSubtractorMOG2(
#     history=500, varThreshold=MOG2_VAR_THRESHOLD, detectShadows=True)

warm_up(frames: List[np.ndarray]) -> None
# Feed frames in learning-only mode. Sets _warmed_up = True.
# Exposed attributes: _warmup_count, _warmed_up (read by VideoProcessor for progress msgs)

apply(frame: np.ndarray) -> List[ObjectCrop]
# 1. subtractor.apply(frame) → raw mask
# 2. Threshold at 200 (removes shadow class = 127)
# 3. cv2.morphologyEx(MORPH_CLOSE, 5×5 kernel) to fill gaps
# 4. cv2.findContours(RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
# 5. Filter: area < MIN_CONTOUR_AREA → discard
#            area > frame_area * MAX_CONTOUR_AREA_RATIO → discard
# 6. cv2.boundingRect → (x,y,w,h), add 10px padding, clamp to frame
# 7. Crop original colour frame → PIL.Image

reset() -> None
# Recreate MOG2 instance (called between sessions)
```

**MOG2 parameter rationale:**

| Parameter | Value | Rationale |
|---|---|---|
| `history` | `500` | Larger window → stable background model for static drone |
| `varThreshold` | `120` | High → tolerates gradual lighting change without false triggers |
| `detectShadows` | `True` | Returns shadow pixels as 127; we threshold at 200 to exclude them |

---

#### `vlm/blip_analyzer.py`

Runs BLIP on object crops and produces structured output.

**`CropAnalysis` dataclass**

```
caption      str    Raw BLIP output sentence
object_type  str    Mapped from constants.extract_object_type()
color        str    First matched COLOR_KEYWORD (or "")
bbox         tuple  (x, y, w, h) from ObjectCrop
```

**`FrameVLMResult` dataclass**

```
crops            List[CropAnalysis]
objects          List[str]   Deduplicated object types across all crops
raw_description  str         All captions joined with "; "
```

**`BLIPAnalyzer` class**

```python
BLIPAnalyzer()
# Loads on init: BlipProcessor + BlipForConditionalGeneration
# Device priority: CUDA → MPS (Apple Silicon) → CPU
# Model size: ~3.7 GB (cached at ~/.cache/huggingface/)

analyze_crop(crop: ObjectCrop) -> CropAnalysis
# processor(image, return_tensors="pt") → model.generate(max_new_tokens=30) → decode

analyze_frame(crops: List[ObjectCrop]) -> Optional[FrameVLMResult]
# Calls analyze_crop() per crop, merges results

analyze_full_image(image: PIL.Image) -> CropAnalysis
# Used by validation scripts; runs BLIP on the full frame (no cropping)
```

---

#### `vlm/lighting_detector.py`

Classifies scene as day or night using frame brightness, independent of wall-clock time.

**`LightingAnalysis` dataclass** *(frozen)*

```
is_night          bool    True if scene is dark
brightness        float   Mean V-channel value in central crop
dark_pixel_ratio  float   Fraction of pixels with V < LIGHTING_DARK_PIXEL_VALUE
label             str     "day" | "night"
```

**`analyze_frame_lighting(frame_bgr) → LightingAnalysis`**

```
Algorithm:
1. Convert BGR → HSV, extract V channel
2. Crop central 65% (LIGHTING_CENTER_SAMPLE_RATIO) to exclude edge overlays
3. brightness = mean(V_crop)
4. dark_pixel_ratio = mean(V_crop < 50)
5. is_night = (brightness < 90) OR (dark_pixel_ratio >= 0.55)
```

Used by `AlertRulesEngine` for RULE-01 and RULE-05 instead of wall-clock time — works correctly on pre-recorded footage at any play speed.

---

<a id="45-agent"></a>
### 4.5 `agent/`

#### `agent/alert_rules.py`

The deterministic safety layer. Runs before the LLM on every frame with detections.

**`Alert` dataclass**

```
rule_id   str   "RULE-01" … "RULE-05"
frame_id  int   Sampled frame counter
message   str   Human-readable alert message
severity  str   "high" | "medium"
```

**`threat_from_alerts(alerts, objects) → str`**

Returns `"high"` if any HIGH alert fired, `"medium"` if any MEDIUM, `"low"` if objects present but no alerts, `"none"` otherwise.

**`AlertRulesEngine` class**

Internal state:
- `_person_alert_issued: set` — track IDs that already triggered RULE-01 (prevents repeat firing)
- `_vehicle_alert_issued: set` — track IDs that triggered RULE-02
- `_person_loiter_start: dict` — `track_id → first_seen_timestamp` for RULE-04 dwell timing
- `_loiter_alert2_issued: set` — track IDs that triggered the prolonged RULE-04 alert

```python
evaluate(
    frame_id, timestamp, location, objects, activity,
    description, bbox=None, color="", is_night=False
) -> List[Alert]

reset() -> None   # clears all state (called between sessions)

last_track_update: Optional[TrackUpdate]   # set by VideoProcessor before calling evaluate()
```

**Rule implementations:**

```
RULE-01  person at night
         Condition: "person" in objects AND is_night
         Fires once per track_id (per-track dedup via _person_alert_issued)
         Severity: HIGH

RULE-02  unrecognized vehicle
         Condition: motor_vehicle in objects
                    AND description not in KNOWN_VEHICLES (case-insensitive substring)
                    AND track is new entry (is_new_entry=True from tracker)
         Fires once per new track_id (per-track dedup via _vehicle_alert_issued)
         Severity: MEDIUM

RULE-03  repeat vehicle entry
         Condition: motor_vehicle in objects
                    AND track.entry_count > REPEAT_ENTRY_LIMIT
         Fires when entry_count increments past the limit
         Severity: MEDIUM

RULE-04  person loitering / stationary — two-stage
         Stage 1: any detection of person with activity loitering|stationary
                  → immediate HIGH alert + record timestamp
         Stage 2: dwell_time > LOITER_THRESHOLD_SECONDS since stage 1
                  → second HIGH alert "loitering for Xs"
         Both stages deduped by track_id

RULE-05  perimeter activity at night
         Condition: location == "perimeter" AND is_night
         No per-track dedup (fires every frame at perimeter at night)
         Severity: HIGH
```

---

#### `agent/security_agent.py`

LangChain ReAct agent wrapping GPT-4o-mini.

**`SecurityAgent` class**

```python
SecurityAgent(store: EventStore, index: FrameIndex)
# On init:
#   1. init_tools(store, index)   — inject shared DB instances into tools.py
#   2. ChatOpenAI(model=OPENAI_MODEL, temperature=0)
#   3. ConversationBufferWindowMemory(k=AGENT_MEMORY_K, memory_key="chat_history",
#                                     return_messages=True)
#   4. create_react_agent(llm, tools, prompt)
#   5. AgentExecutor(agent, tools, memory, verbose=AGENT_VERBOSE,
#                    handle_parsing_errors=True, max_iterations=6,
#                    callbacks=[SafeStdOutCallbackHandler()])

process(
    frame_id, timestamp, location, objects, activity,
    description, pre_alerts, vehicle_context=None, bbox=None
) -> str
# Builds query string with all context, invokes AgentExecutor.invoke(),
# returns agent response string
```

**System prompt excerpt:**

```
You are Flying Police, an AI security analyst monitoring surveillance camera footage for a property.
You have access to tools to log events and check history.

For each frame you receive, you MUST call log_event() once with a summary.
If you see something suspicious that was NOT already flagged as a pre-fired alert, use trigger_alert().
Use query_history() if you need cross-frame context.
```

**Agent query string format:**

```
Frame {frame_id} | {location} | {timestamp}
Objects: {objects}
Activity: {activity}
Caption: {description}
Lighting: {day|night}
Pre-fired alerts: {pre_alerts list}
Vehicle context: {track_id, entry_count, bbox_center, trajectory (last N points)}
```

---

#### `agent/tools.py`

LangChain `@tool` functions injected with shared `EventStore` and `FrameIndex` instances.

```python
init_tools(store: EventStore, index: FrameIndex) -> None
# Must be called before agent is constructed. Sets module-level _store and _index.

@tool
log_event(message: str) -> str
# EventStore.log_event(frame_id_from_agent_context, message, severity="low", type="log")
# Returns: "Event logged: evt_<id>"

@tool
trigger_alert(message: str, severity: str = "medium") -> str
# EventStore.log_alert(frame_id, "AGENT", message, severity)
# Returns: "Alert stored: alt_<id>"

@tool
query_history(query: str) -> str
# FrameIndex.query(query, n_results=3)
# Returns formatted string of top-3 matching frames

@tool
query_track_positions(location: str) -> str
# FrameIndex.get_recent_at_location(location, n_results=6)
# Returns formatted string of recent detections at that location
```

---

#### `agent/vehicle_tracker.py`

Maintains cross-frame object identity using spatial proximity matching.

**`TrackUpdate` dataclass**

```
track_id            str     Unique track identifier ("T-1", "T-2", …)
is_continuing       bool    True if matched an existing track
is_new_entry        bool    True if this is a first or re-entry
entry_count         int     How many times this object has entered
object_type         str     Current frame's detected type
prior_object_type   str     Previous detection's type
label_changed       bool    True if object_type differs from prior
bbox                tuple   Current bounding box (x,y,w,h)
center              tuple   (cx, cy)
frame_gap           int     Frames since last sighting
center_distance     float   Pixel distance moved since last sighting
max_match_distance  float   Threshold used for this match decision
trajectory          list    Last N (x,y) centers
```

**`VehicleTracker` class**

```python
update(
    frame_id, location, description, bbox=None, color="", object_type="unknown"
) -> Optional[TrackUpdate]
# 1. Compute center from bbox
# 2. For each active track: compute max_match_distance = MAX_VELOCITY * gap * MARGIN
# 3. Find nearest active track center within max_match_distance
# 4. Hit → continue track, update last_bbox/frame/type, append center to trajectory
# 5. Miss → new track: track_id = T-{counter}, entry_count = 1
# 6. Return TrackUpdate

reset() -> None
```

**Matching formula:**

```
max_match_distance = VEHICLE_TRACK_MAX_VELOCITY_PX_PER_FRAME
                     × max(frame_gap, 1)
                     × VEHICLE_TRACK_VELOCITY_MARGIN
                   = 25 × gap × 1.5
```

A track is considered the same object if the new bbox center falls within this radius.

---

#### `agent/vehicle_context.py`

Converts `TrackUpdate` to a plain dict for inclusion in the agent's prompt.

```python
primary_crop_bbox(vlm_result: FrameVLMResult) -> Optional[Tuple[int,int,int,int]]
# Returns bbox of the first crop in vlm_result.crops

primary_crop_color(vlm_result: FrameVLMResult) -> str
# Returns color of the first crop

track_to_context(track: TrackUpdate) -> dict
# Returns: {track_id, is_continuing, is_new_entry, entry_count, object_type,
#           bbox_center, frame_gap, trajectory}
```

---

#### `agent/callbacks.py`

**`SafeStdOutCallbackHandler`** — patches `StdOutCallbackHandler` to handle `serialized=None` passed by `langchain-core 0.3` in `on_chain_start`, which would otherwise raise a `TypeError`.

---

<a id="46-storage"></a>
### 4.6 `storage/`

#### `storage/event_store.py`

SQLite wrapper with two tables.

**Schema:**

```sql
CREATE TABLE events (
    event_id  TEXT PRIMARY KEY,
    frame_id  INTEGER,
    timestamp TEXT,
    type      TEXT,   -- "log" | "alert"
    message   TEXT,
    severity  TEXT    -- "low" | "medium" | "high"
);

CREATE TABLE alerts (
    alert_id  TEXT PRIMARY KEY,
    event_id  TEXT,
    frame_id  INTEGER,
    timestamp TEXT,
    rule_id   TEXT,   -- "RULE-01"…"RULE-05" | "AGENT"
    message   TEXT,
    severity  TEXT
);
```

**`EventStore` class**

```python
EventStore(db_path=EVENTS_DB_PATH)
# Creates tables on init

log_event(frame_id, message, severity="low", type="log") -> str
# INSERT INTO events; returns event_id = "evt_<uuid4[:8]>"

log_alert(frame_id, rule_id, message, severity) -> str
# INSERT INTO events (type="alert") + INSERT INTO alerts
# Returns alert_id = "alt_<uuid4[:8]>"

get_alerts(severity=None) -> List[dict]
# SELECT * FROM alerts [WHERE severity = ?] ORDER BY timestamp

get_events_by_timerange(start: str, end: str) -> List[dict]
# SELECT * FROM events WHERE timestamp BETWEEN ? AND ?

get_all_events() -> List[dict]
# SELECT * FROM events ORDER BY timestamp

close() -> None
# conn.close()
```

---

#### `storage/frame_index.py`

ChromaDB wrapper for semantic frame search.

**Collection spec:**

```
name:       "frames"
embedding:  sentence-transformers/all-MiniLM-L6-v2  (384-dim)
similarity: cosine (HNSW index)
id:         str(frame_id)  — upserts on repeat frame_id
```

**`FrameIndex` class**

```python
FrameIndex(chroma_dir=CHROMA_DIR)

add_frame(
    frame_id, description, timestamp, location, objects,
    threat_level, bbox=None, track_id=None
) -> None
# Document: description
# Metadata: frame_id, timestamp, location, objects (joined), threat_level,
#           center_x, center_y (from bbox midpoint), track_id

query(
    text: str,
    n_results: int = 5,
    location: str = None,
    threat_level: str = None
) -> List[dict]
# Semantic search; optional metadata filters
# Returns: [{frame_id, description, timestamp, location, objects, score}, …]

get_recent_at_location(location: str, n_results: int = 5) -> List[dict]
# Semantic query "activity at {location}" filtered by location metadata

list_all_frames() -> List[dict]
# Returns all documents + metadata

count() -> int
# collection.count()
```

---

<a id="47-query"></a>
### 4.7 `query/`

#### `query/query_engine.py`

Routes natural-language questions to the right database backend.

**Query routing logic:**

```
Input text
    │
    ├─► COUNT_QUERY_PATTERN match?  →  _query_count()
    │       returns {type:"count", count:N, individuals:M, label:…}
    │
    ├─► "alert" / "threat" keyword?  →  _query_alerts()
    │       returns all alerts from EventStore
    │
    ├─► time keyword? (T22, T23, midnight, night, T00–T05)  →  _query_by_time()
    │       returns events in matching hour range from EventStore
    │
    └─► default  →  FrameIndex.query(text, n_results=5, location=location)
            optionally filtered by _extract_location(text) and _extract_object_type(text)
```

**`QueryEngine` class**

```python
QueryEngine(store: EventStore, index: FrameIndex)

query(text: str) -> List[dict] | dict
format_results(results) -> str     # pretty-print for CLI
run_interactive() -> None          # REPL: prompt "> " until "quit"
```

**`OBJECT_TYPE_ALIASES`** maps query keywords to taxonomy sets:

```python
{
    "person": PERSON_KEYWORDS,
    "vehicle": VEHICLE_KEYWORDS | {"motor vehicle"},
    "police_car": POLICE_CAR_KEYWORDS,
    "animal": ANIMAL_KEYWORDS,
    "bicycle": BICYCLE_KEYWORDS,
}
```

---

#### `query/track_merge.py`

Deduplicates fragmented tracks into distinct physical individuals for count queries.

**`merge_tracks_into_individuals(frames) → (raw_tracks, merged_individuals)`**

Uses union-find. Two tracks are merged if:
- Same location AND overlapping frame ranges AND center distance < `MIN_DISTINCT_PERSON_SEPARATION_PX` (250px)
- OR non-overlapping with gap ≤ `VEHICLE_TRACK_MAX_FRAME_GAP` (45 frames)

Used by `QueryEngine._query_count()` to return accurate "how many people" counts.

---

<a id="48-notifications"></a>
### 4.8 `notifications/`

#### `notifications/telegram_notifier.py`

Real-time alert delivery via Telegram Bot API.

**Constants:**

```python
_SEVERITY_EMOJI  = {"high": "🔴", "medium": "🟡", "low": "🟢"}
_SEVERITY_COLOR_BGR = {"high": (0,0,220), "medium": (0,165,255), "low": (0,200,0)}
_API_TIMEOUT = 10   # seconds per HTTP request
```

**`send_alert(rule_id, message, severity, frame_id, timestamp, location, frame=None, bbox=None)`**

```
1. Guard: if TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing → return silently
2. Guard: if severity.lower() != "high" → return silently
3. Build caption string (rule_id, message, severity, frame_id, timestamp, location)
4. If frame provided:
     _annotate_frame() → draws bbox + label in red → JPEG encode → bytes
     spawn daemon thread → _send_photo(caption, image_bytes)
   Else:
     spawn daemon thread → _send_text(caption)
```

**`_annotate_frame(frame, rule_id, severity, bbox) → bytes`**

```
1. Copy frame
2. cv2.rectangle(copy, (x,y), (x+w,y+h), color=(0,0,220), thickness=3)
3. cv2.putText() → "RULE-01 | HIGH" label above bbox
4. cv2.imencode(".jpg", copy, [IMWRITE_JPEG_QUALITY, 85]) → bytes
```

**`_send_photo(caption, image_bytes) → None`**

```
POST https://api.telegram.org/bot{TOKEN}/sendPhoto
  multipart/form-data: chat_id, photo=(frame.jpg, bytes), caption
  timeout: 10s
```

**Fire-and-forget pattern:**

```python
threading.Thread(target=_send_photo, args=(caption, image_bytes), daemon=True).start()
```

The daemon flag ensures the thread is killed when the main process exits — no blocking on network I/O.

---

<a id="49-data"></a>
### 4.9 `data/`

#### `data/simulated_frames.py`

Seven hand-crafted scenarios covering all alert rule paths.

**`FrameAnalysis` dataclass**

```
frame_id         int
timestamp        str     ISO 8601
location         str     main_gate | garage | perimeter
raw_description  str     BLIP-like caption
objects          list    object type list
activity         str     entering | loitering | stationary | clear
threat_level     str     high | medium | low | none
telemetry        dict    drone GPS + battery
image_path       str     optional path to sample image
```

**Simulated scenarios:**

| ID | Time | Location | Description | Expected Output |
|---|---|---|---|---|
| S-01 | 08:00 | main_gate | Blue Ford F150 entering | LOG: vehicle entry |
| S-02 | 12:00 | garage | Blue Ford F150 parked | LOG: vehicle at garage |
| S-03 | 08:30 | main_gate | Blue Ford F150 entering again | LOG + ALERT: RULE-03 repeat entry |
| S-04 | 23:58 | main_gate | Person approaching | ALERT: RULE-01 person at night |
| S-05 | 00:01 | main_gate | Person at gate | ALERT: RULE-01 + RULE-04 loitering |
| S-06 | 14:00 | perimeter | Unknown white sedan | ALERT: RULE-02 unknown vehicle |
| S-07 | 09:00 | garage | No activity | LOG: clear |

---

#### `data/telemetry.py`

Simulates drone telemetry for testing.

```python
get_telemetry(timestamp, battery_pct=85) -> dict
# Returns: {drone_lat: 18.5204, drone_lon: 73.8567, altitude_m: 10.0,
#           battery_pct: battery_pct, timestamp: timestamp}

battery_for_frame(frame_index, total_frames, start_pct=95) -> int
# Linear drain: start_pct - (frame_index/total_frames) * 15, floor at 10

estimate_ground_coverage(frame_w, frame_h, altitude_m, h_fov_deg) -> dict
# Pinhole geometry → {ground_width_m, ground_height_m, coverage_area_m2, px_per_meter}
```

Drone position: fixed at 18.5204°N, 73.8567°E (Pune, India), 10m altitude, 90° horizontal FOV.

---

#### `data/validation_capture.py`

Golden fixture system for regression testing without re-running the full model stack.

**Dataclasses:**

```
CropRecord       caption, object_type, color, bbox, image_file
AlertRecord      rule_id, message, severity
CaptureRecord    frame_id, raw_idx, timestamp, location, has_foreground,
                 full_frame_file, activity, threat_level,
                 blip: List[CropRecord], alerts: List[AlertRecord],
                 agent_response, telemetry, vehicle_track
```

**`ValidationCaptureRecorder` class**

```python
should_capture(frame_id) -> bool
# True every VALIDATION_CAPTURE_EVERY_N frames AND total < VALIDATION_MAX_CAPTURES

save_capture(frame_id, raw_idx, ..., bgr_frame, crops) -> CaptureRecord
# Writes: full.jpg  (annotated frame)
#         crop_00.jpg, crop_01.jpg, … (individual object crops)
#         record.json  (full CaptureRecord as dict)
# Directory: VALIDATION_FIXTURES_DIR/frame_{frame_id:04d}/

write_manifest(video_path, sample_every_n, capture_on="sampled") -> str
# Writes manifest.json summarizing all captures
```

**Utility functions:**

```python
discover_capture_dirs(fixtures_dir) -> List[str]
load_manifest(fixtures_dir) -> dict
rebuild_manifest_from_captures(fixtures_dir, write=True) -> dict
load_capture_record(capture_dir) -> CaptureRecord
list_capture_dirs(fixtures_dir) -> List[str]
fixtures_available(fixtures_dir=None) -> bool
run_capture_pipeline(video_path, output_dir, ...) -> str   # CLI-callable runner
```

---

<a id="410-ui"></a>
### 4.10 `ui/`

#### `ui/app.py`

Streamlit web interface. Supports any MP4 upload, live analysis with frame preview, alert feed, and post-session NL queries.

**Page layout:**

```
Sidebar
  ├─ Location selector (main_gate | garage | perimeter)
  └─ Use LangChain Agent toggle (default: ON)

Main area
  ├─ Upload MP4 widget
  ├─ Start Analysis button
  │
  ├─ [during processing]
  │   ├─ Live frame preview (annotated RGB image)
  │   ├─ Live detection metadata (caption, objects, activity, track)
  │   ├─ Progress bar
  │   ├─ Session log (last MAX_LOG_LINES=400 lines)
  │   └─ Alert feed (coloured cards, scrollable)
  │
  └─ [after processing]
      ├─ Metrics row: frames processed | detections | alerts fired
      ├─ Full alert list with coloured severity badges
      └─ Q&A chat: natural-language query → QueryEngine.query() → formatted response
```

**Key functions:**

```python
_init_state() -> None
# Initializes st.session_state: session_id, running, log_lines, alerts,
# last_frame_data, metrics, chat_history

_severity_badge(severity) -> str
# Returns HTML <span> with background-color per severity

_render_alert_card(alert) -> None
_render_frame_panel(frame_data, *, title) -> None
_render_session_log(log_slot) -> None
_render_alert_feed(alerts_slot) -> None

_run_pipeline(video_path, location, use_agent,
              live_frame_slot, live_meta_slot, progress_bar,
              log_slot, alerts_slot) -> None
# Creates new session via session.py
# Instantiates store, index, rules, agent, processor per session
# Iterates PipelineEvent stream:
#   "progress" → update progress_bar
#   "frame" → update live_frame_slot + live_meta_slot + log
#   "alert" → append to st.session_state.alerts + update alerts_slot
#   "done" → write metrics

main() -> None
# Streamlit entry point
```

**Alert severity colours:**

```python
ALERT_SEVERITY_COLORS = {
    "high":   "#ff4b4b",
    "medium": "#ffa500",
    "low":    "#4b9fff",
}
```

---

<a id="5-alert-rules-reference"></a>
## 5. Alert Rules Reference

```
RULE-01  Person detected at night
├─ Trigger:    "person" in objects AND is_night (from LightingAnalysis)
├─ Severity:   HIGH
├─ Dedup:      Once per track_id per session
└─ Message:    "Person detected at {location} at night"

RULE-02  Unrecognized vehicle
├─ Trigger:    motor_vehicle in objects
│              AND caption not substring-matched to KNOWN_VEHICLES
│              AND track.is_new_entry = True
├─ Severity:   MEDIUM
├─ Dedup:      Once per track_id per session
└─ Message:    "Unrecognized vehicle at {location}: {description}"

RULE-03  Repeat vehicle entry
├─ Trigger:    motor_vehicle in objects AND track.entry_count > REPEAT_ENTRY_LIMIT
├─ Severity:   MEDIUM
├─ Dedup:      Fires when count increments past limit (not every frame)
└─ Message:    "Vehicle entered {location} {N} times this session"

RULE-04  Person loitering / stationary  [two-stage]
├─ Stage 1 trigger:  "person" in objects AND activity in {loitering, stationary}
│  Severity:   HIGH
│  Fires:      Immediately on first detection
│  Message:    "Person loitering at {location}"
│
└─ Stage 2 trigger:  Stage 1 fired AND dwell_time > LOITER_THRESHOLD_SECONDS
   Severity:   HIGH
   Fires:      Once (second alert per track_id)
   Message:    "Person loitering at {location} for {N}s"

RULE-05  Perimeter activity at night
├─ Trigger:    location == "perimeter" AND is_night
├─ Severity:   HIGH
├─ Dedup:      None (fires every frame at perimeter at night)
└─ Message:    "Activity detected at perimeter at night"
```

**Alert suppression summary:**

| Rule | Per-session dedup | Per-track dedup |
|---|---|---|
| RULE-01 | — | Yes (track_id) |
| RULE-02 | — | Yes (track_id, new_entry only) |
| RULE-03 | — | On count increment |
| RULE-04 | — | Yes (track_id, both stages) |
| RULE-05 | — | No |

---

<a id="6-storage-schema"></a>
## 6. Storage Schema

<a id="61-sqlite-dataeventsdb"></a>
### 6.1 SQLite (`data/events.db`)

```sql
-- All observations logged by the agent and rules engine
CREATE TABLE events (
    event_id  TEXT PRIMARY KEY,   -- "evt_<uuid8>"
    frame_id  INTEGER,
    timestamp TEXT,               -- ISO 8601
    type      TEXT,               -- "log" | "alert"
    message   TEXT,
    severity  TEXT                -- "low" | "medium" | "high"
);

-- Security alerts (subset of events where type="alert")
CREATE TABLE alerts (
    alert_id  TEXT PRIMARY KEY,   -- "alt_<uuid8>"
    event_id  TEXT,               -- FK → events.event_id
    frame_id  INTEGER,
    timestamp TEXT,
    rule_id   TEXT,               -- "RULE-01"…"RULE-05" | "AGENT"
    message   TEXT,
    severity  TEXT
);
```

<a id="62-chromadb-datachroma"></a>
### 6.2 ChromaDB (`data/chroma/`)

```
Collection: "frames"
Model:      sentence-transformers/all-MiniLM-L6-v2  (384-dim float32)
Similarity: cosine  (HNSW)

Document:   raw_description  (BLIP caption string — what gets embedded)

Metadata fields per document:
  frame_id    int     Sampled frame counter
  timestamp   str     ISO 8601
  location    str     main_gate | garage | perimeter
  objects     str     Space-joined object type list
  threat_level str    high | medium | low | none
  center_x    float   Horizontal bbox midpoint (normalized 0–1)
  center_y    float   Vertical bbox midpoint (normalized 0–1)
  track_id    str     VehicleTracker track identifier
```

---

<a id="7-configuration-reference"></a>
## 7. Configuration Reference

All values in `config.py`. Override via environment variable where applicable.

**Processing tuning:**

| Constant | Default | Effect of increasing |
|---|---|---|
| `SAMPLE_EVERY_N_FRAMES` | `2` | Skip more frames → faster but coarser |
| `MOG2_VAR_THRESHOLD` | `120` | Higher → less sensitive to lighting change |
| `MOG2_WARMUP_FRAMES` | `5` | More warmup → better BG model, later first detection |
| `MIN_CONTOUR_AREA` | `2000 px²` | Higher → ignore smaller blobs |
| `MAX_CONTOUR_AREA_RATIO` | `0.40` | Lower → reject more large blobs |

**Alert thresholds:**

| Constant | Default | Production value |
|---|---|---|
| `LOITER_THRESHOLD_SECONDS` | `15` | `300` (5 min) |
| `REPEAT_ENTRY_LIMIT` | `1` | `2` (allow one re-entry before alert) |
| `AFTER_HOURS_START` | N/A | Uses lighting detection, not clock |

**Tracker tuning:**

| Constant | Default | Effect of increasing |
|---|---|---|
| `VEHICLE_TRACK_MAX_FRAME_GAP` | `45` | Track survives longer occlusions |
| `VEHICLE_TRACK_MAX_VELOCITY_PX_PER_FRAME` | `25 px` | Allow faster-moving objects |
| `VEHICLE_TRACK_VELOCITY_MARGIN` | `1.5` | More position tolerance per frame |

---

<a id="8-test-suite"></a>
## 8. Test Suite

### Structure

```
tests/
├── conftest.py           # Pytest markers + validation fixture gate
└── unit/
    ├── conftest.py
    ├── test_agent.py           # SecurityAgent (mocked OpenAI)
    ├── test_alerts.py          # AlertRulesEngine rules logic
    ├── test_callbacks.py       # SafeStdOutCallbackHandler
    ├── test_frame_index.py     # FrameIndex.add / query / count
    ├── test_frame_preview.py   # prepare_frame_preview resize + bbox
    ├── test_indexing.py        # ChromaDB add + semantic query
    ├── test_lighting_detector.py  # Day/night classification edge cases
    ├── test_object_types.py    # extract_object_type keyword matching
    ├── test_pipeline.py        # PipelineEvent defaults + infer_activity
    ├── test_query_engine.py    # Query routing (count, time, alert, semantic)
    ├── test_track_merge.py     # union-find merge logic
    └── test_vehicle_tracker.py # Spatial matching + entry counting
```

**Total: 64 unit tests**. No model downloads, no API calls, no Telegram credentials required.

### Running tests

```bash
source venv/bin/activate
python3 -m pytest tests/unit/ -v
```

### Test markers

```
@pytest.mark.unit        Applied automatically to tests/unit/ tests
@pytest.mark.validation  Applied to tests/validation/ — skipped unless
                         validation fixtures exist (run scripts/capture_validation_set.py first)
@pytest.mark.integration Applied to tests/integration/
```

### Generating validation fixtures

```bash
source venv/bin/activate
python3 scripts/capture_validation_set.py \
    --video data/sample_video/entrance_area_720p.mp4 \
    --output data/validation_fixtures \
    --every-n 30 \
    --max 15
```

This saves annotated JPEG frames, crop images, and `record.json` files used by the validation test suite.

---

<a id="9-reference-videos"></a>
## 9. Reference Videos

All videos are stored in `data/sample_video/`. These were used for development, tuning, and manual validation.

| File | Source | Purpose |
|---|---|---|
| `entrance_area_720p.mp4` | Internal — BLK-HDPTZ12 parking lot surveillance | **Primary demo video** (default `DEMO_VIDEO`). 720p, fixed overhead camera. Used for all CLI demos and most rule-firing validation. |
| `outside_entry_720p.mp4` | Internal | Second outdoor entrance scene. Used to validate MOG2 background calibration on a different static scene. |
| `night_surveillance_Am8tq-0FQJU.mp4` | YouTube — [https://www.youtube.com/watch?v=Am8tq-0FQJU](https://www.youtube.com/watch?v=Am8tq-0FQJU) | Night-scene footage. Used to validate `LightingAnalysis` classification (RULE-01 and RULE-05 require is_night=True). |
| `Qubo Outdoor Security Camera - Person Detection.mp4` | YouTube — [https://www.youtube.com/watch?v=eHS1GrJeXEE](https://www.youtube.com/watch?v=eHS1GrJeXEE) | Qubo outdoor camera. 1920×1080, 30fps, 18s. Person walking through outdoor scene. Used to validate BLIP person detection on a real consumer-grade camera feed. |

**Video selection rationale:**

- The primary video provides a stable overhead angle matching the "fixed drone" model assumption — necessary for MOG2 to build a good background model.
- The night video validates the lighting detector independently of wall-clock time, which is critical because the demo video may be replayed at any time of day.
- The Qubo video tests BLIP on a consumer camera with real compression artifacts, validating caption quality on lower-quality input.

---

<a id="10-design-decisions"></a>
## 10. Design Decisions

### Why two databases?

SQLite and ChromaDB serve fundamentally different query types:

| Query type | Backend | Example |
|---|---|---|
| Exact structured | SQLite | "show alerts with severity=high" |
| Time-range | SQLite | "what happened between 22:00 and 06:00" |
| Semantic similarity | ChromaDB | "show frames where someone was near the entrance" |
| Count by type | Both + track_merge | "how many people were detected" |

SQL `LIKE` on a caption string is neither reliable nor scalable for semantic search. A pure vector database makes exact time-range queries awkward. Using both gives the full spectrum and routes automatically.

### Why deterministic rules before the LLM?

Alert logic cannot hallucinate. The rules engine evaluates pure boolean conditions on structured data. The LangChain agent receives already-fired alerts as context and adds narrative — it cannot suppress a rule decision or override its severity. This makes the alert path auditable, reproducible, and free of LLM latency.

### Why BLIP on crops instead of the full frame?

BLIP on a full 1920×1080 frame with a small moving object in one corner produces captions dominated by the static background ("a parking lot with trees"). Cropping to just the foreground object forces BLIP to describe what actually changed: "a man in a dark jacket walking toward the door."

### Why LangChain 0.3 specifically?

LangChain 1.x restructured the `create_react_agent` + `AgentExecutor` API with breaking changes. Version 0.3.0 is the last stable release using the interface this project is built on. The `langchain-core 0.3` `on_chain_start` `serialized=None` edge case is patched in `SafeStdOutCallbackHandler`.

### Why lighting detection instead of wall-clock time?

Pre-recorded footage is often replayed outside the hours it was recorded. Detecting night from frame brightness makes RULE-01 and RULE-05 correct regardless of when the analysis runs. The V-channel of HSV is used (not raw luminance) because it is perceptually closer to human brightness perception and less affected by colour casts from artificial lighting.

### Why numpy two-step install?

`langchain==0.3.0` metadata declares `numpy<2.0.0`. `opencv-python>=4.9` requires `numpy>=2`. At runtime both work fine with `numpy==2.4.6`. The conflict exists only in pip's dependency solver. Installing langchain with `--no-deps` bypasses the solver while keeping the actual working runtime.

---

<a id="11-troubleshooting-debugging"></a>
## 11. Troubleshooting & Debugging

<a id="111-environment-installation"></a>
### 11.1 Environment & Installation

---

**`ERROR: ResolutionImpossible` during `pip install -r requirements.txt`**

```
ERROR: Cannot install langchain==0.3.0 and numpy>=2.0 because these package versions
have conflicting dependencies.
```

Cause: `langchain==0.3.0` metadata declares `numpy<2.0.0`. OpenCV requires `numpy>=2`.  
Fix: Use the two-step install — never install both files together:

```bash
pip install -r requirements.txt                      # step 1: core deps
pip install --no-deps -r requirements-langchain.txt  # step 2: langchain without solver
```

---

**`ModuleNotFoundError: No module named 'langchain'`** (after two-step install)

Cause: `--no-deps` skipped a transitive dependency that is not listed in `requirements-langchain.txt`.  
Fix: Install the missing package individually, then add it to `requirements-langchain.txt`:

```bash
pip install <missing-package>
```

---

**`torch` uses CPU even though MPS/CUDA is available**

Symptom: BLIP is slow (>5s per crop).  
Check:

```python
import torch
print(torch.backends.mps.is_available())   # Apple Silicon
print(torch.cuda.is_available())           # NVIDIA
```

Fix: `BLIPAnalyzer` auto-selects CUDA → MPS → CPU. If the right device is not being picked, check your `torch` build:

```bash
python3 -c "import torch; print(torch.__version__)"
# For Apple Silicon should end in  e.g. 2.x.x
# For CUDA builds should show e.g. 2.x.x+cu118
```

Reinstall torch for your platform if the build is wrong.

---

**`ImportError: libGL.so.1: cannot open shared object file`** (Linux)

Cause: OpenCV requires system OpenGL libs.  
Fix:

```bash
sudo apt-get install libgl1-mesa-glx   # Ubuntu/Debian
```

---

<a id="112-configuration-env"></a>
### 11.2 Configuration & `.env`

---

**Pipeline runs but OpenAI calls fail — `AuthenticationError`**

```
openai.AuthenticationError: Incorrect API key provided
```

Cause: `OPENAI_API_KEY` is empty or wrong.  
Check:

```bash
python3 -c "from config import OPENAI_MODEL; import os; print(os.getenv('OPENAI_API_KEY','MISSING'))"
```

If it prints `MISSING`, your `.env` file is either absent or `load_dotenv()` did not run before `os.getenv()`.  
Critical: `config.py` calls `load_dotenv()` at the top of the file. Any module that reads env vars must `import config` (directly or transitively) **before** calling `os.getenv()`. Never call `load_dotenv()` after `config` has already been imported.

---

**Telegram notifications not arriving — no error in logs**

Cause A: `TELEGRAM_BOT_TOKEN` or `TELEGRAM_CHAT_ID` is empty.

```bash
python3 -c "from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID; print(repr(TELEGRAM_BOT_TOKEN), repr(TELEGRAM_CHAT_ID))"
```

Both must be non-empty strings. If either prints `''`, fill in `.env` and restart.

Cause B: `load_dotenv()` ran after `config.py` was already imported (UI-specific bug).  
Symptom: works from CLI (`python3 main.py`) but not from Streamlit (`streamlit run ui/app.py`).  
Fix: `config.py` already has `load_dotenv()` at line 3. Do **not** call `load_dotenv()` anywhere else (e.g. inside `ui/app.py`) after importing config — the env vars will already be stale empty strings by then.

Cause C: Alert severity is not `HIGH`. The notifier silently ignores `MEDIUM` and `LOW` alerts by design.

---

**Telegram 400 Bad Request**

```
requests.exceptions.HTTPError: 400 Client Error: Bad Request for url: .../sendMessage
```

Cause: `parse_mode="Markdown"` was set and the message contains characters Telegram's Markdown v1 parser rejects (backticks, square brackets in certain positions).  
Fix: The notifier uses plain text (no `parse_mode`). If you add custom formatting, use `parse_mode="HTML"` with `<b>` / `<code>` tags instead of Markdown.

---

**Telegram 404 Not Found when testing bot token**

```
{"ok": false, "error_code": 404, "description": "Not Found"}
```

Cause: The URL format used to test was wrong, or the bot token has a typo.  
Fix: Do not test via `getUpdates` URL manually. Instead:
1. Message `@userinfobot` on Telegram to get your chat ID.
2. Test with a simple curl:

```bash
curl "https://api.telegram.org/bot<TOKEN>/sendMessage" \
  -d "chat_id=<CHAT_ID>&text=test"
```

---

<a id="113-video-processing"></a>
### 11.3 Video Processing

---

**Every frame shows `[SKIP] No foreground detected` — nothing is ever analyzed**

Cause A: MOG2 warmup did not complete — the background model never learned the scene.  
Check: Lower `MOG2_WARMUP_FRAMES` in `config.py` (default 5). If the video is very short, even 5 frames can be too many.

Cause B: `MOG2_VAR_THRESHOLD` is too high — foreground objects are within the learned variance.  
Fix: Lower `MOG2_VAR_THRESHOLD` from 120 to 50–80 to make detection more sensitive.

Cause C: `MIN_CONTOUR_AREA` is too large for the resolution.  
Fix: On a 480p video, objects may be <2000px². Lower `MIN_CONTOUR_AREA` to 500–1000.

Cause D: The video file contains no actual motion (static scene throughout).  
Check: Open the video in any player and confirm movement is visible.

---

**Every sampled frame triggers a detection — flood of false positives**

Cause A: `MOG2_VAR_THRESHOLD` is too low — minor lighting variation is classified as foreground.  
Fix: Increase `MOG2_VAR_THRESHOLD` toward 150–200.

Cause B: The scene has flickering artificial lighting (fluorescent, street light cycling).  
Fix: Increase `MOG2_WARMUP_FRAMES` to 15–20 so the model learns the flicker pattern.

Cause C: Wind causing foliage or flags to move constantly.  
Fix: Increase `MIN_CONTOUR_AREA` to filter small oscillating blobs.

---

**`cv2.error: (-215:Assertion failed) !_src.empty()` in background_subtractor**

Cause: Video file could not be opened, or a frame read returned an empty array.  
Check:

```python
import cv2
cap = cv2.VideoCapture("data/sample_video/your_video.mp4")
print(cap.isOpened())   # must be True
ret, frame = cap.read()
print(ret, frame is not None)
```

If `isOpened()` is `False`: the file path is wrong, the file is corrupt, or the codec is unsupported.  
Fix: Re-encode the video with ffmpeg:

```bash
ffmpeg -i input.mp4 -c:v libx264 -preset fast output.mp4
```

---

**`DEMO_VIDEO` not found — `FileNotFoundError`**

```
FileNotFoundError: data/sample_video/entrance_area_720p.mp4
```

Fix: Place an MP4 in `data/sample_video/` and update `DEMO_VIDEO` in `config.py` to match the filename exactly (case-sensitive on macOS/Linux).

---

<a id="114-blip-vlm"></a>
### 11.4 BLIP / VLM

---

**BLIP produces generic captions — "a car in a parking lot" for every crop**

Cause: MOG2 contours are too loose — the "crop" is nearly the full frame.  
Check: Print `crop.bbox` for each `ObjectCrop`. If width × height is close to frame resolution, the contour is not isolating a real object.  
Fix: Increase `MOG2_VAR_THRESHOLD` and `MIN_CONTOUR_AREA` to tighten contour selection.

---

**BLIP download hangs or fails on first run**

Cause: HuggingFace model download interrupted or network restricted.  
Fix:

```bash
# Pre-download manually
python3 -c "
from transformers import BlipProcessor, BlipForConditionalGeneration
BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
print('done')
"
```

Cache location: `~/.cache/huggingface/hub/`. Requires ~3.7 GB free.

---

**`OSError: [Errno 28] No space left on device` during BLIP load**

Cause: HuggingFace cache disk is full.  
Fix: Free space or redirect cache:

```bash
export HF_HOME=/path/to/larger/disk/.cache/huggingface
```

---

**`object_type` always returns `"unknown"`**

Cause: BLIP caption does not contain any keyword from `vlm/constants.py`.  
Debug:

```python
from vlm.constants import extract_object_type
caption = "a blurry shape moving in the frame"
print(extract_object_type(caption))   # → "unknown"
```

This is expected for low-quality crops. The alert rules and tracker still run on the `"unknown"` type — only RULE-02 and RULE-03 require a motor vehicle match.

---

<a id="115-alert-rules"></a>
### 11.5 Alert Rules

---

**RULE-01 / RULE-05 never fire even on night footage**

Cause: `LightingAnalysis.is_night` is `False` — the video is bright enough to be classified as day.  
Debug:

```python
from vlm.lighting_detector import analyze_frame_lighting
import cv2, numpy as np

cap = cv2.VideoCapture("data/sample_video/your_video.mp4")
ret, frame = cap.read()
result = analyze_frame_lighting(frame)
print(result)   # brightness, dark_pixel_ratio, is_night
```

Fix: Lower `LIGHTING_NIGHT_MEAN_THRESHOLD` (default 90) or `LIGHTING_NIGHT_DARK_PIXEL_RATIO` (default 0.55) in `config.py` to match your footage's actual brightness levels.

---

**RULE-04 fires on the first frame but never fires the second prolonged alert**

Cause: `LOITER_THRESHOLD_SECONDS` (default 15s) is longer than the total video duration, or the person leaves frame before the threshold is reached.  
Fix for testing: Set `LOITER_THRESHOLD_SECONDS = 5` in `config.py`.  
Production value: `300` (5 minutes).

---

**RULE-03 never fires even after multiple vehicle appearances**

Cause: `VehicleTracker` is not matching the re-entry to the same track — it creates a new track each time, so `entry_count` never exceeds `REPEAT_ENTRY_LIMIT`.  
Debug:

```python
# In pipeline output look for track events:
# [track] T-1  entry_count=1  is_new_entry=True
# [track] T-2  entry_count=1  is_new_entry=True   ← two separate tracks, not one
```

Cause: Vehicle moved too far between appearances (exceeds `max_match_distance`).  
Fix: Increase `VEHICLE_TRACK_MAX_FRAME_GAP` or `VEHICLE_TRACK_MAX_VELOCITY_PX_PER_FRAME` in `config.py`.

---

**All rules fire every frame instead of once per track**

Cause: `AlertRulesEngine` state was not reset between runs. Each run calls `rules.reset()` via `VideoProcessor` init — verify this is happening.  
Check: In `main.py`, `AlertRulesEngine()` is re-instantiated fresh. In the UI, each session calls `reset_session()` which creates a new `AlertRulesEngine`. If you reuse an engine instance across multiple videos, call `rules.reset()` manually.

---

<a id="116-langchain-agent"></a>
### 11.6 LangChain Agent

---

**Agent never calls any tool — returns a raw text response immediately**

Cause A: Wrong LangChain version. LangChain 1.x changed the ReAct prompt format and tool-calling API.  
Check:

```bash
python3 -c "import langchain; print(langchain.__version__)"  # must be 0.3.x
```

Cause B: The prompt template does not match what `create_react_agent` expects.  
The template must have `{tools}`, `{tool_names}`, `{agent_scratchpad}`, and `{chat_history}` placeholders in the correct positions.

---

**`KeyError: 'output'` from AgentExecutor**

Cause: Agent hit `max_iterations=6` without producing a `Final Answer:`.  
Fix: `AgentExecutor` is configured with `handle_parsing_errors=True` which returns a fallback string. If you still see this, check that `AGENT_VERBOSE=True` in `config.py` and read the ReAct chain trace to see where it stalls.

---

**`TypeError: on_chain_start() got an unexpected keyword argument 'serialized'`**

Cause: `langchain-core 0.3` passes `serialized=None` to `on_chain_start`, which the base `StdOutCallbackHandler` does not expect in some versions.  
Fix: Already patched in `agent/callbacks.py` via `SafeStdOutCallbackHandler`. Ensure `security_agent.py` uses this handler, not the base `StdOutCallbackHandler`.

---

**Agent always uses `log_event` and never `trigger_alert`**

This is expected behaviour. `trigger_alert` is for suspicious events not already caught by the rules engine. Since the rules engine catches all hard-coded cases, the agent uses `trigger_alert` only for novel patterns. You can verify by checking `store.get_alerts()` — all `rule_id="AGENT"` rows came from the agent.

---

<a id="117-storage"></a>
### 11.7 Storage

---

**`sqlite3.OperationalError: no such table: events`**

Cause: `EventStore.__init__` did not run `create_tables()`, or the DB file was deleted mid-run.  
Fix: `EventStore()` creates tables on init. Re-instantiate the store.

---

**`chromadb.errors.InvalidCollectionException`** or ChromaDB returns no results after a re-run

Cause: Stale ChromaDB data directory from a previous run with a different schema or embedding model.  
Fix: Delete `data/chroma/` before re-running:

```bash
rm -rf data/chroma/
python3 main.py
```

`main.py` does this automatically at startup. If running the UI, each session gets its own isolated ChromaDB directory under `data/sessions/<id>/chroma/`.

---

**ChromaDB first query is very slow (10–30s)**

Cause: `sentence-transformers/all-MiniLM-L6-v2` is being downloaded on first use.  
Expected: This happens once. After the first run the model is cached at `~/.cache/huggingface/`.

---

<a id="118-streamlit-ui"></a>
### 11.8 Streamlit UI

---

**UI hangs after clicking "Start Analysis" — no progress**

Cause A: The pipeline is running but events are not yielding fast enough. BLIP inference on CPU can take 3–8s per crop.  
Expected behaviour: the progress bar and live frame should update after the MOG2 warmup period (first 5 sampled frames).

Cause B: An exception was raised inside `_run_pipeline` and swallowed.  
Fix: Run the same video from the CLI first:

```bash
python3 main.py
```

Any exception will print to the terminal and give the exact traceback.

---

**`StreamlitAPIException: Values for st.* cannot be set outside of the main thread`**

Cause: A background thread (e.g. Telegram daemon thread) tried to call a Streamlit API.  
Fix: Telegram notifications run in daemon threads and must never call any `st.*` function. The current implementation is correct — if you add new background work, pass results back via `st.session_state` from the main thread only.

---

**Uploaded video disappears after page refresh**

Expected: Streamlit re-runs the script on every interaction. `save_upload()` writes the file to `data/uploads/` persistently, but `st.session_state` is reset on refresh (not on re-runs triggered by widget interaction). The session and uploaded video path survive widget interactions but not a full browser page refresh.

---

<a id="119-quick-diagnostic-checklist"></a>
### 11.9 Quick Diagnostic Checklist

Run this before opening a bug report:

```bash
# 1. Python version
python3 --version                        # must be 3.10+

# 2. Key package versions
python3 -c "import cv2; print('opencv', cv2.__version__)"
python3 -c "import torch; print('torch', torch.__version__)"
python3 -c "import langchain; print('langchain', langchain.__version__)"
python3 -c "import chromadb; print('chromadb', chromadb.__version__)"
python3 -c "import numpy; print('numpy', numpy.__version__)"

# 3. Env vars loaded
python3 -c "
from config import OPENAI_MODEL, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
import os
print('OPENAI_API_KEY:', 'SET' if os.getenv('OPENAI_API_KEY') else 'MISSING')
print('TELEGRAM_BOT_TOKEN:', 'SET' if TELEGRAM_BOT_TOKEN else 'MISSING')
print('TELEGRAM_CHAT_ID:', 'SET' if TELEGRAM_CHAT_ID else 'MISSING')
"

# 4. Video file accessible
python3 -c "
import cv2
from config import SAMPLE_VIDEO_DIR, DEMO_VIDEO
import os
path = os.path.join(SAMPLE_VIDEO_DIR, DEMO_VIDEO)
cap = cv2.VideoCapture(path)
print('video opens:', cap.isOpened())
print('total frames:', int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
cap.release()
"

# 5. BLIP cached
python3 -c "
from transformers import BlipProcessor
try:
    BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base', local_files_only=True)
    print('BLIP: cached')
except:
    print('BLIP: NOT cached — will download on first run (~3.7 GB)')
"

# 6. Unit tests
python3 -m pytest tests/unit/ -q     # should print: 64 passed
```

---

<a id="1110-expected-normal-output"></a>
### 11.10 Expected Normal Output

When the pipeline is working correctly, a CLI run looks like this:

```
============================================================
  FLYING POLICE
============================================================

  === SESSION INFO ===
  Monitoring : MAIN GATE
  Resolution : 1280x720 px
  Frame rate : 25.0 fps
  Duration   : 18s (450 frames, sampling every 2)
  Sampled    : ~225 frames to process
  === BACKGROUND CALIBRATION (5 warmup frames) ===
  Calibrating background model... (frame 1/5)
  Calibrating background model... (frame 2/5)
  ...
  Calibrating background model... (frame 5/5)

  Frame 006 (raw 012) — detected
    [BLIP] a man in a dark jacket walking toward a door
    [objects] person  [activity] entering  [lighting] night
    [track] T-1  entry_count=1  new_entry=True
    [RULE-01] Person detected at main_gate at night  HIGH
    [agent] Logged: person approaching gate at night; RULE-01 pre-fired

  Frame 008 (raw 016) — skip  (no foreground)

  Frame 010 (raw 020) — detected
  ...
```

If you see only `skip` lines with no `detected` frames at all, jump to §11.3.  
If you see `detected` but no alerts, check §11.5.  
If alerts fire but no Telegram message arrives, check §11.2.

---

<a id="12-extension-guide"></a>
## 12. Extension Guide

<a id="121-adding-a-new-alert-rule"></a>
### 12.1 Adding a New Alert Rule

**Step 1 — Name and define the rule**

Pick the next rule ID (`RULE-06`). Decide:
- What structured data it checks (objects, activity, location, is_night, track state)
- Severity (HIGH or MEDIUM)
- Whether it should dedup per track or fire every frame

**Step 2 — Add state to `AlertRulesEngine` (if needed)**

If the rule needs dedup or dwell timing, add a set or dict to `__init__` and `reset()`:

```python
# agent/alert_rules.py

class AlertRulesEngine:
    def __init__(self):
        ...
        self._rule06_issued: set = set()   # track IDs that triggered RULE-06

    def reset(self):
        ...
        self._rule06_issued.clear()
```

**Step 3 — Add the evaluation logic inside `evaluate()`**

Add a new block at the end of `evaluate()`, before the `return alerts` line:

```python
# RULE-06: bicycle detected anywhere
if "bicycle" in objects and track_id not in self._rule06_issued:
    self._rule06_issued.add(track_id)
    alerts.append(Alert(
        rule_id="RULE-06",
        frame_id=frame_id,
        message=f"Bicycle detected at {location}",
        severity="medium",
    ))
```

`track_id` comes from `self.last_track_update.track_id` if a track update exists, else use `f"{location}_{frame_id}"` as a fallback key.

**Step 4 — Add to the README and CODE_DOCS alert tables**

Update the Alert Rules table in both `README.md` and §5 of this document.

**Step 5 — Add a unit test**

```python
# tests/unit/test_alerts.py

def test_rule06_bicycle():
    engine = AlertRulesEngine()
    # Simulate track update
    engine.last_track_update = make_track(track_id="T-1", is_new_entry=True)
    alerts = engine.evaluate(
        frame_id=1, timestamp="2026-01-01T10:00:00", location="main_gate",
        objects=["bicycle"], activity="entering", description="a bicycle",
    )
    assert any(a.rule_id == "RULE-06" for a in alerts)
```

---

<a id="122-adding-a-new-agent-tool"></a>
### 12.2 Adding a New Agent Tool

**Step 1 — Define the tool in `agent/tools.py`**

```python
@tool
def summarise_session(query: str) -> str:
    """Return a one-line summary of all events in the current session."""
    events = _store.get_all_events()
    if not events:
        return "No events recorded yet."
    return f"{len(events)} events logged. Latest: {events[-1]['message']}"
```

Rules:
- Docstring is the tool description the LLM reads — make it precise about when to use it
- Return a plain string (the agent reads it as an `Observation`)
- Never raise exceptions — catch internally and return an error string

**Step 2 — Register the tool in `SecurityAgent`**

```python
# agent/security_agent.py

from agent.tools import log_event, trigger_alert, query_history, query_track_positions, summarise_session

tools = [log_event, trigger_alert, query_history, query_track_positions, summarise_session]
```

**Step 3 — Update the system prompt**

Add the new tool to the tools list in `PROMPT_TEMPLATE` so the agent knows it exists:

```
Tools available:
- log_event: ...
- trigger_alert: ...
- query_history: ...
- query_track_positions: ...
- summarise_session: Use when asked for an overall session summary   ← add this
```

**Step 4 — Test in isolation**

```python
from agent.tools import init_tools, summarise_session
from storage.event_store import EventStore
from storage.frame_index import FrameIndex

store = EventStore()
index = FrameIndex()
init_tools(store, index)

store.log_event(1, "Test event", severity="low")
print(summarise_session("anything"))   # should return "1 events logged..."
```

---

<a id="123-swapping-blip-for-a-different-vlm"></a>
### 12.3 Swapping BLIP for a Different VLM

The VLM interface is `BLIPAnalyzer.analyze_crop(crop: ObjectCrop) -> CropAnalysis`. Any replacement must satisfy this contract.

**Step 1 — Create `vlm/new_analyzer.py`**

```python
from dataclasses import dataclass
from vlm.background_subtractor import ObjectCrop
from vlm.constants import extract_object_type, extract_color

@dataclass
class CropAnalysis:
    caption: str
    object_type: str
    color: str
    bbox: tuple

class NewAnalyzer:
    def __init__(self):
        # load your model here
        pass

    def analyze_crop(self, crop: ObjectCrop) -> CropAnalysis:
        # run inference, return CropAnalysis
        caption = "..."   # your model output
        return CropAnalysis(
            caption=caption,
            object_type=extract_object_type(caption),
            color=extract_color(caption),
            bbox=crop.bbox,
        )

    def analyze_frame(self, crops):
        # same signature as BLIPAnalyzer.analyze_frame
        from vlm.blip_analyzer import FrameVLMResult
        results = [self.analyze_crop(c) for c in crops]
        objects = list({r.object_type for r in results if r.object_type != "unknown"})
        description = "; ".join(r.caption for r in results)
        return FrameVLMResult(crops=results, objects=objects, raw_description=description)
```

**Step 2 — Swap the import in `pipeline/video_processor.py`**

```python
# from vlm.blip_analyzer import BLIPAnalyzer       ← comment out
from vlm.new_analyzer import NewAnalyzer as BLIPAnalyzer  # drop-in replacement
```

No other file needs to change — the rest of the pipeline uses the `FrameVLMResult` dataclass interface, not BLIP directly.

---

<a id="124-adding-a-new-location"></a>
### 12.4 Adding a New Location

Locations are validated only by `config.py` and referenced in alert messages. To add `"parking_lot"`:

**Step 1 — Add to `LOCATIONS` in `config.py`**

```python
LOCATIONS = ["main_gate", "garage", "perimeter", "parking_lot"]
```

**Step 2 — Add RULE-05 coverage if needed**

RULE-05 checks `location == "perimeter"` explicitly. If you want the same rule to cover the new location, update the condition in `agent/alert_rules.py`:

```python
PERIMETER_LOCATIONS = {"perimeter", "parking_lot"}

# In evaluate():
if location in PERIMETER_LOCATIONS and is_night:
    ...
```

**Step 3 — Update the Streamlit sidebar**

```python
# ui/app.py — location selector
location = st.sidebar.selectbox(
    "Location",
    ["main_gate", "garage", "perimeter", "parking_lot"],   # add here
)
```

---

<a id="125-adding-a-new-known-vehicle"></a>
### 12.5 Adding a New Known Vehicle

To whitelist a vehicle from RULE-02, add its description to `KNOWN_VEHICLES` in `config.py`:

```python
KNOWN_VEHICLES = ["blue ford f150", "blue truck", "white toyota camry"]
```

Matching is a case-insensitive substring check against the full BLIP caption. The string should match what BLIP typically produces for that vehicle — test with `analyze_full_image()` on a sample frame first.

---

<a id="13-scripts-reference"></a>
## 13. Scripts Reference

<a id="131-scriptscapture-validation-setpy"></a>
### 13.1 `scripts/capture_validation_set.py`

Runs the full pipeline on a video and saves annotated fixtures (frames + crops + JSON records) for regression testing.

```
Usage:
  python scripts/capture_validation_set.py [OPTIONS]

Options:
  --video PATH          Input video file
                        Default: data/sample_video/entrance_area_720p.mp4

  --output PATH         Output directory for fixtures
                        Default: data/validation_fixtures/

  --every N             Save a fixture every N BLIP detections (or sampled frames
                        if --capture-on=sampled)
                        Default: VALIDATION_CAPTURE_EVERY_N (30)

  --max N               Maximum number of fixtures to save
                        Default: VALIDATION_MAX_CAPTURES (15)

  --sample-every N      Process every Nth raw video frame
                        Default: 2

  --capture-on MODE     sampled   — save every Nth sampled frame regardless
                        detection — save every Nth frame that had a BLIP hit (default)

  --skip-agent          Skip LangChain agent; capture BLIP output only.
                        Use this when no OPENAI_API_KEY is available.
```

**What gets saved per fixture:**

```
data/validation_fixtures/
└── frame_0042/
    ├── full.jpg          Annotated full frame (bbox drawn)
    ├── crop_00.jpg       First ObjectCrop image
    ├── crop_01.jpg       Second ObjectCrop image (if multiple)
    └── record.json       Full CaptureRecord as JSON
        {
          "frame_id": 42,
          "raw_idx": 84,
          "timestamp": "2026-01-01T00:01:24",
          "location": "main_gate",
          "has_foreground": true,
          "activity": "entering",
          "threat_level": "high",
          "blip": [{"caption": "...", "object_type": "person", ...}],
          "alerts": [{"rule_id": "RULE-01", "severity": "high", ...}],
          "agent_response": "Person approaching gate...",
          "telemetry": {...},
          "vehicle_track": {...}
        }
```

**Example runs:**

```bash
# Full capture with agent (needs OPENAI_API_KEY)
python scripts/capture_validation_set.py

# BLIP only, no API calls, custom video
python scripts/capture_validation_set.py \
    --video data/sample_video/outside_entry_720p.mp4 \
    --skip-agent --every 10 --max 20

# Dense capture every sampled frame (good for debugging MOG2)
python scripts/capture_validation_set.py \
    --capture-on sampled --every 1 --max 50
```

**After capturing**, run the validation test suite:

```bash
python3 -m pytest tests/validation/ -v
```

If no fixtures exist, validation tests are automatically skipped with a message:
```
SKIPPED: No validation fixtures — run scripts/capture_validation_set.py first
```

---

<a id="132-scriptsgenerate-architecture-pptpy"></a>
### 13.2 `scripts/generate_architecture_ppt.py`

Generates the 7-slide `Flying_Police_Architecture.pptx` using `python-pptx`.

```bash
python scripts/generate_architecture_ppt.py
# Output: Flying_Police_Architecture.pptx (in project root)
```

Slides generated:
1. Title slide (project name + tech stack pills)
2. Full pipeline flow (numbered step cards)
3. VLM Layer — MOG2 + BLIP detail
4. Alert Rules Engine
5. Dual-Database Storage
6. LangChain ReAct Agent
7. Telegram Notifications

To modify slides: edit the `add_slide_*()` functions in the script. The deck uses a dark theme (`#0D1117` background) with accent colours per component category. Fonts require `Fira Code` or fall back to `Courier New`.

---

<a id="14-environment-variables-reference"></a>
## 14. Environment Variables Reference

### `.env` file

Create this file in the project root. It is loaded by `config.py` via `python-dotenv` at import time.

```bash
# .env — copy this template and fill in your values

# ── Required ─────────────────────────────────────────────────────────────────
OPENAI_API_KEY=sk-...           # From platform.openai.com/api-keys
                                # Used by: agent/security_agent.py (GPT-4o-mini)

# ── Optional: Telegram alerts ────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN=             # From @BotFather on Telegram (/newbot)
TELEGRAM_CHAT_ID=               # From @userinfobot on Telegram (your numeric ID)
                                # Leave blank to disable — pipeline runs without it
```

**How to get each value:**

| Variable | Steps |
|---|---|
| `OPENAI_API_KEY` | Log in to [platform.openai.com](https://platform.openai.com) → API Keys → Create new secret key |
| `TELEGRAM_BOT_TOKEN` | Open Telegram → search `@BotFather` → send `/newbot` → follow prompts → copy the token |
| `TELEGRAM_CHAT_ID` | Open Telegram → search `@userinfobot` → send any message → copy the `Id` field |

**Security notes:**
- Never commit `.env` to git — it is already in `.gitignore`
- The bot token gives full control of your Telegram bot — treat it like a password
- `OPENAI_API_KEY` incurs charges — set a monthly spend limit at [platform.openai.com/account/limits](https://platform.openai.com/account/limits)

---

<a id="15-module-import-graph"></a>
## 15. Module Import Graph

Shows which modules depend on which. Use this to understand the effect of changing a module and to diagnose circular import errors.

```
config.py
  └── (imported by almost every module — always first)

vlm/constants.py
  └── vlm/blip_analyzer.py
  └── query/query_engine.py

vlm/background_subtractor.py
  └── pipeline/video_processor.py

vlm/blip_analyzer.py
  ├── vlm/constants.py
  └── pipeline/video_processor.py

vlm/lighting_detector.py
  └── pipeline/video_processor.py

agent/vehicle_tracker.py
  ├── config.py
  └── pipeline/video_processor.py

agent/vehicle_context.py
  ├── vlm/blip_analyzer.py  (FrameVLMResult type)
  └── pipeline/video_processor.py

agent/alert_rules.py
  ├── agent/vehicle_tracker.py  (TrackUpdate type)
  ├── config.py
  └── pipeline/video_processor.py

agent/tools.py
  ├── storage/event_store.py
  ├── storage/frame_index.py
  └── agent/security_agent.py

agent/callbacks.py
  └── agent/security_agent.py

agent/security_agent.py
  ├── agent/tools.py
  ├── agent/callbacks.py
  ├── storage/event_store.py
  ├── storage/frame_index.py
  └── pipeline/video_processor.py

storage/event_store.py
  ├── config.py
  ├── agent/tools.py
  └── query/query_engine.py

storage/frame_index.py
  ├── config.py
  ├── agent/tools.py
  └── query/query_engine.py

pipeline/frame_preview.py
  └── pipeline/video_processor.py
  └── ui/app.py

pipeline/session.py
  └── ui/app.py

pipeline/video_processor.py
  ├── config.py
  ├── vlm/background_subtractor.py
  ├── vlm/blip_analyzer.py
  ├── vlm/lighting_detector.py
  ├── agent/alert_rules.py
  ├── agent/security_agent.py
  ├── agent/vehicle_context.py
  ├── storage/event_store.py
  ├── storage/frame_index.py
  ├── notifications/telegram_notifier.py
  ├── pipeline/frame_preview.py
  └── data/telemetry.py

notifications/telegram_notifier.py
  └── config.py

query/track_merge.py
  └── query/query_engine.py

query/query_engine.py
  ├── storage/event_store.py
  ├── storage/frame_index.py
  ├── query/track_merge.py
  └── vlm/constants.py

main.py
  ├── pipeline/video_processor.py
  ├── storage/event_store.py
  ├── storage/frame_index.py
  ├── agent/alert_rules.py
  ├── agent/security_agent.py
  ├── query/query_engine.py
  └── config.py

ui/app.py
  ├── pipeline/video_processor.py
  ├── pipeline/session.py
  ├── pipeline/frame_preview.py
  ├── storage/event_store.py
  ├── storage/frame_index.py
  ├── agent/alert_rules.py
  ├── agent/security_agent.py
  ├── query/query_engine.py
  └── config.py
```

**Safe to change in isolation** (no downstream dependents inside this project):
- `notifications/telegram_notifier.py`
- `data/telemetry.py`
- `data/simulated_frames.py`
- `scripts/*`
- `agent/callbacks.py`

**High-impact modules** (changing these breaks many dependents):
- `config.py` — imported everywhere
- `pipeline/video_processor.py` — owns `PipelineEvent` used by both `main.py` and `ui/app.py`
- `storage/event_store.py` and `storage/frame_index.py` — used by agent tools, query engine, and UI

---

<a id="16-known-limitations"></a>
## 16. Known Limitations

| Limitation | Detail | Workaround |
|---|---|---|
| Fixed camera only | MOG2 builds a background model of a static scene. A PTZ (pan-tilt-zoom) or moving camera will produce false positives on every frame as the background shifts. | Use a fixed-mount camera. For PTZ, replace MOG2 with an optical-flow or feature-matching approach. |
| Single location per session | `VideoProcessor` takes one `location` string for the entire video. All events are tagged with the same location. | Run separate sessions for different camera angles. |
| BLIP confidence not exposed | `BlipForConditionalGeneration.generate()` returns token IDs only. There is no confidence score to filter low-quality captions. | Add `output_scores=True` to `generate()` and compute sequence log-probability if filtering is needed. |
| No face or license plate recognition | Object types are limited to the keyword taxonomies in `vlm/constants.py`. Faces and plates are not detected. | Integrate a dedicated model (e.g. `easyocr` for plates, `deepface` for faces) as an additional VLM layer after BLIP. |
| Night detection depends on frame brightness | `LightingAnalysis` uses V-channel mean and dark-pixel ratio. Very bright artificial lighting (floodlit parking lot at night) may be classified as day, suppressing RULE-01 and RULE-05. | Lower `LIGHTING_NIGHT_MEAN_THRESHOLD` for scenes with high artificial lighting, or add a time-of-day fallback using video file metadata. |
| BLIP on very small crops is unreliable | Crops smaller than ~80×80px produce generic or incorrect captions because BLIP was trained on larger images. | Increase `MIN_CONTOUR_AREA` to ensure only sufficiently large objects are sent to BLIP. |
| No multi-object tracking across occlusions | `VehicleTracker` matches by last known position. If two objects cross paths, their tracks can swap identities. | Replace with a proper MOT algorithm (e.g. SORT, DeepSORT) if track identity under occlusion matters. |
| ChromaDB cosine scores are not calibrated | A score of `0.85` does not mean "85% confident." Scores are relative — the highest-scored result in a query set is the best match, but the absolute value has no fixed meaning. | Use scores for ranking only. Filter by `score > 0.5` as a rough relevance gate; tune per dataset. |
| Session data is ephemeral | Each CLI run deletes and rebuilds `events.db` and `data/chroma/`. There is no cross-session history. | Use the UI with per-session directories under `data/sessions/` which persist until manually deleted. |
| Agent `max_iterations=6` | Long ReAct chains (multiple tool calls per frame) are cut off at 6 iterations. The agent returns a partial answer. | Increase `max_iterations` in `SecurityAgent.__init__` — note this increases latency and token cost per frame. |

---

<a id="17-performance-expectations"></a>
## 17. Performance Expectations

Measured on Apple Silicon (M-series, MPS) and Intel CPU. Times are per **sampled frame with a detection** (MOG2 found foreground).

| Stage | Apple Silicon (MPS) | Intel CPU | Notes |
|---|---|---|---|
| MOG2 background subtraction | < 5 ms | < 10 ms | Vectorised C++ in OpenCV |
| Lighting detection | < 1 ms | < 2 ms | NumPy V-channel mean |
| BLIP inference (per crop) | 300–600 ms | 2,000–5,000 ms | Largest bottleneck on CPU |
| VehicleTracker update | < 1 ms | < 1 ms | Pure Python arithmetic |
| AlertRulesEngine.evaluate() | < 1 ms | < 1 ms | Boolean checks on dicts |
| OpenAI API call (agent) | 800–2,000 ms | 800–2,000 ms | Network-bound; gpt-4o-mini |
| ChromaDB add_frame | 5–20 ms | 10–40 ms | Embedding + HNSW insert |
| ChromaDB query | 10–50 ms | 20–80 ms | Depends on collection size |
| SQLite write | < 1 ms | < 1 ms | Local file I/O |
| Telegram sendPhoto | 200–800 ms | 200–800 ms | Network-bound; daemon thread (non-blocking) |

**Overall throughput (frames/sec, end-to-end with agent):**

| Hardware | Approx. fps |
|---|---|
| Apple Silicon (MPS, agent on) | 0.5 – 1.5 fps |
| Apple Silicon (MPS, agent off) | 1.5 – 3.0 fps |
| Intel CPU (agent on) | 0.1 – 0.3 fps |
| Intel CPU (agent off) | 0.3 – 0.7 fps |

For a 30fps video sampled every 2nd frame (225 sampled frames from an 18s clip):

| Config | Estimated processing time |
|---|---|
| Apple Silicon + agent | 2 – 5 minutes |
| Apple Silicon, no agent | 1 – 2 minutes |
| Intel CPU + agent | 10 – 25 minutes |
| Intel CPU, no agent | 5 – 12 minutes |

**Tips to speed up:**

- Set `SAMPLE_EVERY_N_FRAMES = 4` to process 25% of raw frames — good for long recordings
- Disable the agent (`SecurityAgent` optional in `VideoProcessor`) for pure detection runs
- On CPU, consider reducing BLIP to `max_new_tokens=15` (faster at cost of shorter captions)
- Telegram notifications are fire-and-forget daemon threads — they add zero latency to the pipeline

---

<a id="18-query-result-interpretation"></a>
## 18. Query Result Interpretation

### ChromaDB similarity scores

`FrameIndex.query()` returns results with a `score` field in the range `[0, 1]` (cosine similarity, higher = more similar).

| Score range | Interpretation |
|---|---|
| `≥ 0.80` | Strong semantic match — the caption closely describes what you searched for |
| `0.60 – 0.79` | Moderate match — related topic, may include partial matches |
| `0.40 – 0.59` | Weak match — surface-level word overlap only |
| `< 0.40` | Poor match — effectively unrelated; likely returned because the index is small |

**Important:** scores are relative within a query result set, not absolute. In a small index (< 50 frames), even unrelated frames may score `0.60+` simply because they are the closest available match. Use scores for ranking, not as a confidence threshold.

**Example — interpreting a query result:**

```
Query: "person walking near entrance"
─────────────────────────────────────────────────────────────────
score=0.87  Frame 012 | "a man in a dark jacket approaching the gate"
score=0.74  Frame 031 | "a person standing near the door"
score=0.61  Frame 007 | "a figure moving along the wall"
score=0.43  Frame 019 | "a car parked in the driveway"     ← weak, different topic
```

Frame 012 and 031 are genuine matches. Frame 007 is a partial match (person but not "walking near entrance"). Frame 019 is noise — the engine returns up to `n_results=5` regardless of relevance.

### Count query results

`QueryEngine._query_count()` returns:

```python
{
    "type": "count",
    "label": "person",            # what was counted
    "count": 8,                   # raw track count from ChromaDB metadata
    "individuals": 3,             # after merge_tracks_into_individuals()
    "raw_tracks": [...],          # individual track details
    "merged_individuals": [...]   # deduplicated individuals
}
```

Use `individuals` for "how many people" questions — it applies union-find dedup to collapse fragmented tracks (same person appearing as T-1 and T-3 due to brief occlusion) into a single individual.

Use `count` only if you want the raw number of detected segments, not physical individuals.
