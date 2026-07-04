# Architecture Plan
## Flying Police

---

## 1. High-Level Data Flow

```
Simulated Frames + Telemetry
         │
         ▼
┌─────────────────────┐
│  Background         │  OpenCV MOG2
│  Subtractor         │  → binary mask → bounding box crops
└────────┬────────────┘
         │ object crops (or skip if empty mask)
         ▼
┌─────────────────────┐
│  VLM Analyzer       │  BLIP (blip-image-captioning-base)
│  (per crop)         │  → caption, object type, color
└────────┬────────────┘
         │ FrameAnalysis struct
         ▼
┌─────────────────────┐
│  Alert Rules Engine │  Deterministic rule check (no LLM)
│  (alert_rules.py)   │  → fires alerts before agent if rule matched
└────────┬────────────┘
         │ FrameAnalysis + triggered alerts
         ▼
┌─────────────────────┐
│  LangChain Agent    │  gpt-4o-mini via OpenAI API
│  (security_agent)   │  ReAct loop with tools:
│                     │  • log_event
│                     │  • trigger_alert
│                     │  • query_history
│  Memory: k=10 window│
└────────┬────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐  ┌──────────────┐
│ SQLite │  │  ChromaDB    │
│ event  │  │  frame index │
│  log   │  │  (embeddings)│
└────────┘  └──────────────┘
```

---

## 2. Component Breakdown

### 2.1 Background Subtractor (`vlm/background_subtractor.py`)

- **Class:** `BackgroundSubtractor`
- **Model:** `cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)`
- **Warm-up:** First 30 frames fed in learning-only mode (no analysis)
- **Per frame:**
  - Apply subtractor → foreground mask
  - Morphological ops (dilate) to fill gaps in mask
  - `findContours` → filter by min area (500px²)
  - Return list of `(x, y, w, h)` bounding boxes + crops
  - If no contours → return empty list (caller skips BLIP)

### 2.2 VLM Analyzer (`vlm/blip_analyzer.py`)

- **Class:** `BLIPAnalyzer`
- **Model:** `Salesforce/blip-image-captioning-base` via HuggingFace Transformers
- **Input:** Single image crop (PIL Image)
- **Output:** `CropAnalysis` dataclass
  ```python
  @dataclass
  class CropAnalysis:
      caption: str        # raw BLIP caption
      object_type: str    # "vehicle" | "person" | "unknown"
      color: str          # extracted color keyword or ""
  ```
- **Merge:** All crop analyses for a frame merged into one `FrameAnalysis`

### 2.3 FrameAnalysis Struct (`data/simulated_frames.py`)

```python
@dataclass
class FrameAnalysis:
    frame_id: int
    timestamp: str          # ISO 8601
    location: str           # main_gate | garage | perimeter
    raw_description: str    # joined BLIP captions
    objects: list[str]      # ["vehicle", "person"]
    activity: str           # entering | exiting | loitering | stationary | clear
    threat_level: str       # high | medium | low | none
    telemetry: dict
```

### 2.4 Alert Rules Engine (`agent/alert_rules.py`)

Deterministic rule evaluation — runs before the LLM agent to catch obvious cases cheaply.

| Rule | Logic |
|------|-------|
| RULE-01 | `"person" in objects AND 22 <= hour OR hour < 6` |
| RULE-02 | `"vehicle" in objects AND vehicle not in known_vehicles` |
| RULE-03 | `vehicle_entry_count[vehicle_id] > 2` |
| RULE-04 | `activity == "loitering" AND location dwell_time > 300s` |
| RULE-05 | `location == "perimeter" AND (22 <= hour OR hour < 6)` |

Returns list of `Alert` objects. Agent can add more via `trigger_alert` tool.

### 2.5 LangChain Agent (`agent/security_agent.py`)

- **LLM:** `ChatOpenAI(model="gpt-4o-mini")` via `langchain-openai` — fast, cost-effective, strong instruction-following
- **Agent type:** ReAct (`create_react_agent`)
- **Memory:** `ConversationBufferWindowMemory(k=10)` — last 10 frame summaries
- **System prompt:** Instructs the agent to act as a security analyst, use tools to log and alert, and reason about cross-frame patterns
- **Tools:** defined in `agent/tools.py`

### 2.6 Agent Tools (`agent/tools.py`)

| Tool | Signature | Side Effect |
|------|-----------|-------------|
| `log_event` | `(frame_id, message, severity)` | Writes to SQLite event log |
| `trigger_alert` | `(frame_id, rule_id, message, severity)` | Writes alert to SQLite + prints |
| `query_history` | `(query: str)` | Queries ChromaDB, returns top-3 past frames |

### 2.7 Event Store (`storage/event_store.py`)

- **DB:** SQLite at `data/events.db`
- **Tables:**
  - `events(event_id, frame_id, timestamp, type, message, severity)`
  - `alerts(alert_id, event_id, rule_id, severity, acknowledged)`
- **Interface:** `EventStore.log(...)`, `EventStore.get_alerts(...)`, `EventStore.get_by_timerange(...)`

### 2.8 Frame Index (`storage/frame_index.py`)

- **DB:** ChromaDB persistent at `data/chroma/`
- **Collection:** `frames`
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Document:** `raw_description` (the BLIP caption string)
- **Metadata:** `{frame_id, timestamp, location, objects, threat_level}`
- **Queries:** semantic similarity search + metadata filter

### 2.9 Query Engine (`query/query_engine.py`)

- Accepts natural language query string
- Routes to ChromaDB semantic search (`query_texts`)
- Applies metadata filters where extractable (time range, location, object type)
- Returns formatted result list

---

## 3. Data Flow for a Single Frame

```
Frame dict (from simulated_frames.py)
  │
  ├─ Attach telemetry (from telemetry.py)
  │
  ├─ BackgroundSubtractor.apply(image) → crops[]
  │     └─ empty? → log "clear", skip BLIP
  │
  ├─ BLIPAnalyzer.analyze(crop) × N → CropAnalysis[]
  │     └─ merge → FrameAnalysis
  │
  ├─ AlertRulesEngine.evaluate(FrameAnalysis) → Alert[]
  │
  ├─ SecurityAgent.process(FrameAnalysis, alerts) → agent response
  │     └─ calls tools: log_event, trigger_alert, query_history
  │
  └─ FrameIndex.add(FrameAnalysis)
```

---

## 4. Storage Layout

```
data/
├── events.db          # SQLite
├── chroma/            # ChromaDB persistent storage
└── sample_images/     # Generated scene images
```

---

## 5. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Deterministic rules before LLM | Reliable, fast, no hallucination risk for clear rule violations |
| BLIP on crops, not full frame | Focused captions; avoids describing static background |
| MOG2 warm-up on empty frames | Prevents first-frame false positives |
| OpenAI gpt-4o-mini | Fast, low cost per token, excellent instruction-following; requires `OPENAI_API_KEY` in `.env` |
| ChromaDB for frame index | Semantic search without a heavy infra setup |
| SQLite for event log | Simple, file-based, queryable with standard SQL |
| ConversationBufferWindowMemory k=10 | Enough for cross-frame context (e.g., "truck entered before") without unbounded memory |

---

## 6. Inter-Module Dependencies

```
main.py
 ├── data/simulated_frames.py
 ├── data/telemetry.py
 ├── vlm/background_subtractor.py
 ├── vlm/blip_analyzer.py
 ├── agent/alert_rules.py
 ├── agent/security_agent.py
 │    └── agent/tools.py
 │         ├── storage/event_store.py
 │         └── storage/frame_index.py
 └── query/query_engine.py
      └── storage/frame_index.py
```
