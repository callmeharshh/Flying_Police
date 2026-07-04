# Build Plan
## Flying Police

---

## Overview

Phased build — each phase produces a runnable, testable slice. Later phases layer on top.

| Phase | Focus | Deliverable |
|-------|-------|-------------|
| 1 | Project scaffold + simulated data | Runnable data pipeline skeleton |
| 2 | Storage layer | SQLite + ChromaDB working |
| 3 | VLM pipeline | BLIP analyzing crops |
| 4 | Alert rules engine | Deterministic alerts firing |
| 5 | LangChain agent | Agent using tools, cross-frame memory |
| 6 | Query interface | Natural language queries over frame index |
| 7 | Tests + cleanup | All test cases passing |
| 8 | README + docs | Submission-ready |

---

## Phase 1 — Project Scaffold + Simulated Data

**Goal:** Get the folder structure in place and simulated frame/telemetry data producing output.

### Tasks
- [ ] Create directory structure per PRD Section 11
- [ ] `config.py` — constants (paths, model names, alert thresholds, location names)
- [ ] `data/simulated_frames.py` — 7 scenario frames as Python dicts matching `FrameAnalysis` shape
- [ ] `data/telemetry.py` — generates telemetry dict for a given timestamp
- [ ] `data/sample_images/` — generate 3–5 simple scene images with OpenCV (colored rectangles representing vehicles/people; no real photos needed for scaffold)
- [ ] `main.py` skeleton — loop over frames, print frame dict

**Done when:** `python main.py` prints all 7 simulated frame dicts with telemetry attached.

---

## Phase 2 — Storage Layer

**Goal:** SQLite event log and ChromaDB frame index working independently.

### Tasks
- [ ] `storage/event_store.py`
  - `EventStore` class with SQLite connection
  - `create_tables()` on init
  - `log_event(frame_id, message, severity, type)` → inserts row
  - `get_alerts()` → returns all alerts
  - `get_by_timerange(start, end)` → filtered query
- [ ] `storage/frame_index.py`
  - `FrameIndex` class with ChromaDB client
  - `add_frame(frame_analysis)` → embeds description, stores with metadata
  - `query(text, filters)` → semantic search, returns top-k results
- [ ] `tests/test_indexing.py` — TC-04, TC-05

**Done when:** Can add a frame to ChromaDB and query it back; can log an event to SQLite and retrieve it.

---

## Phase 3 — VLM Pipeline

**Goal:** BLIP analyzing image crops and returning structured output.

### Tasks
- [ ] `vlm/background_subtractor.py`
  - `BackgroundSubtractor` class wrapping MOG2
  - `warm_up(frames)` — feed N blank/static frames
  - `apply(image)` → returns list of `(crop, bbox)` tuples; empty list if no foreground
- [ ] `vlm/blip_analyzer.py`
  - `BLIPAnalyzer` class, loads model once on init
  - `analyze(crop_image)` → returns `CropAnalysis(caption, object_type, color)`
  - `merge_crops(crop_analyses, frame_meta)` → returns `FrameAnalysis`
- [ ] `data/extract_frames.py` — utility to load sample images from disk for testing
- [ ] Wire into `main.py`: for each frame, run subtractor → BLIP → FrameAnalysis

**Done when:** BLIP produces a meaningful caption for at least one sample image crop.

---

## Phase 4 — Alert Rules Engine

**Goal:** Deterministic rules firing correctly for all 5 rule conditions.

### Tasks
- [ ] `agent/alert_rules.py`
  - `AlertRulesEngine` class
  - `evaluate(frame_analysis)` → returns `list[Alert]`
  - Implement RULE-01 through RULE-05
  - Stateful: maintains `vehicle_entry_counts` and `location_dwell_times` across frames
- [ ] `Alert` dataclass: `(rule_id, frame_id, message, severity)`
- [ ] `tests/test_alerts.py` — TC-02, TC-03, TC-06

**Done when:** S-03 triggers RULE-03, S-04 triggers RULE-01, S-06 triggers RULE-02, S-07 produces no alert.

---

## Phase 5 — LangChain Agent

**Goal:** Agent processes each frame, uses tools, maintains cross-frame memory.

### Tasks
- [ ] `agent/tools.py`
  - `log_event` tool wrapping `EventStore.log_event`
  - `trigger_alert` tool wrapping `EventStore.log_event` with type=alert
  - `query_history` tool wrapping `FrameIndex.query`
- [ ] `agent/security_agent.py`
  - `SecurityAgent` class
  - Init: `ChatOpenAI(model="gpt-4o-mini")`, ReAct agent, `ConversationBufferWindowMemory(k=10)`
  - System prompt: security analyst persona, instructs tool use
  - `process(frame_analysis, pre_alerts)` → runs agent, returns response string
- [ ] Wire into `main.py`: agent processes each FrameAnalysis after rules engine
- [ ] `tests/test_agent.py` — TC-01, TC-07

**Done when:** Agent logs S-01 correctly and recognizes the truck has entered before when processing S-03.

---

## Phase 6 — Query Interface

**Goal:** Natural language queries returning relevant frame results.

### Tasks
- [ ] `query/query_engine.py`
  - `QueryEngine` class wrapping `FrameIndex`
  - `query(text)` → formats and returns results
  - Handle: time queries, object queries, location queries
- [ ] Add interactive query loop at end of `main.py` (after processing all frames)
  - Print prompt `> `, accept input, print results
  - Exit on `quit`

**Done when:** Queries "show all truck events", "what happened at midnight?", "activity at main gate" return relevant frames.

---

## Phase 7 — Tests + Cleanup

**Goal:** All 7 test cases passing, code cleaned up.

### Tasks
- [ ] Complete `tests/test_agent.py` — TC-01, TC-07
- [ ] Complete `tests/test_alerts.py` — TC-02, TC-03, TC-06
- [ ] Complete `tests/test_indexing.py` — TC-04, TC-05
- [ ] Add TC-06 (no false alert during day — person at gate 10:00)
- [ ] Review and remove debug prints from all modules
- [ ] Confirm all 7 scenarios produce expected log/alert output
- [ ] Add session summary output at end of `main.py` (Bonus 13.1)

**Done when:** `pytest tests/` passes all 7 test cases.

---

## Phase 8 — README + Documentation

**Goal:** Submission-ready repo.

### Tasks
- [ ] `README.md`
  - Setup instructions (Python env, pip install, Ollama install + pull gemma2)
  - Run instructions (`python main.py`, query examples)
  - Architecture summary with diagram reference
  - Design decisions (why MOG2, why BLIP, why gemma2, why ChromaDB)
  - AI tools used and how they helped
- [ ] Flowchart / architecture diagram (export from ARCHITECTURE.md or draw separately)
- [ ] `requirements.txt` with pinned versions

**Done when:** A fresh clone can run `pip install -r requirements.txt` + `python main.py` and see all scenario outputs.

---

## Dependencies Between Phases

```
Phase 1 (scaffold)
    │
    ├──► Phase 2 (storage)
    │         │
    │         └──► Phase 5 (agent) ──► Phase 7 (tests)
    │                                        │
    ├──► Phase 3 (VLM)                       └──► Phase 8 (docs)
    │         │
    │         └──► Phase 4 (rules) ──► Phase 5
    │
    └──► Phase 6 (query) ── needs Phase 2
```

Phases 2, 3, and 6 can be built in parallel once Phase 1 is done.

---

## Environment Setup (for reference)

```bash
# Python env
python -m venv venv && source venv/bin/activate

# Step 1 — core packages
pip install -r requirements.txt

# Step 2 — LangChain (separate due to numpy constraint; see requirements-langchain.txt)
pip install --no-deps -r requirements-langchain.txt

# Add OpenAI key to .env
echo "OPENAI_API_KEY=sk-..." >> .env

# Run
python main.py
```

---

## Risk & Mitigation

| Risk | Mitigation |
|------|------------|
| OpenAI API latency / rate limits | gpt-4o-mini is fast; if rate-limited, add a small sleep between frames or batch fewer tool calls per turn |
| BLIP poor on generated images | Generated images use clear colored shapes — BLIP should caption "a blue rectangle" at minimum; acceptable for prototype |
| ChromaDB first-run slow | Embedding model downloads once on first run; cached after |
| MOG2 warm-up frames not available | Use first 5 simulated frames as warm-up (no image needed, just call apply() with blank numpy array) |
