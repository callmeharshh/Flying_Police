# Flying Police

Flying Police is a video surveillance analysis system that processes footage frame-by-frame, identifies objects and activities using a vision-language model, fires deterministic security alerts, and delivers real-time annotated notifications via Telegram.

**Repository:** [github.com/callmeharshh/Flying-Police](https://github.com/callmeharshh/Flying-Police)

---

## Demo

**[Watch the full demo on Loom](https://www.loom.com/share/edf1db4b1fe8478a9639fade8c22fe8e)** — includes video processing, live alert firing, Telegram notification, and the Q&A query interface.

**[Demo trial videos (Google Drive)](https://drive.google.com/drive/u/0/folders/1Xj53ns2BOVryUWKgidTWby3QNLLKiPzH)** — raw and processed footage used during development and testing.

Landing page : regal-cajeta-336b67.netlify.app
---

## Sample Output

Real output from running `python3 main.py` on `outside_entry_720p.mp4`.

---

### BLIP Frame Descriptions

Red bounding boxes and captions are drawn by the pipeline on each detected crop.

### outside_entry_720p.mp4 — Outdoor entry area (daytime)

| Frame | Object | BLIP Caption | Alert |
|---|---|---|---|
| 30 | person | "a man is seen in this surveillance image" | RULE-04 |
| 32 | person | "a man is walking down the sidewalk with a camera" | — |

### Additional Security Scenarios

**Unknown person spotted in warehouse**

![unknown person spotted in warehouse](data/sample_images/readme_frames/unknown-person-warehouse.png)

**Unknown person spotted near oil tanker**

![unknown person spotted near oil tanker](data/sample_images/readme_frames/unknown-person-oil-tanker.png)

---

### LangChain ReAct Agent — Thought / Action / Observation traces

Real agent output from `outside_entry_720p.mp4`. The agent reasons over BLIP captions, queries track history, and logs events.

**Frame 22 — new person identified via track position query:**

```
Frame 022 → 1 object detected
  Objects  : person
  Desc     : a man is seen in a surveillance photo
  Activity : entering | Threat: high
  Track    : track_002 (new object #2)

Action: query_track_positions("main_gate")
  → Frame 21: center=(163.5, 279.0), track=track_001
  → Frame 20: center=(150.5, 274.5), track=track_001
  → Frame 19: center=(149.5, 273.5), track=track_001
  → Frame 18: center=(149.0, 264.0), track=track_001

Thought: Current bbox center is far from track_001's last position — this is a new individual.
Action:  log_event("A new person was detected entering the main gate with no prior positions recorded.")
→ Logged: evt_a4f2

Final Answer: A new person was detected entering the main gate with no prior positions recorded.
```

**Frame 33 — stationary person (loitering detection):**

```
Frame 033 → 1 object detected
  Objects  : person
  Desc     : a man is sitting on the floor in front of a door
  Activity : stationary | Threat: high
  Track    : track_003 (new object #3)

Action: query_track_positions("main_gate")
  → No prior positions at this location.

Action: log_event("A person is sitting on the floor in front of a door at the main gate.")
→ Logged

Final Answer: A person is sitting on the floor in front of a door at the main gate,
             identified as a new entry with no prior positions recorded.
```

**Frame 28 — cross-track position reasoning:**

```
Frame 028 → 1 object detected
  Desc  : a man is seen in a surveillance photo
  Track : track_002 (continuing)

Action: query_track_positions("main_gate")
  → Frame 27: center=(242.5, 296.0), track=track_001
  → Frame 26: center=(267.5, 268.0), track=track_002
  → Frame 25: center=(260.5, 293.5), track=track_002
  → Frame 24: center=(257.0, 302.0), track=track_002

Thought: Current center (236.0, 253.0) is consistent with track_002 trajectory.
         Movement direction and distance confirm same individual.
Action:  log_event("A person is continuing to enter the main gate, same track from previous frames.")
Final Answer: A person is continuing to enter the main gate, identified as the same track.
```

---

### Event Log (SQLite) + ChromaDB Semantic Search

```
EVENT LOG (48 events)
  [HIGH] Frame 14 | a man is seen in this surveillance image [main_gate]
  [HIGH] Frame 23 | a man walking down a street with a black umbrella [main_gate]
  [HIGH] Frame 33 | a man is sitting on the floor in front of a door [main_gate]
  [HIGH] Frame 34 | a man in a hat is sitting on the floor [main_gate]
  [HIGH] Frame 36 | a man wearing a hat [main_gate]
  ...48 total events across 24 detected frames

FRAME INDEX — ChromaDB semantic queries
  Query: "person walking"
    [score=0.49]  Frame 26 | a man is walking down a sidewalk with a black umbrella
    [score=0.41]  Frame 23 | a man walking down a street with a black umbrella

  Query: "vehicle entering"
    [score=0.36]  Frame 18 | surveillance photo of a man who was caught in a car
    [score=0.27]  Frame 25 | surveillance photo of a man in a parking lot

  Query: "person at door"
    [score=0.61]  Frame 33 | a man is sitting on the floor in front of a door
    [score=0.30]  Frame 40 | a small window with a small window on the side

NATURAL LANGUAGE QUERY — "How many men were detected?"
  Frame 19 | the suspect is seen in this surveillance image       (score=0.297)
  Frame 14 | a man is seen in this surveillance image             (score=0.292)
  Frame 21 | a man is seen in this surveillance image             (score=0.292)
  Frame 30 | a man is seen in this surveillance image             (score=0.292)
  Frame 31 | a man is seen in this surveillance image             (score=0.292)

SESSION COMPLETE — 24 frames analyzed, 84 raw frames, 7.0 fps
```

---

## What it does

1. Reads a video file and samples every other frame
2. Uses OpenCV MOG2 background subtraction to find **moving** foreground blobs
3. **Skips frames with no motion** — BLIP and alert rules run only when something moves (object detection is not required on every frame)
4. Passes motion crops to **BLIP** (Salesforce vision-language model) for plain-English descriptions
5. Infers activity (`entering` / `loitering` / `stationary`) from the caption
6. Evaluates five deterministic alert rules (RULE-01 to RULE-05)
7. Runs a **LangChain ReAct agent** (GPT-4o-mini) on frames with detections for context-aware logging
8. Stores events in **SQLite** and indexes **frames with detections** in **ChromaDB**
9. Sends HIGH-severity alerts with an annotated frame photo to **Telegram**
10. Exposes a natural-language query interface over the session

---

## System architecture

### Full system

End-to-end flow from video input through MOG2 motion detection, BLIP captioning, alert rules, agent logging, storage, and Telegram. See [CODE_DOCS.md](CODE_DOCS.md) for module-level detail.

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

---

## Requirements

- Python 3.11+ (developed and tested on **3.14**)
- macOS or Linux (Windows untested)
- **CLI:** OpenAI API key required (`gpt-4o-mini` agent runs on every detection)
- **Web UI:** OpenAI key optional — disable **LangChain agent** in the sidebar to process with BLIP + rules only; Q&A works without OpenAI
- ~4 GB free disk space for the BLIP model cache (downloaded from HuggingFace on first run)

---

## Project structure

```
.
├── agent/
│   ├── alert_rules.py        # Deterministic rules engine (RULE-01 to RULE-05)
│   ├── security_agent.py     # LangChain ReAct agent setup
│   ├── tools.py              # LangChain @tool functions (log_event, trigger_alert, query_history)
│   └── vehicle_tracker.py    # Spatial motion tracker for cross-frame object continuity
├── config.py                 # All constants and env-var loading
├── data/
│   └── sample_video/         # Place input videos here
├── main.py                   # CLI entry point
├── notifications/
│   └── telegram_notifier.py  # Telegram Bot API integration
├── pipeline/
│   └── video_processor.py    # Frame-by-frame pipeline, yields PipelineEvent stream
├── query/
│   └── query_engine.py       # NL query routing (SQLite + ChromaDB)
├── requirements.txt
├── requirements-langchain.txt  # LangChain stack (install with --no-deps; see Setup)
├── storage/
│   ├── event_store.py        # SQLite wrapper (events + alerts tables)
│   └── frame_index.py        # ChromaDB wrapper (vector index of BLIP captions)
├── tests/
│   ├── unit/                 # Fast unit tests (no model or API calls)
│   └── integration/          # BLIP + integration QA tests (optional)
├── scripts/
│   ├── run_ui.sh                 # Start Streamlit web UI
│   └── download_sample_videos.sh # Fetch C-MOR + YouTube test clips (macOS)
├── ui/
│   └── app.py                # Streamlit web interface
└── vlm/
    ├── background_subtractor.py  # MOG2 wrapper with contour filtering
    └── blip_analyzer.py          # BLIP captioning + keyword extraction
```

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/callmeharshh/Flying-Police.git
cd Flying-Police
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

Dependencies are split into two files because `langchain==0.3.0` metadata declares `numpy<2.0.0` while `opencv-python>=4.9` requires `numpy>=2`. Both work fine together at runtime — the constraint in langchain's metadata is stale. Installing in two steps avoids the conflict:

```bash
# Step 1 — core packages (opencv, torch, BLIP, ChromaDB, Streamlit)
pip install -r requirements.txt

# Step 2 — LangChain packages (installed without dependency resolution to bypass stale numpy constraint)
pip install --no-deps -r requirements-langchain.txt
```

Key packages installed:


| Package               | Version   |
| --------------------- | --------- |
| opencv-python         | 4.13.0.92 |
| torch                 | 2.12.0    |
| transformers          | 5.10.2    |
| chromadb              | 1.5.9     |
| sentence-transformers | 5.5.1     |
| langchain             | 0.3.0     |
| langchain-openai      | 0.2.0     |
| streamlit             | ≥ 1.33.0  |
| numpy                 | 2.4.6     |


> The BLIP model (`Salesforce/blip-image-captioning-base`, ~~3.7 GB) is downloaded from HuggingFace on first run and cached at `~~/.cache/huggingface/`.

### 3. Create a `.env` file

```bash
cp .env.example .env   # or create it manually
```

Minimum required:

```
OPENAI_API_KEY=sk-...
```

Optional (Telegram alerts):

```
TELEGRAM_BOT_TOKEN=<token from @BotFather>
TELEGRAM_CHAT_ID=<your chat ID from @userinfobot>
```

Leave `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` blank to disable notifications — the pipeline runs fully without them.

### 4. Download test videos (macOS)

Sample videos are **not in git** (`data/sample_video/` is gitignored). Use the clips below — the same ones used for development and integration tests.

#### Test videos


| File                                                  | Source                                                                                                                                               | Used for                                                |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| `entrance_area_720p.mp4`                              | [C-MOR — Motion detection, entrance area](https://www.c-mor.com/video-surveillance-demo/sample-recordings-of-the-video-surveillance-system-c-mor)    | CLI default (`DEMO_VIDEO`); parking lot motion          |
| `outside_entry_720p.mp4`                              | [C-MOR — Motion detection, outside entrance](https://www.c-mor.com/video-surveillance-demo/sample-recordings-of-the-video-surveillance-system-c-mor) | Person + vehicle; query / track tests                   |
| `night_surveillance_Am8tq-0FQJU.mp4`                  | [YouTube — Security camera at the gate](https://www.youtube.com/watch?v=Am8tq-0FQJU)                                                                 | Night lighting + RULE-01 (person at night)              |
| `Qubo Outdoor Security Camera - Person Detection.mp4` | Local / optional (no fixed public URL)                                                                                                               | Person-detection demo clip; add manually if you have it |


#### One-command download (recommended)

From the project root, with Homebrew available:

```bash
# yt-dlp is only required for the YouTube night clip
brew install yt-dlp

bash scripts/download_sample_videos.sh
```

The script downloads the two C-MOR clips via `curl` and the night clip via `yt-dlp` into `data/sample_video/`.

#### Manual download (macOS)

```bash
mkdir -p data/sample_video
cd data/sample_video

# C-MOR — entrance area (CLI default)
curl -L -o entrance_area_720p.mp4 \
  "https://www.c-mor.com/video-surveillance-demo/sample-recordings-of-the-video-surveillance-system-c-mor?download=27:motion-detection-entrance-area"

# C-MOR — outside entry (person / vehicle)
curl -L -o outside_entry_720p.mp4 \
  "https://www.c-mor.com/video-surveillance-demo/sample-recordings-of-the-video-surveillance-system-c-mor?download=29:motion-detection-outside-entrance"

# YouTube — night gate footage (needs yt-dlp: brew install yt-dlp)
yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best" \
  --merge-output-format mp4 \
  -o "night_surveillance_Am8tq-0FQJU.mp4" \
  "https://www.youtube.com/watch?v=Am8tq-0FQJU"

cd ../..
```

The CLI default is set in `config.py`:

```python
DEMO_VIDEO = "entrance_area_720p.mp4"
```

Change `DEMO_VIDEO` to match another downloaded file, or use the **Web UI** to upload any video without editing config.

### 5. Streamlit web UI (optional)

Streamlit is included in `requirements.txt` (step 2 above). No extra install is needed if you already ran both pip commands.

Quick check:

```bash
source venv/bin/activate
streamlit --version
```

You should see `Streamlit, version 1.x.x` or newer.

The UI entry point is `ui/app.py`. Session data (uploads, SQLite, ChromaDB) is written under `data/sessions/` and `data/uploads/` — created automatically on first run.

---

## Running

### CLI

Processes `DEMO_VIDEO` from `data/sample_video/`, prints frame-by-frame output, then opens an interactive query prompt.

**Requires:** video file present (see Setup §4) and `OPENAI_API_KEY` in `.env` (agent is always enabled in CLI).

```bash
source venv/bin/activate
python3 main.py
```

Each run clears the previous session's SQLite database and ChromaDB index before starting. BLIP loads on first run (~3.7 GB download); runs on **CPU** by default (CUDA if available).

### Web UI (Streamlit)

Upload any MP4 in the browser, watch live frame analysis, view alerts, and query the session.

#### Start the app

From the project root, with the virtual environment active:

```bash
source venv/bin/activate
bash scripts/run_ui.sh
```

Or run Streamlit directly:

```bash
source venv/bin/activate
streamlit run ui/app.py
```

The app opens at **[http://localhost:8501](http://localhost:8501)** (Streamlit may open it automatically).

To use a different port:

```bash
streamlit run ui/app.py --server.port 8502
```

#### Use the UI

1. **Settings** (sidebar): choose location (`main_gate`, `garage`, `perimeter`). Enable **LangChain agent** if `OPENAI_API_KEY` is set in `.env`.
2. **Upload video** (main page): drag and drop or browse for MP4 / WebM / MOV / AVI. You can also upload a file from `data/sample_video/` after placing it there locally.
3. Click **Process video**. Live analysis shows the current frame, per-frame logs, and alerts as they fire.
4. When processing finishes, use the **chat input** at the bottom to query the session (e.g. `how many men were detected?`, `any alerts?`, `person walking`).

> **Requirements:** BLIP runs locally (first run downloads the model). **LangChain agent** needs `OPENAI_API_KEY` — uncheck it in the sidebar to skip OpenAI during processing. **Q&A** uses SQLite + ChromaDB and does not call OpenAI. Telegram is optional (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`).

#### Stop the app

Press `Ctrl+C` in the terminal where Streamlit is running.

---

## Configuration reference

All values live in `config.py`. Only Telegram settings are read from environment variables (via `.env`).


| Constant                          | Default                  | Description                                                 |
| --------------------------------- | ------------------------ | ----------------------------------------------------------- |
| `DEMO_VIDEO`                      | `entrance_area_720p.mp4` | Video file used by `main.py`                                |
| `SAMPLE_EVERY_N_FRAMES`           | `2`                      | Process every Nth raw frame                                 |
| `MOG2_WARMUP_FRAMES`              | `5`                      | Frames used to calibrate background model                   |
| `MOG2_VAR_THRESHOLD`              | `120`                    | MOG2 sensitivity (higher = less sensitive)                  |
| `MIN_CONTOUR_AREA`                | `2000`                   | Minimum foreground blob size in px²                         |
| `MAX_CONTOUR_AREA_RATIO`          | `0.40`                   | Blobs covering >40% of frame are rejected                   |
| `LOITER_THRESHOLD_SECONDS`        | `15`                     | RULE-04 prolonged dwell threshold (use 300 for production)  |
| `REPEAT_ENTRY_LIMIT`              | `1`                      | RULE-03 fires when entry count exceeds this                 |
| `LIGHTING_NIGHT_MEAN_THRESHOLD`   | `90`                     | Mean frame brightness below this → night (RULE-01, RULE-05) |
| `LIGHTING_NIGHT_DARK_PIXEL_RATIO` | `0.55`                   | Dark-pixel fraction that implies night                      |
| `OPENAI_MODEL`                    | `gpt-4o-mini`            | LLM used by the LangChain agent                             |
| `AGENT_MEMORY_K`                  | `10`                     | Sliding window memory size for the agent                    |


---

## Alert rules


| Rule    | Condition                                                                                     | Severity |
| ------- | --------------------------------------------------------------------------------------------- | -------- |
| RULE-01 | Person detected at night (scene lighting)                                                     | HIGH     |
| RULE-02 | Unrecognized vehicle                                                                          | MEDIUM   |
| RULE-03 | Vehicle entry count exceeds `REPEAT_ENTRY_LIMIT`                                              | MEDIUM   |
| RULE-04 | Person loitering or stationary (fires on first sight; again after `LOITER_THRESHOLD_SECONDS`) | HIGH     |
| RULE-05 | Any activity at `perimeter` location at night                                                 | HIGH     |


HIGH alerts trigger a Telegram notification. MEDIUM alerts are stored in SQLite only.

---

## Running tests

Fast unit tests (no OpenAI key, no BLIP model load):

```bash
source venv/bin/activate
bash scripts/run_test_suites.sh
```

Or unit tests only:

```bash
python3 -m pytest tests/unit/ -v
```

Expected: **64 passed**.

Optional integration tests (loads BLIP, ~15s):

```bash
python3 -m pytest tests/integration -m integration -v
```

---

## Telegram setup (optional)

1. Message `@BotFather` on Telegram → `/newbot` → copy the token
2. Message `@userinfobot` → copy your numeric user ID
3. Add both to `.env`:

```
TELEGRAM_BOT_TOKEN=123456789:ABC-...
TELEGRAM_CHAT_ID=987654321
```

On the next run, every HIGH-severity alert will send an annotated JPEG of the triggering frame to your Telegram account.

---

## Design decisions

**Why BLIP over other VLMs?**
BLIP runs entirely locally with no API call, produces consistent short captions suitable for keyword extraction, and handles the low-resolution crops produced by MOG2 reliably. GPT-4V would be more accurate but adds latency and cost per frame.

**Why deterministic rules before the LLM?**
Alert logic must not hallucinate. The rules engine fires first and its output is final — the LangChain agent receives already-fired alerts as context and adds narrative, but cannot suppress or override a rule decision.

**Why two databases?**
SQLite handles structured queries (time ranges, severity filters, exact alert lookups) efficiently. ChromaDB handles semantic queries ("show me frames where someone was near the door") using vector similarity — something SQL `LIKE` cannot do meaningfully. The Query Engine routes each question to the appropriate backend automatically.

**Why LangChain 0.3 specifically?**
LangChain 1.x completely rewrote the `create_react_agent` API in a breaking way. Version 0.3.0 is the last stable release with the interface used here.

---

## How I used AI

I used two AI tools simultaneously throughout this project — **Cursor** for in-file implementation and **Codex** for system-level work — and kept them deliberately separate.

### The split

**Codex** handled everything that spans multiple files or requires understanding the whole system:
- Designed the full pipeline (MOG2 → BLIP → rules → agent) before any code was written
- Diagnosed the `numpy<2.0 / opencv>=4.9` conflict and proposed the two-step install workaround
- Caught a `load_dotenv()` ordering bug where Streamlit imported `config.py` before env vars were loaded
- Patched `SafeStdOutCallbackHandler` — `langchain-core 0.3` started passing `serialized=None`, which crashed the default handler
- Proposed brightness-based lighting detection (HSV V-channel mean + dark-pixel ratio) instead of wall-clock time-of-day
- Generated all documentation: PRD, architecture diagram, build plan, README, CODE_DOCS

**Cursor** handled self-contained, in-file work:
- Function bodies once the interface was already designed in Codex
- SQLite query methods, LangChain `@tool` stubs, unit test boilerplate
- In-file fixes during debugging sessions

The one place Cursor fell short: it would sometimes bypass `config.py` and write `os.getenv()` directly, or use a local constant instead of the shared one. Codex caught these on integration because it has the full project in context.

### Dry-running edge cases before writing code

Before implementing any component, I walked through failure scenarios with Codex to find the constraints I'd otherwise discover mid-debug:

- *Same person re-enters* → revealed the need for per-track deduplication, not just per-frame counting
- *MOG2 fragments one object into three blobs* → minimum area filter (2000 px²) + union-find dedup on count queries
- *Vehicle shadow larger than the vehicle* → `detectShadows=True` + threshold at 200 (instead of 127) in MOG2
- *No warmup frames* → 5-frame silent learning period before any detection fires
- *LangChain `log_event` called twice per frame* → added explicit constraint in the agent system prompt

Running these as "what breaks if…" conversations surfaced design constraints before they became runtime bugs.

### PRD + architecture first

Before writing a single line of code, I used Codex to produce a PRD, a full architecture diagram, and a build plan — and only started implementation once those were stable. Telegram notifications were added at the PRD stage, not retrofitted later, because the threat model made real-time alerting a first-class requirement.

### Custom setup

- Installed the `code-documentation` skill from `skillcreatorai/Ai-Agent-Skills` into `.codex/skills/` so Codex could generate structured module-level docs on demand
- Pre-approved permissions in `.codex/settings.local.json` for `pip install *`, `yt-dlp`, and specific domains (arxiv, huggingface, moondream) to avoid repeated prompts during research
- Added repo-level Codex guidance in `AGENTS.md` and a reusable workflow guide in `CODEX_PRACTICES.md`
- Added Cursor project rules: all constants from `config.py`, type annotations required, no trivial comments
