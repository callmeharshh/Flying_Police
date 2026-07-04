# How I Built Flying Police

**By Jayesh Shete**  
**Repository:** [github.com/callmeharshh/Flying-Police](https://github.com/callmeharshh/Flying-Police)  
**Stack:** Python 3.14 · OpenCV · BLIP · LangChain 0.3 · GPT-4o-mini · ChromaDB · SQLite · Telegram · Streamlit

---

## 1. Where I Started — PRD, Architecture, Build Plan

Before writing any code, I produced three documents using Codex: a PRD, an architecture doc, and a build plan. That order mattered.

The **PRD** defined the functional requirements, hardware constraints (CPU/MPS, no GPU), and scope boundaries. Writing it upfront settled questions like the dual-database design before they became implementation conflicts.

The **Architecture doc** laid out the full pipeline, data flow, and technology choices before a single file was created. Every module had a defined interface to build toward.

The **Build Plan** was the implementation roadmap — install order, milestones, and a risk table. It turned out to be useful: the numpy/langchain conflict was in the risk table and the two-step install workaround was planned before it became a blocker.

The pipeline I sketched on day one stayed unchanged through to the final build:

```
Motion detection → Object description → Alert logic → Agent reasoning → Storage + Query
```

---

## 2. Picking the Right Vision Model

The VLM choice determined everything downstream. Requirements: runs locally on CPU/MPS, produces natural language the agent can reason about, fast enough for a real demo.


| Model     | Output               | Hardware          | CPU speed             | Verdict                                         |
| --------- | -------------------- | ----------------- | --------------------- | ----------------------------------------------- |
| CLIP      | Similarity scores    | Low               | Very fast             | Classifier, not captioner — wrong output format |
| YOLO      | Class labels + bbox  | Low               | Very fast             | Fixed 80-class vocabulary, no language output   |
| **BLIP**  | **Natural language** | **Low (CPU/MPS)** | **0.3–5s/crop**       | **Chosen**                                      |
| BLIP-2    | Rich NL + VQA        | GPU (14GB VRAM)   | Very slow             | Too heavy for CPU                               |
| LLaVA-7B  | Conversational NL    | GPU (14GB VRAM)   | 30–120s/image         | Would make LangChain agent redundant            |
| Moondream | Natural language     | Low               | 2–4× slower than BLIP | Verbose captions, worse keyword extraction      |


**CLIP** was out after day one — it returns `{person: 0.87}`, not sentences. **YOLO** can say "person" but not "person in a dark jacket loitering near the gate." **LLaVA** was the model I most wanted to use, but 30–120 seconds per frame on CPU isn't a pipeline. **BLIP base** hit the sweet spot — focused natural language captions on tight crops, runs on MPS, fast enough at 2–3 fps:

```
Crop: person at gate (128×200px)
BLIP: "a man in a dark jacket standing near a door"
```

That single sentence gives the rules engine and agent everything they need.

---

## 3. The Background Subtraction Insight

Running BLIP on every frame was the first thing I tried. The captions were useless — a person occupying 3% of a 1080p frame produces "a large empty parking lot." BLIP was trained on images where the subject fills the frame.

The fix came from thinking about how YOLO works: locate first, then describe. MOG2 (Mixture of Gaussians v2) learns the static background from warmup frames, then flags only moving foreground pixels. Only the resulting crops go to BLIP.

```
Warmup frames → background model
New frame → MOG2 mask → contours → bounding box crops → BLIP
                                     (1–3 per frame, not the full image)
```

Caption quality improved immediately. As a side effect, ~90% of frames are skipped entirely (empty scenes produce no contours), making the pipeline viable on CPU.

Key tuning: `varThreshold=120` — low enough to catch slow-moving objects, high enough to ignore cloud shadows. `detectShadows=True` with a 200-threshold strips car shadows that would otherwise generate phantom detections. A 5-frame warmup prevents the first frame (entirely "new" to an empty model) from triggering a full-frame detection.

---

## 4. Designing the Alert Rules

The goal was five rules that each detect a reliably observable pattern — not an exhaustive list, just the patterns a real security operator would prioritise.

**RULE-01 — Person after hours:** fires when a person crop is detected and the frame is classified as dark. Day/night comes from frame brightness (HSV V-channel), not wall-clock time — pre-recorded footage plays back at arbitrary times, so the clock is useless. Deduplicates per track ID so one alert fires per arrival, not per frame.

**RULE-02 — Unrecognised vehicle:** fires on any vehicle detection. No plate recognition in this prototype, so every vehicle is "unrecognised." The name signals the production extension: suppress for registered plates.

**RULE-03 — Repeat entry:** fires when the same track ID has entered the scene more times than a configured threshold. Requires the vehicle tracker — without cross-frame identity, this rule is impossible to implement correctly.

**RULE-04 — Loitering:** two-stage. Stage 1 (informational) fires at a low dwell threshold. Stage 2 (HIGH severity) fires when the dwell exceeds a higher threshold. The two stages give the agent context about severity rather than a binary flag.

**RULE-05 — Unattended object:** fires when a non-person crop appears near a location where a person track recently went inactive. The object appeared, the person left, the object remains. Approximate by nature — MOG2 adapts to stationary objects within 10–20 seconds, so the detection window is narrow.

Rules I left out: perimeter breach (needs per-camera polygon config), crowd formation (person counting unreliable at distance), vehicle speed (pixel-velocity too noisy). Five consistent rules beat eight inconsistent ones.

---

## 5. Why LangChain

The rules engine handles deterministic yes/no checks. What it can't do is connect observations across time — a vehicle that behaved normally at 9am but returned at midnight requires memory of the earlier event. LangChain's `ConversationBufferWindowMemory(k=10)` gives the agent a rolling window of the last 10 frames. The ReAct loop (Reason → Act → Observe) then lets it reason before calling tools:

```
Thought: RULE-01 fired. This person wasn't in the last 10 frames — new arrival.
Action:  log_event("Person at main gate after hours — new arrival at 23:45")
```

The LLM never decides whether to fire an alert. Rules fire first; the agent receives them as context and adds narrative. This keeps alerts deterministic and auditable. I stayed on LangChain 0.3 — the 1.x LCEL rewrite broke the `create_react_agent` + `AgentExecutor` interface with no functional benefit for this project.

---

## 6. Building With AI — Cursor and Codex

The project was built with two AI tools running in parallel, each used for what it does well.

**Cursor** handled in-file implementation: function bodies once interfaces were defined, SQLite query methods, LangChain tool stubs, unit test boilerplate. Fast and low-friction for anything self-contained within a single file. Its limitation: no model of the full project. It repeatedly generated `os.getenv()` calls directly in new files, unaware that `config.py` is the single source of truth for env vars.

**Codex** handled everything requiring a view of the whole system:

- Designed the pipeline architecture and module interfaces before any code was written
- Diagnosed the numpy/langchain conflict and designed the two-step install
- Caught the `load_dotenv()` ordering bug (Streamlit importing `config.py` before env vars were loaded — invisible without tracing the full import sequence)
- Patched the `SafeStdOutCallbackHandler` crash (`langchain-core 0.3` passes `serialized=None` to `on_chain_start`)
- Proposed brightness-based lighting detection instead of wall-clock time
- Generated all documentation

**The workflow:**

```
Codex → design interface + algorithm
Cursor      → implement function bodies
Codex → wire modules, debug integration
Cursor      → in-file fixes
Codex → documentation
```

**Dry-running edge cases before implementing** was one of the most valuable things I did. Before writing the rules engine, I walked through scenarios with Codex — not to get code, just to think through what would break:

- *Same person re-enters frame* → RULE-01 floods without per-track deduplication
- *MOG2 fragments one object into multiple contours* → same person logged twice, two track IDs assigned → need minimum area filter + union-find dedup
- *Vehicle shadow larger than vehicle* → shadow gets its own track ID and BLIP caption → `detectShadows=True` + 200-threshold
- *No warmup frames* → first frame is 100% foreground → full-frame detection, useless caption
- *LangChain calls `log_event` twice* → duplicate events → needed an explicit "call exactly once per frame" constraint in the prompt

Every scenario walked through before building was a bug that didn't have to be found in testing.

**Customisations:**

- Installed `code-documentation` skill from `skillcreatorai/Ai-Agent-Skills` into `.codex/skills/` — gave Codex a documentation framework (ADR format, progressive disclosure, examples-first) that produced the 2,568-line `CODE_DOCS.md`
- Pre-approved permissions in `.codex/settings.local.json`: `pip install` *, `yt-dlp`* , `WebFetch` for arxiv/huggingface/moondream, `Skill(deep-research)` — eliminated approval friction during debugging sessions
- Cursor project rules: all constants from `config.py`, type annotations on public signatures, no comments unless the WHY is non-obvious

---

## 7. Things I Had to Build from Scratch

**Lighting detector:** Replaced wall-clock day/night detection with HSV V-channel brightness analysis on a central 65% crop. Pre-recorded footage plays back at arbitrary times — the clock is meaningless. Two thresholds must agree (mean brightness + dark-pixel ratio) to handle partially-lit scenes.

**Vehicle tracker:** Replaced DeepSORT (GPU-dependent) with spatial proximity matching. Match distance scales with frame gap: `25 × gap × 1.5`. Tracks survive 45 frames of absence. Cheaper, CPU-friendly, correct enough for the use case.

**Union-find track dedup:** Naive track ID counting overcounts — a person who leaves and re-enters gets two IDs. Union-find merges segments that are temporally close, spatially overlapping, and within 250px, producing accurate individual counts.

**Telegram notifications:** Added for a real-world reason: a HIGH severity alert that sits in a database until someone checks the dashboard is not useful. A person loitering at a restricted gate at 1am needs to be seen immediately, on someone's phone, with proof. On every HIGH alert, Telegram sends the rule ID, BLIP caption, and the annotated frame (red bounding box, rule ID overlaid) to the configured chat. Implemented as a daemon thread per notification — a synchronous 200–800ms API call would stall the pipeline on every alert.

**SafeStdOutCallbackHandler:** `langchain-core 0.3` passes `serialized=None` to `on_chain_start` in some paths, crashing the standard callback handler with `TypeError`. Minimal subclass handles only this case; everything else propagates normally.

---

## 8. Key Decisions at a Glance


| Decision                       | Alternative              | Reason                                                           |
| ------------------------------ | ------------------------ | ---------------------------------------------------------------- |
| MOG2 before BLIP               | BLIP on every frame      | 90% frames empty; full-frame captions background-dominated       |
| BLIP base                      | BLIP-2 / LLaVA           | CPU constraint; base captions sufficient                         |
| Brightness-based day/night     | Wall-clock time          | Pre-recorded footage plays at arbitrary times                    |
| Deterministic rules first      | LLM decides alerts       | LLMs hallucinate; safety alerts must be deterministic            |
| Spatial proximity tracker      | DeepSORT                 | DeepSORT needs GPU                                               |
| SQLite + ChromaDB              | One database             | SQL for structured queries; ChromaDB for semantic search         |
| LangChain 0.3                  | LangChain 1.x            | 1.x broke the agent API with no functional gain here             |
| Telegram daemon threads        | Synchronous              | 200–800ms API call would block pipeline per alert                |
| `load_dotenv()` in `config.py` | In entry points          | Streamlit imports config before any entry point runs             |
| Two-step pip install           | Single requirements file | `langchain 0.3` declares `numpy<2`; only `--no-deps` resolves it |


---

## 9. Why the Architecture Looks the Way It Does

Every layer exists because the version without it produced a specific failure:

```
Video → MOG2             ← without this: BLIP runs on empty frames, captions are background
      → BLIP             ← without this: no natural language for the agent to reason about
      → LightingDetector ← without this: RULE-01 depends on when you run the demo, not the video
      → VehicleTracker   ← without this: RULE-03 impossible, alert deduplication broken
      → Rules engine     ← without this: LLM decides alerts — hallucination risk, not auditable
      → LangChain agent  ← without this: no cross-frame memory, no contextual narrative
      → SQLite+ChromaDB  ← without this: semantic queries impossible in SQL; counts inefficient in vectors
      → Telegram         ← without this: HIGH alerts sit unseen until someone checks the dashboard
```

Each component earned its place by solving a real, observable failure — not a hypothetical one.

---

## 10. Testing — Validating Against Real Footage

I didn't want to test against synthetic data. The whole point of the system was to work on real surveillance video, so that's what I tested against.

Four real videos were used as fixtures:

- `entrance_area_720p.mp4` — BLK-HDPTZ12 parking lot camera, the primary demo video
- `outside_entry_720p.mp4` — second outdoor scene, different lighting and angle
- `night_surveillance_Am8tq-0FQJU.mp4` — downloaded from YouTube, nighttime footage for RULE-01/05 coverage
- `Qubo Outdoor Security Camera - Person Detection.mp4` — YouTube, explicit person detection scenarios

To avoid re-running BLIP (slow) on every test run, I built a `ValidationCaptureRecorder` that runs the pipeline once against each video, captures the frame crops, BLIP captions, MOG2 masks, and alert outcomes into a `manifest.json` fixture. Subsequent test runs replay from the fixture — deterministic, fast, and no model inference needed.

The test suite is organised by marker: `@pytest.mark.unit` for pure logic (alert rule evaluation, tracker proximity matching, union-find dedup), `@pytest.mark.integration` for pipeline stages with real fixture data, and `@pytest.mark.slow` for full end-to-end runs. This meant fast feedback during development without skipping real validation.

The most useful tests were the ones that caught edge cases I'd dry-run but not yet proven: that the same person re-entering only fires RULE-01 once, that shadow contours don't generate their own events, that track dedup produces correct individual counts when tracks fragment.

---

## 11. Reference Videos — Why Real Footage Mattered

The reference videos weren't an afterthought — they were a development tool. I downloaded them specifically to expose failure modes that synthetic or simulated frames would never surface.

Simulated frames (hardcoded detection dictionaries) can verify that the rules engine logic is correct. They can't tell you that MOG2 at `varThreshold=50` fires on every passing cloud, that a car's shadow in afternoon light is larger than the car itself, or that BLIP produces "a large empty parking lot" when a person is in the far corner of a 1080p frame. Those required running against real video and watching what came out.

The YouTube videos were chosen specifically to cover scenarios the local camera footage didn't: `night_surveillance` for genuine low-light conditions, the Qubo video for close-range person detection at various distances. `yt-dlp` was the download tool — its permission was pre-approved in `.codex/settings.local.json` so downloads could happen inline without interrupting sessions.

---

## 12. What I'd Do Differently — and What Comes Next

**If I were building this for production:**

- **Replace BLIP with BLIP-2 or LLaVA on GPU hardware.** BLIP base was the right call for CPU/MPS, but on a real deployment with a dedicated GPU, BLIP-2's instruction-following ("describe any suspicious behaviour") would produce significantly richer agent context.
- **Replace the proximity tracker with DeepSORT or ByteTrack.** The spatial proximity model works but fails when two people cross paths. Appearance-based re-ID is the correct solution once GPU is available.
- **Add plate recognition to RULE-02.** Right now every vehicle is "unrecognised." A production system would suppress RULE-02 for registered plates and only alert on unknowns — the architecture already supports this as a suppression list.
- **Add per-camera polygon zones for perimeter breach detection.** The rules engine is designed to accept additional rules; the missing piece is a configuration UI for drawing zone boundaries per camera.
- **Move from SQLite to PostgreSQL.** SQLite works fine for a prototype but concurrent writes from multiple camera streams would need a proper database.

**What the prototype proved:** the layered architecture — MOG2 for localisation, BLIP for description, deterministic rules before LLM reasoning, dual-database storage — is sound. Each component's responsibility is narrow and replaceable. Upgrading BLIP to BLIP-2, or the proximity tracker to DeepSORT, would be module swaps, not architectural rewrites.

---

