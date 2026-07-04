# Product Requirements Document
## Flying Police

**Version:** 1.0  
**Date:** 2026-06-09  
**Author:** Jayesh Shete  

---

## 1. Overview

### 1.1 Product Summary

Flying Police is an AI-powered security monitoring system for a fixed property. A docked drone captures video footage daily. The system processes that footage frame-by-frame, identifies objects and events, generates alerts for security threats, and maintains a searchable index of all activity.

### 1.2 Problem Being Solved

Property owners currently rely on passive CCTV systems that require human review after an incident. There is no real-time automated analysis, no cross-session object tracking ("that truck has entered 3 times this week"), and no intelligent alerting based on context (time of day, location, behavior).

### 1.3 Value Proposition

> Automated, AI-powered drone surveillance that detects, logs, and alerts property owners to security events in real time — without requiring manual video review.

---

## 2. Goals & Non-Goals

### Goals
- Process simulated drone video frames and telemetry data
- Identify objects (vehicles, people) and log them with context
- Generate security alerts based on predefined rules
- Index all frames for searchability by time, object, and location
- Demonstrate a working prototype with clear architecture

### Non-Goals
- Real drone hardware integration
- Live video streaming
- Model training or fine-tuning
- Mobile/web application UI
- Multi-property support

---

## 3. Users

| User | Description | Primary Need |
|------|-------------|-------------|
| Property Owner | Homeowner or facility manager | Know what happened, when, and get alerted |
| Security Operator | Reviews logs and alerts | Query past events, investigate incidents |

---

## 4. Key Requirements

### KR-1: Frame Analysis
The system must analyze each video frame (real or simulated) and extract structured information:
- Objects present (vehicle type, color, person)
- Activity (entering, exiting, loitering, stationary)
- Location on property (main gate, garage, perimeter)
- Timestamp

### KR-2: Alert Generation
The system must trigger alerts based on predefined rules:
- Person detected between 22:00 – 06:00
- Unknown/unrecognized vehicle on property
- Person loitering > 5 minutes in one location
- Same vehicle entering more than twice in a day

### KR-3: Frame Indexing & Search
All processed frames must be stored and queryable:
- Query by time range ("what happened at midnight?")
- Query by object ("show all truck events")
- Query by location ("activity at main gate today")

---

## 5. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT LAYER                          │
│  Simulated Video Frames  +  Drone Telemetry Data        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│           BACKGROUND SUBTRACTION LAYER                  │
│  OpenCV MOG2 (background_subtractor.py)                 │
│  • Learns static scene: gate, road, trees               │
│  • Returns foreground mask of NEW objects only          │
│  • Extracts bounding box crop per new object            │
│  • Empty mask → skip frame (no new activity)            │
└────────────────────┬────────────────────────────────────┘
                     │ crops of new objects only
                     ▼
┌─────────────────────────────────────────────────────────┐
│                    VLM LAYER                            │
│  BLIP (Salesforce/blip-image-captioning-base)           │
│  • Runs on each object crop independently               │
│  • Caption + color + person/vehicle per crop            │
│  • Results merged into one FrameAnalysis per frame      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  AGENT LAYER (LangChain)                │
│  LLM: gpt-4o-mini via OpenAI API                        │
│  Tools:                                                 │
│  • log_event     — write to event log                   │
│  • trigger_alert — fire security alert                  │
│  • query_history — check if object seen before          │
│  Memory: ConversationBufferWindowMemory (k=10)          │
└──────────┬──────────────────────────┬───────────────────┘
           │                          │
           ▼                          ▼
┌──────────────────┐      ┌───────────────────────────────┐
│   EVENT LOG      │      │       FRAME INDEX             │
│   (SQLite)       │      │   ChromaDB (vector search)    │
│   • Logs         │      │   • Frames + embeddings       │
│   • Alerts       │      │   • Queryable by semantics    │
└──────────────────┘      └───────────────────────────────┘
```

---

## 6. Tech Stack

| Component | Technology | Reason |
|-----------|-----------|--------|
| Language | Python 3.14 | Required by PS |
| Background Subtraction | OpenCV MOG2 | Fixed drone = static background; isolates new objects before VLM, eliminates false positives from static scene |
| VLM | BLIP (`Salesforce/blip-image-captioning-base`) | Lightweight, CPU-friendly; runs per object crop for focused descriptions |
| Agent Framework | LangChain | Required by PS |
| LLM Backend | OpenAI `gpt-4o-mini` | Fast, cost-effective, strong instruction-following; accessed via `langchain-openai` |
| Frame Indexing | ChromaDB | Semantic search, easy setup |
| Event Storage | SQLite | Lightweight, no server needed |
| Video Processing | OpenCV | Required by PS |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Fast, CPU-friendly |

---

## 7. Data Model

### 7.1 Frame Record
```json
{
  "frame_id": 1,
  "timestamp": "2026-06-09T00:01:00",
  "location": "main_gate",
  "raw_description": "A person standing near the gate at night",
  "objects": ["person"],
  "activity": "loitering",
  "threat_level": "high",
  "telemetry": {
    "drone_lat": 18.5204,
    "drone_lon": 73.8567,
    "altitude_m": 10,
    "battery_pct": 85
  }
}
```

### 7.2 Event Log Entry
```json
{
  "event_id": "evt_001",
  "frame_id": 1,
  "timestamp": "2026-06-09T00:01:00",
  "type": "alert",
  "message": "Person loitering at main gate, 00:01",
  "severity": "high"
}
```

---

## 8. Alert Rules

| Rule ID | Condition | Severity | Example Output |
|---------|-----------|----------|---------------|
| RULE-01 | Person detected between 22:00–06:00 | High | "Person loitering at main gate, 00:01" |
| RULE-02 | Unknown vehicle on property | Medium | "Unrecognized vehicle at garage, 14:30" |
| RULE-03 | Same vehicle enters > 2x in one day | Medium | "Blue Ford F150 entered 3 times today" |
| RULE-04 | Person stationary > 5 min | High | "Person loitering at perimeter, 23:45" |
| RULE-05 | Any activity at perimeter after hours | High | "Movement detected at perimeter, 02:15" |

---

## 9. Simulated Scenarios

These scenarios cover all required outputs from the PS:

| Scenario | Time | Location | Frame Description | Expected Output |
|----------|------|----------|-------------------|----------------|
| S-01 | 08:00 | main_gate | Blue Ford F150 entering | LOG: Vehicle entry |
| S-02 | 12:00 | garage | Blue Ford F150 parked | LOG: Vehicle at garage |
| S-03 | 08:30 | main_gate | Blue Ford F150 entering again | LOG + ALERT: Vehicle entered twice |
| S-04 | 23:58 | main_gate | Person approaching gate | ALERT: Person after hours |
| S-05 | 00:01 | main_gate | Person standing at gate | ALERT: Loitering at midnight |
| S-06 | 14:00 | perimeter | Unknown white sedan | ALERT: Unrecognized vehicle |
| S-07 | 09:00 | garage | No activity | LOG: Clear |

---

## 10. Query Interface

Users can query the index using natural language:

```
> show all truck events
> what happened at midnight?
> any alerts today?
> show activity at main gate
> how many times did the blue truck enter?
```

---

## 11. File Structure

```
```
Flying-Police/
├── main.py                        # Entry point
├── config.py                      # Config and constants
├── data/
│   ├── simulated_frames.py        # Simulated frame scenarios
│   ├── telemetry.py               # Simulated telemetry generator
│   ├── extract_frames.py          # Stanford dataset frame extractor utility
│   └── sample_images/             # Generated scene images (visual shapes, no text)
├── vlm/
│   ├── background_subtractor.py   # OpenCV MOG2 — isolates new objects per frame
│   └── blip_analyzer.py           # BLIP analysis per object crop
├── agent/
│   ├── security_agent.py          # LangChain ReAct agent (gpt-4o-mini via OpenAI)
│   ├── tools.py                   # Agent tools (log, alert, query)
│   └── alert_rules.py             # Predefined deterministic alert rules engine
├── storage/
│   ├── event_store.py             # SQLite event log
│   └── frame_index.py             # ChromaDB frame index
├── query/
│   └── query_engine.py            # Natural language query interface
├── tests/
│   ├── test_agent.py
│   ├── test_alerts.py
│   └── test_indexing.py
└── README.md
```

---

## 12. Test Cases

| Test ID | Description | Input | Expected Output |
|---------|-------------|-------|----------------|
| TC-01 | Truck logged correctly | Frame: blue truck at gate, 08:00 | Log entry created |
| TC-02 | Midnight alert triggered | Frame: person at gate, 00:01 | Alert: severity=high |
| TC-03 | Repeat vehicle detected | Same truck enters twice | Alert: RULE-03 triggered |
| TC-04 | Frame indexed correctly | Any frame processed | Queryable in ChromaDB |
| TC-05 | Query returns results | "show all truck events" | Returns S-01, S-02, S-03 |
| TC-06 | No false alert during day | Person at gate, 10:00 | LOG only, no alert |
| TC-07 | Telemetry attached to frame | Any frame | telemetry fields populated |

---

## 13. Bonus Features (if time permits)

### 13.1 Video Summarization
After processing all frames, generate a one-line session summary:
> "Blue Ford F150 entered property twice. One person detected loitering at midnight near main gate."

### 13.2 Follow-up Q&A
Allow natural language questions after a session:
> "What objects were seen today?"  
> "Was any vehicle flagged?"

---

## 14. Out of Scope for v1

- Real video file processing (MP4, etc.)
- Face recognition
- License plate reading
- Push notifications
- Web dashboard

---

## 15. Success Criteria

The prototype is considered complete when:
- [ ] All 7 simulated scenarios produce correct log/alert output
- [ ] All 7 test cases pass
- [ ] Frame index supports at least 3 query types
- [ ] BLIP correctly describes at least 1 real image
- [ ] LangChain agent uses cross-frame memory (knows truck entered before)
- [ ] README covers setup + run instructions
