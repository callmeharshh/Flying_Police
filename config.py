import os
from dotenv import load_dotenv

load_dotenv()
load_dotenv(".env.local", override=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Storage paths
DATA_DIR = os.path.join(BASE_DIR, "data")
SAMPLE_VIDEO_DIR = os.path.join(DATA_DIR, "sample_video")
DEMO_VIDEO = "entrance_area_720p.mp4"  # BLK-HDPTZ12 parking lot surveillance
SAMPLE_IMAGES_DIR = os.path.join(DATA_DIR, "sample_images")
EVENTS_DB_PATH = os.path.join(DATA_DIR, "events.db")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma")
UI_SESSIONS_DIR = os.path.join(DATA_DIR, "sessions")
UI_UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")

# Video processing
SAMPLE_EVERY_N_FRAMES = 2   # process every Nth raw frame (speed vs accuracy)

# Validation fixtures (BLIP + agent golden captures)
VALIDATION_FIXTURES_DIR = os.path.join(DATA_DIR, "validation_fixtures")
VALIDATION_CAPTURE_EVERY_N = 30   # save every Nth sampled frame
VALIDATION_MAX_CAPTURES = 15      # cap total saved captures per run

# VLM
BLIP_MODEL = "Salesforce/blip-image-captioning-base"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MIN_CONTOUR_AREA = 2000     # px² — ignore smaller blobs
MAX_CONTOUR_AREA_RATIO = 0.40  # reject blobs covering >40% of frame (lighting artifacts)
MOG2_VAR_THRESHOLD = 120    # higher = less sensitive to lighting variation
MOG2_WARMUP_FRAMES = 5

# LLM
OPENAI_MODEL = "gpt-4o-mini"   # cheap, fast, good instruction-following
AGENT_MEMORY_K = 10
AGENT_VERBOSE = False  # True for CLI debug; UI keeps this off for cleaner logs

# Locations
LOCATIONS = ["main_gate", "garage", "perimeter"]

# Known vehicles (recognized = no RULE-02 alert)
KNOWN_VEHICLES = ["blue ford f150", "blue truck"]

# Telegram notifications (optional — leave blank to disable)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Monad evidence anchoring (optional — leave blank to run fully local)
MONAD_RPC_URL = os.getenv("MONAD_RPC_URL", "")
MONAD_CHAIN_ID = int(os.getenv("MONAD_CHAIN_ID", "0") or "0")
MONAD_PRIVATE_KEY = os.getenv("MONAD_PRIVATE_KEY", "")
EVIDENCE_REGISTRY_ADDRESS = os.getenv("EVIDENCE_REGISTRY_ADDRESS", "")
MONAD_EXPLORER_TX_URL = os.getenv("MONAD_EXPLORER_TX_URL", "")

# Lighthouse IPFS evidence storage (optional — leave blank to skip IPFS upload)
LIGHTHOUSE_API_KEY = os.getenv("LIGHTHOUSE_API_KEY", "") or os.getenv("LIGHTHOUSE_TOKEN", "")
LIGHTHOUSE_UPLOAD_URL = "https://upload.lighthouse.storage/api/v0/add"
LIGHTHOUSE_GATEWAY_URL = "https://gateway.lighthouse.storage/ipfs"
LIGHTHOUSE_UPLOAD_TIMEOUT_SECONDS = 45

# Alert thresholds
LOITER_THRESHOLD_SECONDS = 15    # RULE-04: 15 seconds (testing); change to 300 for production
REPEAT_ENTRY_LIMIT = 1           # RULE-03: alert if > this count (fires on 2nd+ entry)

# Scene lighting (RULE-01, RULE-05) — day/night from frame brightness, not wall clock
LIGHTING_CENTER_SAMPLE_RATIO = 0.65       # central crop fraction used for scoring
LIGHTING_NIGHT_MEAN_THRESHOLD = 90        # mean V-channel below this → night
LIGHTING_DARK_PIXEL_VALUE = 50            # pixel V below this counts as "dark"
LIGHTING_NIGHT_DARK_PIXEL_RATIO = 0.55    # dark-pixel fraction that implies night

# Motion tracking — match by position history, not BLIP caption
VEHICLE_TRACK_MAX_FRAME_GAP = 45              # max sampled frames between sightings of same object
VEHICLE_TRACK_MAX_VELOCITY_PX_PER_FRAME = 25  # expected max blob movement per frame
VEHICLE_TRACK_VELOCITY_MARGIN = 1.5           # tolerance multiplier on predicted travel distance
VEHICLE_TRACK_MAX_TRAJECTORY_POINTS = 8     # positions kept for agent / history queries

# Person counting — merge fragmented tracks into distinct individuals
MIN_DISTINCT_PERSON_SEPARATION_PX = 250   # same-frame centers farther apart = 2 people
