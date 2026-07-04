#!/usr/bin/env bash
# Download sample surveillance videos used for testing (macOS).
# Run from project root: bash scripts/download_sample_videos.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${ROOT}/data/sample_video"
mkdir -p "$OUT_DIR"

CMOR_ENTRANCE_URL="https://www.c-mor.com/video-surveillance-demo/sample-recordings-of-the-video-surveillance-system-c-mor?download=27:motion-detection-entrance-area"
CMOR_OUTSIDE_URL="https://www.c-mor.com/video-surveillance-demo/sample-recordings-of-the-video-surveillance-system-c-mor?download=29:motion-detection-outside-entrance"
YOUTUBE_NIGHT_URL="https://www.youtube.com/watch?v=Am8tq-0FQJU"

echo "=== Downloading sample videos → ${OUT_DIR} ==="

download_curl() {
  local url="$1"
  local outfile="$2"
  if [[ -f "${OUT_DIR}/${outfile}" ]]; then
    echo "  skip (exists): ${outfile}"
    return 0
  fi
  echo "  downloading: ${outfile}"
  curl -L --fail --retry 3 -o "${OUT_DIR}/${outfile}" "$url"
}

download_curl "$CMOR_ENTRANCE_URL" "entrance_area_720p.mp4"
download_curl "$CMOR_OUTSIDE_URL" "outside_entry_720p.mp4"

if [[ -f "${OUT_DIR}/night_surveillance_Am8tq-0FQJU.mp4" ]]; then
  echo "  skip (exists): night_surveillance_Am8tq-0FQJU.mp4"
else
  if ! command -v yt-dlp >/dev/null 2>&1; then
    echo ""
    echo "yt-dlp not found. Install for the night-surveillance clip:"
    echo "  brew install yt-dlp"
    echo ""
    echo "Then re-run this script, or download manually:"
    echo "  ${YOUTUBE_NIGHT_URL}"
    exit 1
  fi
  echo "  downloading: night_surveillance_Am8tq-0FQJU.mp4 (YouTube)"
  yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best" \
    --merge-output-format mp4 \
    -o "${OUT_DIR}/night_surveillance_Am8tq-0FQJU.mp4" \
    "$YOUTUBE_NIGHT_URL"
fi

echo ""
echo "Done. Files in ${OUT_DIR}:"
ls -lh "$OUT_DIR"/*.mp4 2>/dev/null || true
echo ""
echo "CLI default (config.py): entrance_area_720p.mp4"
echo "Night / RULE-01 testing:  night_surveillance_Am8tq-0FQJU.mp4"
echo "Person + car motion:      outside_entry_720p.mp4"
