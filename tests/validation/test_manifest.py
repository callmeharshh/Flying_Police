import os

from config import VALIDATION_FIXTURES_DIR
from data.validation_capture import (
    discover_capture_dirs,
    fixtures_available,
    load_manifest,
    rebuild_manifest_from_captures,
)


def test_fixtures_available_when_captures_on_disk():
    assert fixtures_available()
    assert len(discover_capture_dirs()) > 0


def test_rebuild_manifest_matches_capture_dirs():
    manifest = rebuild_manifest_from_captures(write=True)
    assert manifest["capture_count"] == len(discover_capture_dirs())
    assert len(manifest["captures"]) == manifest["capture_count"]

    manifest_path = os.path.join(VALIDATION_FIXTURES_DIR, "manifest.json")
    assert os.path.isfile(manifest_path)

    loaded = load_manifest()
    assert loaded["capture_count"] == manifest["capture_count"]
