"""Integration tests — load BLIP and optional live pipeline paths."""
import pytest

from vlm.blip_analyzer import BLIPAnalyzer


@pytest.fixture(scope="session")
def blip_analyzer() -> BLIPAnalyzer:
    return BLIPAnalyzer()
