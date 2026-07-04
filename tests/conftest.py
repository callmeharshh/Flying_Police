"""Shared pytest configuration for all test suites."""
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def pytest_collection_modifyitems(config, items):
    from data.validation_capture import fixtures_available

    for item in items:
        path = str(item.fspath)
        if f"{os.sep}tests{os.sep}unit{os.sep}" in path:
            item.add_marker(pytest.mark.unit)
        elif f"{os.sep}tests{os.sep}validation{os.sep}" in path:
            item.add_marker(pytest.mark.validation)
            if not fixtures_available():
                item.add_marker(
                    pytest.mark.skip(
                        reason="No validation fixtures — run scripts/capture_validation_set.py first",
                    )
                )
        elif f"{os.sep}tests{os.sep}integration{os.sep}" in path:
            item.add_marker(pytest.mark.integration)
