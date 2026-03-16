#!/usr/bin/env python3
"""Tests for multi-model transcription support."""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PASSED = 0
FAILED = 0
ERRORS = []


def run_test(name, func):
    global PASSED, FAILED, ERRORS
    try:
        func()
        print(f"  \u2713 {name}")
        PASSED += 1
    except AssertionError as e:
        print(f"  \u2717 {name}: {e}")
        FAILED += 1
        ERRORS.append((name, str(e)))
    except Exception as e:
        print(f"  \u2717 {name}: EXCEPTION - {e}")
        FAILED += 1
        ERRORS.append((name, f"Exception: {e}"))


# === Database Column Tests ===

def test_recording_has_model_id_column():
    from src.models.recording import Recording
    assert hasattr(Recording, 'transcription_model_id'), \
        "Recording should have transcription_model_id column"


def test_recording_has_source_url_column():
    from src.models.recording import Recording
    assert hasattr(Recording, 'source_url'), \
        "Recording should have source_url column"


if __name__ == '__main__':
    print("\n=== Multi-Model Tests ===\n")

    print("-- Database Column Tests --")
    run_test("Recording has transcription_model_id", test_recording_has_model_id_column)
    run_test("Recording has source_url", test_recording_has_source_url_column)

    print(f"\n{'='*50}")
    print(f"Results: {PASSED} passed, {FAILED} failed")
    if ERRORS:
        print("Failures:")
        for name, err in ERRORS:
            print(f"  - {name}: {err}")
    sys.exit(1 if FAILED else 0)
