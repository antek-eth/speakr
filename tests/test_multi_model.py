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


# === Config Loader Tests ===

def test_load_valid_yaml_config():
    from src.services.transcription.config_loader import load_models_config
    yaml_content = """
models:
  - id: test-model-1
    name: "Test Model 1"
    connector: openai_whisper
    default: true
    config:
      api_key: test-key-123
    capabilities:
      diarization: false
      max_file_size_mb: 25
  - id: test-model-2
    name: "Test Model 2"
    connector: openai_whisper
    config:
      api_key: test-key-456
    capabilities:
      diarization: false
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        f.flush()
        try:
            models = load_models_config(f.name)
            assert len(models) == 2, f"Expected 2 models, got {len(models)}"
            assert models[0]['id'] == 'test-model-1'
            assert models[0]['default'] is True
            assert models[1]['id'] == 'test-model-2'
            assert models[1].get('default') is not True
        finally:
            os.unlink(f.name)


def test_env_var_interpolation():
    from src.services.transcription.config_loader import load_models_config
    os.environ['TEST_API_KEY_XYZ'] = 'resolved-secret-key'
    yaml_content = """
models:
  - id: env-test
    name: "Env Test"
    connector: openai_whisper
    default: true
    config:
      api_key: ${TEST_API_KEY_XYZ}
    capabilities:
      diarization: false
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        f.flush()
        try:
            models = load_models_config(f.name)
            assert models[0]['config']['api_key'] == 'resolved-secret-key', \
                f"Expected 'resolved-secret-key', got '{models[0]['config']['api_key']}'"
        finally:
            os.unlink(f.name)
            del os.environ['TEST_API_KEY_XYZ']


def test_missing_env_var_raises_error():
    from src.services.transcription.config_loader import load_models_config
    from src.services.transcription.exceptions import ConfigurationError
    os.environ.pop('NONEXISTENT_VAR_ABC', None)
    yaml_content = """
models:
  - id: missing-env
    name: "Missing Env"
    connector: openai_whisper
    default: true
    config:
      api_key: ${NONEXISTENT_VAR_ABC}
    capabilities:
      diarization: false
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        f.flush()
        try:
            raised = False
            try:
                load_models_config(f.name)
            except ConfigurationError:
                raised = True
            assert raised, "Should raise ConfigurationError for missing env var"
        finally:
            os.unlink(f.name)


def test_duplicate_model_ids_raises_error():
    from src.services.transcription.config_loader import load_models_config
    from src.services.transcription.exceptions import ConfigurationError
    yaml_content = """
models:
  - id: dup-model
    name: "Model A"
    connector: openai_whisper
    default: true
    config:
      api_key: key1
    capabilities:
      diarization: false
  - id: dup-model
    name: "Model B"
    connector: openai_whisper
    config:
      api_key: key2
    capabilities:
      diarization: false
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        f.flush()
        try:
            raised = False
            try:
                load_models_config(f.name)
            except ConfigurationError:
                raised = True
            assert raised, "Should raise ConfigurationError for duplicate model IDs"
        finally:
            os.unlink(f.name)


def test_missing_yaml_returns_none():
    from src.services.transcription.config_loader import load_models_config
    result = load_models_config('/nonexistent/path/transcription-models.yaml')
    assert result is None, "Should return None for missing YAML file"


def test_multiple_defaults_raises_error():
    from src.services.transcription.config_loader import load_models_config
    from src.services.transcription.exceptions import ConfigurationError
    yaml_content = """
models:
  - id: model-a
    name: "Model A"
    connector: openai_whisper
    default: true
    config:
      api_key: key1
    capabilities:
      diarization: false
  - id: model-b
    name: "Model B"
    connector: openai_whisper
    default: true
    config:
      api_key: key2
    capabilities:
      diarization: false
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        f.flush()
        try:
            raised = False
            try:
                load_models_config(f.name)
            except ConfigurationError:
                raised = True
            assert raised, "Should raise ConfigurationError for multiple default: true"
        finally:
            os.unlink(f.name)


if __name__ == '__main__':
    print("\n=== Multi-Model Tests ===\n")

    print("-- Database Column Tests --")
    run_test("Recording has transcription_model_id", test_recording_has_model_id_column)
    run_test("Recording has source_url", test_recording_has_source_url_column)

    print("\n-- Config Loader Tests --")
    run_test("Load valid YAML config", test_load_valid_yaml_config)
    run_test("Env var interpolation", test_env_var_interpolation)
    run_test("Missing env var raises error", test_missing_env_var_raises_error)
    run_test("Duplicate model IDs raises error", test_duplicate_model_ids_raises_error)
    run_test("Missing YAML returns None", test_missing_yaml_returns_none)
    run_test("Multiple defaults raises error", test_multiple_defaults_raises_error)

    print(f"\n{'='*50}")
    print(f"Results: {PASSED} passed, {FAILED} failed")
    if ERRORS:
        print("Failures:")
        for name, err in ERRORS:
            print(f"  - {name}: {err}")
    sys.exit(1 if FAILED else 0)
