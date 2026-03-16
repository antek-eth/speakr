"""
YAML config loader for multi-model transcription support.

Reads config/transcription-models.yaml, resolves ${ENV_VAR} tokens,
and returns validated model definitions.
"""

import os
import re
import logging

import yaml

from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)

ENV_VAR_PATTERN = re.compile(r'\$\{([A-Z_][A-Z0-9_]*)\}')


def _resolve_env_vars(value):
    """Recursively resolve ${VAR_NAME} tokens in config values."""
    if isinstance(value, str):
        def replacer(match):
            var_name = match.group(1)
            env_val = os.environ.get(var_name)
            if env_val is None:
                raise ConfigurationError(
                    f"Environment variable '{var_name}' is not set "
                    f"(referenced in transcription-models.yaml)"
                )
            return env_val
        return ENV_VAR_PATTERN.sub(replacer, value)
    elif isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


def load_models_config(path):
    """
    Load and validate transcription models from YAML config.

    Args:
        path: Path to transcription-models.yaml

    Returns:
        List of model dicts with resolved env vars, or None if file doesn't exist.

    Raises:
        ConfigurationError: If YAML is invalid, env vars are missing, or IDs are duplicated.
    """
    if not os.path.exists(path):
        logger.info(f"No transcription models config at {path}, using env var fallback")
        return None

    try:
        with open(path, 'r') as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML in {path}: {e}")

    if not raw or 'models' not in raw:
        raise ConfigurationError(f"transcription-models.yaml must contain a 'models' key")

    models = raw['models']
    if not isinstance(models, list) or len(models) == 0:
        raise ConfigurationError(f"'models' must be a non-empty list")

    # Check for duplicate IDs before resolving env vars
    seen_ids = set()
    for model in models:
        model_id = model.get('id')
        if not model_id:
            raise ConfigurationError("Each model must have an 'id' field")
        if model_id in seen_ids:
            raise ConfigurationError(f"Duplicate model id: '{model_id}'")
        seen_ids.add(model_id)

    # Resolve env vars in all config values
    models = _resolve_env_vars(models)

    # Validate required fields and check for exactly one default
    default_count = 0
    for model in models:
        for field in ('id', 'name', 'connector'):
            if not model.get(field):
                raise ConfigurationError(f"Model '{model.get('id', '?')}' missing required field: {field}")
        if model.get('default'):
            default_count += 1

    if default_count > 1:
        raise ConfigurationError(f"Only one model can be marked default: true (found {default_count})")

    # If no explicit default, first model is default
    if default_count == 0:
        models[0]['default'] = True

    logger.info(f"Loaded {len(models)} transcription models from {path}")
    return models
