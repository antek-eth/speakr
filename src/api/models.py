"""API endpoint for listing available transcription models."""

from flask import Blueprint, jsonify
from flask_login import login_required

models_bp = Blueprint('models', __name__)


@models_bp.route('/api/models')
@login_required
def list_models():
    """Return available transcription models for the UI picker."""
    from src.services.transcription import get_registry
    registry = get_registry()
    return jsonify({'models': registry.list_models()})
