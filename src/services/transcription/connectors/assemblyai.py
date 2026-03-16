"""
AssemblyAI transcription connector.

Supports speaker diarization, language detection, and large file handling.
AssemblyAI processes files asynchronously — upload, poll until complete.
"""

import logging
import time
import httpx
from typing import Dict, Any, Set, Optional

from ..base import (
    BaseTranscriptionConnector,
    TranscriptionCapability,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionSegment,
    ConnectorSpecifications,
)
from ..exceptions import TranscriptionError, ConfigurationError

logger = logging.getLogger(__name__)

ASSEMBLYAI_API_BASE = "https://api.assemblyai.com/v2"


class AssemblyAIConnector(BaseTranscriptionConnector):
    """Connector for AssemblyAI transcription API."""

    CAPABILITIES: Set[TranscriptionCapability] = {
        TranscriptionCapability.DIARIZATION,
        TranscriptionCapability.TIMESTAMPS,
        TranscriptionCapability.LANGUAGE_DETECTION,
        TranscriptionCapability.SPEAKER_COUNT_CONTROL,
    }
    PROVIDER_NAME = "assemblyai"

    # AssemblyAI handles large files internally (up to 5GB upload)
    SPECIFICATIONS = ConnectorSpecifications(
        max_file_size_bytes=5 * 1024 * 1024 * 1024,  # 5GB
        handles_chunking_internally=True,
        recommended_chunk_seconds=3600,
    )

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config['api_key']
        self.base_url = config.get('base_url', ASSEMBLYAI_API_BASE)
        self.poll_interval = config.get('poll_interval', 5)
        self.max_poll_time = config.get('max_poll_time', 3600)  # 1 hour max
        # speech_models is an array for priority-based model routing
        # e.g. ["universal-3-pro", "universal-2"] falls back from v3 to v2 for unsupported languages
        self.speech_models = config.get('speech_models', ['universal-3-pro'])
        self._headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json",
        }

    def _validate_config(self) -> None:
        if not self.config.get('api_key'):
            raise ConfigurationError("api_key is required for AssemblyAI connector")

    def _upload_audio(self, audio_file) -> str:
        """Upload audio file and return the upload URL."""
        logger.info("Uploading audio to AssemblyAI...")
        with httpx.Client(timeout=600) as client:
            resp = client.post(
                f"{self.base_url}/upload",
                headers={"Authorization": self.api_key},
                content=audio_file.read(),
            )
            if resp.status_code != 200:
                raise TranscriptionError(f"AssemblyAI upload failed ({resp.status_code}): {resp.text}")
            upload_url = resp.json().get("upload_url")
            if not upload_url:
                raise TranscriptionError("AssemblyAI upload returned no upload_url")
            logger.info("Audio uploaded to AssemblyAI successfully")
            return upload_url

    def _create_transcript(self, audio_url: str, request: TranscriptionRequest) -> str:
        """Create a transcription job and return the transcript ID."""
        body: Dict[str, Any] = {
            "audio_url": audio_url,
        }

        # Diarization
        if request.diarize:
            body["speaker_labels"] = True
            if request.min_speakers:
                body["speakers_expected"] = request.min_speakers
            elif request.max_speakers:
                body["speakers_expected"] = request.max_speakers

        # Language
        if request.language:
            body["language_code"] = request.language
        else:
            body["language_detection"] = True

        # Speech models (priority-based routing)
        if self.speech_models:
            body["speech_models"] = self.speech_models

        # Word boost (hotwords)
        if request.hotwords:
            words = [w.strip() for w in request.hotwords.split(",") if w.strip()]
            if words:
                body["word_boost"] = words

        logger.info(f"Creating AssemblyAI transcript (diarize={request.diarize}, lang={request.language or 'auto'})")

        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{self.base_url}/transcript",
                headers=self._headers,
                json=body,
            )
            if resp.status_code != 200:
                raise TranscriptionError(f"AssemblyAI create transcript failed ({resp.status_code}): {resp.text}")
            data = resp.json()
            transcript_id = data.get("id")
            if not transcript_id:
                raise TranscriptionError("AssemblyAI returned no transcript ID")
            logger.info(f"AssemblyAI transcript created: {transcript_id}")
            return transcript_id

    def _poll_transcript(self, transcript_id: str) -> Dict[str, Any]:
        """Poll until transcript is completed or errored."""
        start = time.time()
        with httpx.Client(timeout=30) as client:
            while True:
                elapsed = time.time() - start
                if elapsed > self.max_poll_time:
                    raise TranscriptionError(f"AssemblyAI transcript {transcript_id} timed out after {int(elapsed)}s")

                resp = client.get(
                    f"{self.base_url}/transcript/{transcript_id}",
                    headers=self._headers,
                )
                if resp.status_code != 200:
                    raise TranscriptionError(f"AssemblyAI poll failed ({resp.status_code}): {resp.text}")

                data = resp.json()
                status = data.get("status")

                if status == "completed":
                    logger.info(f"AssemblyAI transcript {transcript_id} completed in {int(elapsed)}s")
                    return data
                elif status == "error":
                    error = data.get("error", "Unknown error")
                    raise TranscriptionError(f"AssemblyAI transcription failed: {error}")
                else:
                    logger.debug(f"AssemblyAI transcript {transcript_id} status: {status} ({int(elapsed)}s)")
                    time.sleep(self.poll_interval)

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResponse:
        try:
            # Step 1: Upload audio
            upload_url = self._upload_audio(request.audio_file)

            # Step 2: Create transcript
            transcript_id = self._create_transcript(upload_url, request)

            # Step 3: Poll for completion
            result = self._poll_transcript(transcript_id)

            # Step 4: Parse response
            return self._parse_response(result, request.diarize)

        except TranscriptionError:
            raise
        except Exception as e:
            logger.error(f"AssemblyAI transcription failed: {e}")
            raise TranscriptionError(f"AssemblyAI transcription failed: {e}") from e

    def _parse_response(self, data: Dict[str, Any], diarize: bool) -> TranscriptionResponse:
        """Parse AssemblyAI response into standard format."""
        full_text = data.get("text", "")
        language = data.get("language_code")
        duration = data.get("audio_duration")
        model_used = data.get("speech_model_used") or data.get("speech_model") or "universal"
        speakers_set = set()
        segments = []

        if diarize and data.get("utterances"):
            # Use utterances for diarized output
            for utt in data["utterances"]:
                speaker = utt.get("speaker", "Unknown")
                speakers_set.add(speaker)
                segments.append(TranscriptionSegment(
                    text=utt.get("text", ""),
                    speaker=f"SPEAKER_{speaker}" if speaker.isalpha() else speaker,
                    start_time=utt.get("start", 0) / 1000.0,  # ms -> seconds
                    end_time=utt.get("end", 0) / 1000.0,
                    confidence=utt.get("confidence"),
                ))
        elif data.get("words"):
            # Non-diarized: build segments from words grouped by sentences
            # Use the full text as a single segment with timing
            words = data["words"]
            if words:
                segments.append(TranscriptionSegment(
                    text=full_text,
                    start_time=words[0].get("start", 0) / 1000.0,
                    end_time=words[-1].get("end", 0) / 1000.0,
                ))

        speakers = sorted(speakers_set) if speakers_set else None

        return TranscriptionResponse(
            text=full_text,
            segments=segments if segments else None,
            language=language,
            duration=duration,
            speakers=speakers,
            provider=self.PROVIDER_NAME,
            model=model_used,
            raw_response=data,
        )

    def health_check(self) -> bool:
        return bool(self.api_key)

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["api_key"],
            "properties": {
                "api_key": {
                    "type": "string",
                    "description": "AssemblyAI API key"
                },
                "speech_models": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["universal-3-pro"],
                    "description": "Speech models in priority order: 'universal-3-pro', 'universal-2', 'nano'"
                },
                "poll_interval": {
                    "type": "integer",
                    "default": 5,
                    "description": "Seconds between status polls"
                },
            }
        }
