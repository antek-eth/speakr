"""
Amazon Bedrock Converse API connector for audio transcription.

Supports Voxtral and other Bedrock models that accept audio input
via the Converse API content blocks.
"""

import logging
import os
from typing import Dict, Any, Set

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

# Mapping of audio MIME types to Bedrock audio format strings
MIME_TO_FORMAT = {
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/flac": "flac",
    "audio/ogg": "ogg",
    "audio/webm": "webm",
    "audio/mp4": "mp4",
    "audio/m4a": "mp4",
}


class BedrockConverseConnector(BaseTranscriptionConnector):
    """Connector for Amazon Bedrock models with audio input via Converse API."""

    CAPABILITIES: Set[TranscriptionCapability] = {
        TranscriptionCapability.LANGUAGE_DETECTION,
        TranscriptionCapability.TIMESTAMPS,
    }
    PROVIDER_NAME = "bedrock_converse"

    # Bedrock models handle large inputs, but audio size limits vary by model
    SPECIFICATIONS = ConnectorSpecifications(
        max_file_size_bytes=25 * 1024 * 1024,  # 25MB safe default
        handles_chunking_internally=False,
        recommended_chunk_seconds=600,
    )

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        import boto3

        self.model_id = config['model_id']
        self.region = config.get('region', 'us-east-1')
        self.prompt = config.get('prompt', 'Transcribe this audio. Return only the transcription text, nothing else.')
        self.diarize_prompt = config.get('diarize_prompt',
            'Transcribe this audio with speaker diarization. '
            'Format each line as "SPEAKER_XX: text". '
            'Identify different speakers and label them consistently.')
        self.max_tokens = config.get('max_tokens', 16384)

        # Build boto3 client with optional credentials
        client_kwargs = {
            'service_name': 'bedrock-runtime',
            'region_name': self.region,
        }
        if config.get('aws_access_key_id') and config.get('aws_secret_access_key'):
            client_kwargs['aws_access_key_id'] = config['aws_access_key_id']
            client_kwargs['aws_secret_access_key'] = config['aws_secret_access_key']
            if config.get('aws_session_token'):
                client_kwargs['aws_session_token'] = config['aws_session_token']

        self.client = boto3.client(**client_kwargs)

    def _validate_config(self) -> None:
        if not self.config.get('model_id'):
            raise ConfigurationError("model_id is required for Bedrock Converse connector")

    def _detect_audio_format(self, request: TranscriptionRequest) -> str:
        """Detect audio format from MIME type or filename."""
        if request.mime_type and request.mime_type in MIME_TO_FORMAT:
            return MIME_TO_FORMAT[request.mime_type]

        if request.filename:
            ext = request.filename.rsplit('.', 1)[-1].lower()
            ext_map = {'mp3': 'mp3', 'wav': 'wav', 'flac': 'flac',
                       'ogg': 'ogg', 'webm': 'webm', 'mp4': 'mp4', 'm4a': 'mp4'}
            if ext in ext_map:
                return ext_map[ext]

        return 'mp3'  # safe default

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResponse:
        try:
            audio_bytes = request.audio_file.read()
            audio_format = self._detect_audio_format(request)

            # Build the prompt
            if request.diarize:
                system_prompt = self.diarize_prompt
            else:
                system_prompt = self.prompt

            if request.language:
                system_prompt += f" The audio is in {request.language}."

            if request.hotwords:
                system_prompt += f" Key terms: {request.hotwords}."

            # Build Converse API message with audio content block
            content_blocks = [
                {
                    "audio": {
                        "format": audio_format,
                        "source": {
                            "bytes": audio_bytes
                        }
                    }
                },
                {
                    "text": system_prompt
                }
            ]

            logger.info(f"Sending audio to Bedrock Converse: model={self.model_id}, "
                        f"format={audio_format}, size={len(audio_bytes)/1024/1024:.1f}MB")

            converse_kwargs = {
                'modelId': self.model_id,
                'messages': [
                    {
                        'role': 'user',
                        'content': content_blocks
                    }
                ],
                'inferenceConfig': {
                    'maxTokens': self.max_tokens,
                    'temperature': 0.0,
                }
            }

            response = self.client.converse(**converse_kwargs)

            # Extract text from response
            output = response.get('output', {})
            message = output.get('message', {})
            content = message.get('content', [])

            text_parts = []
            for block in content:
                if 'text' in block:
                    text_parts.append(block['text'])

            full_text = '\n'.join(text_parts)

            if not full_text.strip():
                raise TranscriptionError("Bedrock returned empty transcription")

            logger.info(f"Bedrock transcription complete: {len(full_text)} chars, "
                        f"model={self.model_id}")

            # Parse diarized output if requested
            segments = None
            speakers = None
            if request.diarize:
                segments, speakers = self._parse_diarized_text(full_text)

            usage = response.get('usage', {})
            logger.info(f"Bedrock usage: input_tokens={usage.get('inputTokens', '?')}, "
                        f"output_tokens={usage.get('outputTokens', '?')}")

            return TranscriptionResponse(
                text=full_text,
                segments=segments,
                language=request.language,
                speakers=speakers,
                provider=self.PROVIDER_NAME,
                model=self.model_id,
                raw_response={'usage': usage},
            )

        except self.client.exceptions.ClientError as e:
            error_code = e.response['Error']['Code']
            error_msg = e.response['Error']['Message']
            logger.error(f"Bedrock API error: {error_code} - {error_msg}")
            raise TranscriptionError(f"Bedrock API error ({error_code}): {error_msg}") from e
        except TranscriptionError:
            raise
        except Exception as e:
            logger.error(f"Bedrock transcription failed: {e}")
            raise TranscriptionError(f"Bedrock transcription failed: {e}") from e

    def _parse_diarized_text(self, text):
        """Parse speaker-labeled text into segments."""
        import re
        segments = []
        speakers_set = set()
        pattern = re.compile(r'^(SPEAKER_\w+):\s*(.+)', re.MULTILINE)

        for match in pattern.finditer(text):
            speaker = match.group(1)
            sentence = match.group(2).strip()
            speakers_set.add(speaker)
            segments.append(TranscriptionSegment(
                text=sentence,
                speaker=speaker,
            ))

        if not segments:
            # Fallback: return as single segment
            segments.append(TranscriptionSegment(text=text))

        speakers = sorted(speakers_set) if speakers_set else None
        return segments if segments else None, speakers

    def health_check(self) -> bool:
        return bool(self.config.get('model_id'))

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["model_id"],
            "properties": {
                "model_id": {
                    "type": "string",
                    "description": "Bedrock model ID (e.g. mistral.voxtral-mini-3b-2507)"
                },
                "region": {
                    "type": "string",
                    "default": "us-east-1",
                    "description": "AWS region"
                },
                "aws_access_key_id": {
                    "type": "string",
                    "description": "AWS access key (optional, uses instance profile if omitted)"
                },
                "aws_secret_access_key": {
                    "type": "string",
                    "description": "AWS secret key (optional)"
                },
                "max_tokens": {
                    "type": "integer",
                    "default": 16384,
                    "description": "Max tokens for transcription output"
                },
            }
        }
