#!/usr/bin/env python3
"""
Stable transcription script that uses Whisper CLI instead of Python API.
This avoids memory issues with large files.
"""

import subprocess
import sys
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def transcribe_with_cli(audio_path, model="tiny", output_dir=None):
    """
    Transcribe audio using Whisper CLI which is more stable than the API.

    Args:
        audio_path: Path to audio file
        model: Whisper model size
        output_dir: Where to save outputs

    Returns:
        tuple: (segments, transcript_text)
    """
    audio_path = Path(audio_path)

    if output_dir is None:
        output_dir = audio_path.parent
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Transcribing {audio_path} with model {model}")
    logger.info(f"Output directory: {output_dir}")

    try:
        # Run whisper CLI - more memory efficient
        cmd = [
            "whisper",
            str(audio_path),
            "--model", model,
            "--output_dir", str(output_dir),
            "--output_format", "json",
            "--output_format", "txt",
            "--output_format", "srt",
            "--output_format", "vtt",
            "--language", "en",
            "--fp16", "False"  # Disable FP16 for CPU
        ]

        logger.info(f"Running command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        if result.returncode != 0:
            logger.error(f"Whisper failed with return code {result.returncode}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            return None, None

        logger.info("Whisper transcription completed!")
        logger.info(f"STDOUT: {result.stdout}")

        # Load the generated files
        base_name = audio_path.stem
        json_file = output_dir / f"{base_name}.json"
        txt_file = output_dir / f"{base_name}.txt"

        # Load transcript
        transcript = ""
        if txt_file.exists():
            with open(txt_file, 'r', encoding='utf-8') as f:
                transcript = f.read().strip()
            logger.info(f"Loaded transcript: {len(transcript)} characters")
        else:
            logger.warning(f"Transcript file not found: {txt_file}")

        # Load segments
        segments = []
        if json_file.exists():
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                segments = data.get('segments', [])
            logger.info(f"Loaded {len(segments)} segments")
        else:
            logger.warning(f"JSON file not found: {json_file}")

        return segments, transcript

    except subprocess.TimeoutExpired:
        logger.error("Whisper transcription timed out after 1 hour")
        return None, None
    except Exception as e:
        logger.error(f"Error during transcription: {e}", exc_info=True)
        return None, None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python stable_transcribe.py <audio_file> [model] [output_dir]")
        sys.exit(1)

    audio_file = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "tiny"
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None

    segments, transcript = transcribe_with_cli(audio_file, model, output_dir)

    if transcript:
        print(f"✓ Transcription successful!")
        print(f"  Transcript length: {len(transcript)} characters")
        print(f"  Segments: {len(segments)}")
        sys.exit(0)
    else:
        print("✗ Transcription failed!")
        sys.exit(1)
