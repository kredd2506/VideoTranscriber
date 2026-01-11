#!/usr/bin/env python3
"""
Chunked transcription for large audio files.
Splits audio into smaller segments, transcribes each, then combines results.
"""

import subprocess
import json
import sys
from pathlib import Path
import logging
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_audio_duration(audio_path):
    """Get audio duration using ffprobe."""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(audio_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        logger.info(f"Audio duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        return duration
    except Exception as e:
        logger.error(f"Failed to get audio duration: {e}")
        return None


def split_audio(audio_path, chunk_duration=600, output_dir=None):
    """
    Split audio file into chunks using ffmpeg.

    Args:
        audio_path: Path to audio file
        chunk_duration: Duration of each chunk in seconds (default: 600 = 10 minutes)
        output_dir: Where to save chunks

    Returns:
        List of chunk file paths
    """
    audio_path = Path(audio_path)

    if output_dir is None:
        output_dir = audio_path.parent / "chunks"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get total duration
    total_duration = get_audio_duration(audio_path)
    if total_duration is None:
        return []

    num_chunks = math.ceil(total_duration / chunk_duration)
    logger.info(f"Splitting into {num_chunks} chunks of {chunk_duration} seconds each")

    chunk_files = []

    for i in range(num_chunks):
        start_time = i * chunk_duration
        chunk_file = output_dir / f"chunk_{i:03d}.wav"

        logger.info(f"Creating chunk {i+1}/{num_chunks} starting at {start_time}s")

        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output files
            '-i', str(audio_path),
            '-ss', str(start_time),
            '-t', str(chunk_duration),
            '-acodec', 'pcm_s16le',
            '-ar', '16000',  # Whisper prefers 16kHz
            '-ac', '1',  # Mono
            str(chunk_file)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            chunk_files.append(chunk_file)
            logger.info(f"✓ Created chunk: {chunk_file}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create chunk {i}: {e}")
            logger.error(f"STDERR: {e.stderr}")

    return chunk_files


def transcribe_chunk(chunk_path, model="tiny"):
    """Transcribe a single audio chunk using Whisper."""
    logger.info(f"Transcribing chunk: {chunk_path.name}")

    output_dir = chunk_path.parent

    cmd = [
        "whisper",
        str(chunk_path),
        "--model", model,
        "--output_dir", str(output_dir),
        "--output_format", "json",
        "--language", "en",
        "--fp16", "False"
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes per chunk
        )

        if result.returncode != 0:
            logger.error(f"Whisper failed for {chunk_path.name}")
            logger.error(f"STDERR: {result.stderr}")
            return None, None

        # Load the JSON output
        json_file = output_dir / f"{chunk_path.stem}.json"

        if json_file.exists():
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                segments = data.get('segments', [])
                text = data.get('text', '')
                logger.info(f"✓ Transcribed {chunk_path.name}: {len(text)} characters")
                return segments, text
        else:
            logger.error(f"JSON output not found: {json_file}")
            return None, None

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout transcribing {chunk_path.name}")
        return None, None
    except Exception as e:
        logger.error(f"Error transcribing {chunk_path.name}: {e}")
        return None, None


def combine_transcripts(chunk_results, chunk_duration=600):
    """
    Combine transcripts from multiple chunks.

    Args:
        chunk_results: List of (segments, text) tuples
        chunk_duration: Duration of each chunk for timestamp adjustment

    Returns:
        Combined (segments, text)
    """
    combined_text = []
    combined_segments = []

    for chunk_idx, (segments, text) in enumerate(chunk_results):
        if segments is None or text is None:
            logger.warning(f"Skipping chunk {chunk_idx} - transcription failed")
            continue

        # Adjust timestamps for this chunk
        time_offset = chunk_idx * chunk_duration

        for segment in segments:
            adjusted_segment = segment.copy()
            adjusted_segment['start'] += time_offset
            adjusted_segment['end'] += time_offset
            combined_segments.append(adjusted_segment)

        combined_text.append(text.strip())

    final_text = '\n\n'.join(combined_text)

    logger.info(f"Combined transcript: {len(final_text)} characters, {len(combined_segments)} segments")

    return combined_segments, final_text


def save_outputs(segments, transcript, output_dir, base_name):
    """Save transcript in multiple formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save transcript as text
    txt_file = output_dir / f"{base_name}_transcript.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(transcript)
    logger.info(f"✓ Saved transcript: {txt_file}")

    # Save segments as JSON
    json_file = output_dir / f"{base_name}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            'text': transcript,
            'segments': segments
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Saved JSON: {json_file}")

    # Generate SRT
    srt_file = output_dir / f"{base_name}.srt"
    with open(srt_file, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, 1):
            start = format_timestamp_srt(seg['start'])
            end = format_timestamp_srt(seg['end'])
            text = seg['text'].strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
    logger.info(f"✓ Saved SRT: {srt_file}")

    # Generate VTT
    vtt_file = output_dir / f"{base_name}.vtt"
    with open(vtt_file, 'w', encoding='utf-8') as f:
        f.write("WEBVTT\n\n")
        for i, seg in enumerate(segments, 1):
            start = format_timestamp_vtt(seg['start'])
            end = format_timestamp_vtt(seg['end'])
            text = seg['text'].strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
    logger.info(f"✓ Saved VTT: {vtt_file}")

    return txt_file, json_file, srt_file, vtt_file


def format_timestamp_srt(seconds):
    """Format timestamp for SRT (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds):
    """Format timestamp for VTT (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def main(audio_path, model="tiny", chunk_duration=600, output_dir=None):
    """
    Main function to transcribe large audio file using chunking.

    Args:
        audio_path: Path to audio file
        model: Whisper model to use
        chunk_duration: Duration of each chunk in seconds
        output_dir: Where to save outputs
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        return False

    if output_dir is None:
        output_dir = audio_path.parent
    else:
        output_dir = Path(output_dir)

    logger.info("=" * 80)
    logger.info("CHUNKED TRANSCRIPTION")
    logger.info("=" * 80)
    logger.info(f"Audio file: {audio_path}")
    logger.info(f"Model: {model}")
    logger.info(f"Chunk duration: {chunk_duration} seconds")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)

    # Step 1: Split audio into chunks
    logger.info("\nStep 1/4: Splitting audio into chunks...")
    chunk_files = split_audio(audio_path, chunk_duration, output_dir / "chunks")

    if not chunk_files:
        logger.error("Failed to split audio file")
        return False

    logger.info(f"✓ Created {len(chunk_files)} chunks")

    # Step 2: Transcribe each chunk
    logger.info(f"\nStep 2/4: Transcribing {len(chunk_files)} chunks...")
    chunk_results = []

    for i, chunk_file in enumerate(chunk_files, 1):
        logger.info(f"Processing chunk {i}/{len(chunk_files)}")
        segments, text = transcribe_chunk(chunk_file, model)
        chunk_results.append((segments, text))

    # Step 3: Combine results
    logger.info("\nStep 3/4: Combining transcripts...")
    combined_segments, combined_text = combine_transcripts(chunk_results, chunk_duration)

    if not combined_text:
        logger.error("No transcript generated")
        return False

    # Step 4: Save outputs
    logger.info("\nStep 4/4: Saving outputs...")
    base_name = audio_path.stem.replace('_audio', '')
    txt_file, json_file, srt_file, vtt_file = save_outputs(
        combined_segments,
        combined_text,
        output_dir,
        base_name
    )

    logger.info("\n" + "=" * 80)
    logger.info("✓ TRANSCRIPTION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Transcript length: {len(combined_text)} characters")
    logger.info(f"Total segments: {len(combined_segments)}")
    logger.info(f"\nOutput files:")
    logger.info(f"  - Transcript (TXT): {txt_file}")
    logger.info(f"  - Data (JSON): {json_file}")
    logger.info(f"  - Subtitles (SRT): {srt_file}")
    logger.info(f"  - Subtitles (VTT): {vtt_file}")
    logger.info("=" * 80)

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe large audio files using chunking")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("-m", "--model", default="tiny", help="Whisper model (default: tiny)")
    parser.add_argument("-c", "--chunk-duration", type=int, default=600, help="Chunk duration in seconds (default: 600)")
    parser.add_argument("-o", "--output-dir", help="Output directory")

    args = parser.parse_args()

    success = main(args.audio_file, args.model, args.chunk_duration, args.output_dir)
    sys.exit(0 if success else 1)
