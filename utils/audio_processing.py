try:
    from moviepy.editor import AudioFileClip
except ImportError:
    # For moviepy 2.x
    from moviepy.audio.io.AudioFileClip import AudioFileClip
from pathlib import Path
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_audio(video_path: Path, output_dir=None):
    """
    Extract audio from a video file with robust error handling.

    Args:
        video_path (Path): Path to the video file
        output_dir (Path, optional): Directory to save the audio file.
                                     Defaults to /app/data/outputs or same as video

    Returns:
        Path: Path to the extracted audio file

    Raises:
        RuntimeError: If audio extraction fails
    """
    video_path = Path(video_path)

    if not video_path.exists():
        raise RuntimeError(f"Video file not found: {video_path}")

    # Determine output directory
    if output_dir is None:
        # Try to use /app/data/outputs if it exists (Docker environment)
        if Path("/app/data/outputs").exists():
            output_dir = Path("/app/data/outputs")
        else:
            # Otherwise use the same directory as the video
            output_dir = video_path.parent
    else:
        output_dir = Path(output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define output path
    audio_path = output_dir / f"{video_path.stem}_audio.wav"

    # If audio file already exists and is recent, use it
    if audio_path.exists():
        logger.info(f"Using existing audio file: {audio_path}")
        return audio_path

    audio_clip = None
    try:
        logger.info(f"Extracting audio from: {video_path}")

        # Load the video file
        audio_clip = AudioFileClip(str(video_path))

        if audio_clip is None or audio_clip.duration is None:
            raise RuntimeError("Failed to load audio from video file")

        logger.info(f"Audio duration: {audio_clip.duration:.2f} seconds")

        # Write audio file (moviepy 2.x compatibility)
        # Try without logger parameter first (for moviepy 2.x)
        try:
            audio_clip.write_audiofile(
                str(audio_path),
                logger=None,
                fps=44100,
                nbytes=2,
                codec='pcm_s16le'
            )
        except TypeError:
            # Fallback for older moviepy versions
            audio_clip.write_audiofile(
                str(audio_path),
                fps=44100,
                nbytes=2,
                codec='pcm_s16le'
            )

        # Verify the output file was created
        if not audio_path.exists():
            raise RuntimeError(f"Audio file was not created at: {audio_path}")

        file_size = audio_path.stat().st_size
        if file_size == 0:
            raise RuntimeError(f"Audio file is empty: {audio_path}")

        logger.info(f"Audio extracted successfully: {audio_path} ({file_size / 1024 / 1024:.2f} MB)")

        return audio_path

    except Exception as e:
        # Clean up partial file if it exists
        if audio_path.exists():
            try:
                audio_path.unlink()
                logger.info(f"Cleaned up incomplete audio file: {audio_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up audio file: {cleanup_error}")

        raise RuntimeError(f"Audio extraction failed: {str(e)}")

    finally:
        # Always close the audio clip to free resources
        if audio_clip is not None:
            try:
                audio_clip.close()
            except Exception as e:
                logger.warning(f"Error closing audio clip: {e}")
