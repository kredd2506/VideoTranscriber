#!/usr/bin/env python3
"""
Simple Video Transcriber UI
Uses local Mac Whisper + Ollama for transcription and summarization
"""

import streamlit as st
import subprocess
from pathlib import Path
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_ollama_models():
    """Get available Ollama models."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # Parse ollama list output
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            models = [line.split()[0] for line in lines if line.strip()]
            return models
        return []
    except Exception as e:
        logger.error(f"Failed to get Ollama models: {e}")
        return []


def check_ollama_running():
    """Check if Ollama is running."""
    try:
        result = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/tags"],
            capture_output=True,
            timeout=2
        )
        return result.returncode == 0
    except:
        return False


def transcribe_video(video_path, whisper_model="base", output_dir=None):
    """
    Transcribe video using local Whisper.

    Args:
        video_path: Path to video file
        whisper_model: Whisper model (tiny, base, small, medium, large)
        output_dir: Where to save outputs

    Returns:
        dict with paths to generated files
    """
    video_path = Path(video_path)

    if output_dir is None:
        output_dir = video_path.parent / "outputs"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Transcribing {video_path} with Whisper {whisper_model}")

    # Run Whisper
    cmd = [
        "whisper",
        str(video_path),
        "--model", whisper_model,
        "--output_dir", str(output_dir),
        "--output_format", "txt",
        "--output_format", "srt",
        "--output_format", "vtt",
        "--output_format", "json",
        "--language", "en"
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour max
        )

        if result.returncode != 0:
            logger.error(f"Whisper failed: {result.stderr}")
            return None

        # Find generated files
        base_name = video_path.stem
        return {
            "txt": output_dir / f"{base_name}.txt",
            "srt": output_dir / f"{base_name}.srt",
            "vtt": output_dir / f"{base_name}.vtt",
            "json": output_dir / f"{base_name}.json"
        }

    except subprocess.TimeoutExpired:
        logger.error("Whisper timed out")
        return None
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return None


def summarize_with_ollama(transcript_text, model="llama3"):
    """Generate summary using Ollama."""
    try:
        # Prepare the prompt
        prompt = f"""Please provide a comprehensive summary of this meeting transcript. Include key points, decisions made, and action items:

{transcript_text}"""

        # Call Ollama API
        cmd = [
            "curl", "-s", "http://localhost:11434/api/generate",
            "-d", json.dumps({
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3
                }
            })
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes max
        )

        if result.returncode == 0:
            response = json.loads(result.stdout)
            return response.get('response', '')

        return None

    except Exception as e:
        logger.error(f"Ollama summarization failed: {e}")
        return None


def main():
    st.set_page_config(
        page_title="Video Transcriber",
        page_icon="🎥",
        layout="centered"
    )

    st.title("🎥 Video Transcriber")
    st.caption("Transcribe videos with Whisper + Summarize with Ollama")

    # Check if Ollama is running
    ollama_running = check_ollama_running()

    if not ollama_running:
        st.error("❌ Ollama is not running! Please start Ollama first.")
        st.info("Run: `ollama serve` in a terminal")
        return

    # Get available models
    available_models = get_ollama_models()

    if not available_models:
        st.warning("⚠️ No Ollama models found!")
        st.info("Install a model: `ollama pull llama3` or `ollama pull gpt-oss:20b`")
        return

    st.success(f"✅ Ollama is running with {len(available_models)} models")

    # Settings
    st.subheader("Settings")

    col1, col2 = st.columns(2)

    with col1:
        whisper_model = st.selectbox(
            "Whisper Model",
            ["tiny", "base", "small", "medium", "large"],
            index=1,
            help="Larger models are more accurate but slower. 'base' is recommended."
        )

    with col2:
        ollama_model = st.selectbox(
            "Ollama Model",
            available_models,
            help="Select the model for summarization"
        )

    # Video folder
    video_folder = st.text_input(
        "Video Folder",
        value=str(Path.home() / "Documents/GitHub/VideoTranscriber/videos"),
        help="Path to folder containing your videos"
    )

    video_path = Path(video_folder)

    if not video_path.exists():
        st.error(f"Folder not found: {video_folder}")
        return

    # Find videos
    video_files = list(video_path.glob("*.mp4")) + \
                  list(video_path.glob("*.mov")) + \
                  list(video_path.glob("*.avi")) + \
                  list(video_path.glob("*.mkv"))

    if not video_files:
        st.warning(f"No videos found in: {video_folder}")
        st.info("Supported formats: MP4, MOV, AVI, MKV")
        return

    # Select video
    selected_video = st.selectbox(
        "Select Video",
        video_files,
        format_func=lambda x: x.name
    )

    # Process button
    if st.button("🚀 Process Video", type="primary"):
        progress_bar = st.progress(0)
        status = st.empty()

        try:
            # Step 1: Transcribe
            status.text("🎤 Transcribing with Whisper...")
            progress_bar.progress(10)

            output_dir = video_path.parent / "outputs"
            files = transcribe_video(selected_video, whisper_model, output_dir)

            if not files:
                st.error("❌ Transcription failed!")
                return

            progress_bar.progress(50)
            status.text("✓ Transcription complete")

            # Read transcript
            with open(files["txt"], 'r', encoding='utf-8') as f:
                transcript = f.read()

            # Step 2: Summarize
            status.text(f"🤖 Generating summary with {ollama_model}...")
            progress_bar.progress(60)

            summary = summarize_with_ollama(transcript, ollama_model)

            if not summary:
                st.warning("⚠️ Summarization failed, but transcript is available")

            progress_bar.progress(80)

            # Save summary
            if summary:
                summary_file = output_dir / f"{selected_video.stem}_summary.txt"
                with open(summary_file, 'w', encoding='utf-8') as f:
                    f.write(summary)

                # Create full report
                report_file = output_dir / f"{selected_video.stem}_full_report.txt"
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write("=" * 80 + "\n")
                    f.write("VIDEO TRANSCRIPTION REPORT\n")
                    f.write(f"Source: {selected_video.name}\n")
                    f.write(f"Generated: {datetime.now()}\n")
                    f.write("=" * 80 + "\n\n")
                    f.write("SUMMARY\n")
                    f.write("-" * 80 + "\n")
                    f.write(summary + "\n\n")
                    f.write("FULL TRANSCRIPT\n")
                    f.write("-" * 80 + "\n")
                    f.write(transcript)

            progress_bar.progress(100)
            status.text("✅ Processing complete!")

            # Display results
            st.success("✅ Processing complete!")

            tab1, tab2 = st.tabs(["Summary", "Transcript"])

            with tab1:
                if summary:
                    st.markdown("### 📝 Summary")
                    st.write(summary)
                else:
                    st.warning("Summary not available")

            with tab2:
                st.markdown("### 📄 Full Transcript")
                st.text_area("Transcript", transcript, height=400)

            # Downloads
            st.subheader("💾 Download Files")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.download_button(
                    "📄 Transcript (TXT)",
                    data=transcript,
                    file_name=f"{selected_video.stem}.txt",
                    mime="text/plain"
                )

            with col2:
                if summary:
                    st.download_button(
                        "📝 Summary",
                        data=summary,
                        file_name=f"{selected_video.stem}_summary.txt",
                        mime="text/plain"
                    )

            with col3:
                if files.get("srt") and files["srt"].exists():
                    with open(files["srt"], 'r', encoding='utf-8') as f:
                        srt_content = f.read()
                    st.download_button(
                        "📹 Subtitles (SRT)",
                        data=srt_content,
                        file_name=f"{selected_video.stem}.srt",
                        mime="text/plain"
                    )

            # Show output location
            st.info(f"📁 All files saved to: {output_dir}")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            logger.exception("Processing failed")


if __name__ == "__main__":
    main()
