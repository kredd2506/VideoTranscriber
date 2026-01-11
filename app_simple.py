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
from utils.video_chat import VideoChat, check_ollama_available
from utils.video_analysis import run_analysis

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
    return check_ollama_available()


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
        layout="wide"
    )

    st.title("🎥 Video Transcriber")
    st.caption("Transcribe videos with Whisper + Chat with Ollama")

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

    # Initialize session state
    if 'transcript' not in st.session_state:
        st.session_state.transcript = None
    if 'summary' not in st.session_state:
        st.session_state.summary = None
    if 'video_name' not in st.session_state:
        st.session_state.video_name = None
    if 'chat' not in st.session_state:
        st.session_state.chat = None
    if 'files' not in st.session_state:
        st.session_state.files = None
    if 'analysis_report' not in st.session_state:
        st.session_state.analysis_report = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None

    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")

        whisper_model = st.selectbox(
            "Whisper Model",
            ["tiny", "base", "small", "medium", "large"],
            index=1,
            help="Larger models are more accurate but slower. 'base' is recommended."
        )

        ollama_model = st.selectbox(
            "Ollama Model",
            available_models,
            help="Select the model for summarization and chat"
        )

        # Video folder
        video_folder = st.text_input(
            "Video Folder",
            value=str(Path.home() / "Documents/GitHub/VideoTranscriber/videos"),
            help="Path to folder containing your videos"
        )

        st.divider()

        # Export chat option
        if st.session_state.chat and st.session_state.chat.get_history():
            if st.button("💾 Export Chat History"):
                output_dir = Path(video_folder).parent / "outputs"
                chat_file = output_dir / f"{st.session_state.video_name}_chat.txt"
                if st.session_state.chat.export_chat(chat_file):
                    st.success(f"Chat saved to {chat_file.name}")

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

            # Save to session state
            st.session_state.transcript = transcript
            st.session_state.summary = summary
            st.session_state.video_name = selected_video.stem
            st.session_state.files = files

            # Initialize chat
            st.session_state.chat = VideoChat(
                transcript=transcript,
                video_name=selected_video.stem,
                model=ollama_model,
                summary=summary
            )

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

            st.success("✅ Processing complete! You can now chat about the video below.")
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            logger.exception("Processing failed")

    # Display results if available
    if st.session_state.transcript:
        st.divider()

        # Add analysis button
        col_analyze1, col_analyze2 = st.columns([3, 1])
        with col_analyze1:
            st.markdown("### 📊 Deep Analysis Available")
            st.caption("Run comprehensive AI analysis with 5 Whys, goal identification, and strategic insights")
        with col_analyze2:
            if st.button("🔍 Run Deep Analysis", type="secondary"):
                with st.spinner("Running comprehensive analysis..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def progress_callback(pct, msg):
                        progress_bar.progress(pct)
                        status_text.text(msg)

                    output_dir = Path(video_folder).parent / "outputs"
                    analysis_data = run_analysis(
                        st.session_state.transcript,
                        st.session_state.video_name,
                        ollama_model,
                        output_dir,
                        progress_callback
                    )

                    st.session_state.analysis_report = analysis_data['report_text']
                    st.session_state.analysis_results = analysis_data['results']

                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    st.success(f"Analysis saved to: {analysis_data['report_path'].name}")
                    st.rerun()

        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Chat", "📝 Summary", "📄 Transcript", "🔍 Deep Analysis", "💾 Downloads"])

        with tab1:
            st.subheader("💬 Chat About This Video")
            st.caption(f"Ask questions about: {st.session_state.video_name}")

            # Chat interface
            if 'chat_messages' not in st.session_state:
                st.session_state.chat_messages = []

            # Display chat history
            for msg in st.session_state.chat_messages:
                with st.chat_message("user"):
                    st.write(msg["question"])
                with st.chat_message("assistant"):
                    st.write(msg["answer"])

            # Chat input
            if question := st.chat_input("Ask a question about the video..."):
                # Display user message
                with st.chat_message("user"):
                    st.write(question)

                # Get AI response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        answer = st.session_state.chat.ask(question)
                        st.write(answer)

                # Save to session state
                st.session_state.chat_messages.append({
                    "question": question,
                    "answer": answer
                })

            # Clear chat button
            if st.session_state.chat_messages:
                if st.button("🗑️ Clear Chat History"):
                    st.session_state.chat_messages = []
                    st.session_state.chat.clear_history()
                    st.rerun()

        with tab2:
            if st.session_state.summary:
                st.markdown("### 📝 Summary")
                st.write(st.session_state.summary)
            else:
                st.warning("Summary not available")

        with tab3:
            st.markdown("### 📄 Full Transcript")
            st.text_area("Transcript", st.session_state.transcript, height=400)

        with tab4:
            st.subheader("🔍 Deep Analysis Report")

            if st.session_state.analysis_report:
                # Display analysis results in expandable sections
                if st.session_state.analysis_results:
                    st.markdown("#### Analysis Components")

                    for analysis in st.session_state.analysis_results['analyses']:
                        with st.expander(f"📌 {analysis['agent']}", expanded=False):
                            st.markdown(analysis['analysis'])

                    st.divider()

                # Full report
                st.markdown("#### Complete Report")
                st.text_area(
                    "Full Analysis Report",
                    st.session_state.analysis_report,
                    height=500,
                    help="Complete analysis report with all sections"
                )

                # Download button
                st.download_button(
                    "⬇️ Download Analysis Report",
                    data=st.session_state.analysis_report,
                    file_name=f"{st.session_state.video_name}_analysis_report.txt",
                    mime="text/plain"
                )
            else:
                st.info("👆 Click 'Run Deep Analysis' above to generate a comprehensive analysis report")
                st.markdown("""
                **The Deep Analysis includes:**

                - 🎯 **Theme Analysis** - Main themes and topics identified
                - 🎯 **Goal Identification** - Stated and implied objectives
                - 🔄 **5 Whys Analysis** - Root cause analysis of main issues
                - 👥 **Stakeholder Analysis** - Key players and their perspectives
                - ✅ **Action Items** - Decisions, tasks, and next steps
                - 💡 **Strategic Insights** - Recommendations and key observations

                This multi-agent analysis provides deep understanding of the video content.
                """)

        with tab5:
            st.subheader("💾 Download Files")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.download_button(
                    "📄 Transcript (TXT)",
                    data=st.session_state.transcript,
                    file_name=f"{st.session_state.video_name}.txt",
                    mime="text/plain"
                )

            with col2:
                if st.session_state.summary:
                    st.download_button(
                        "📝 Summary",
                        data=st.session_state.summary,
                        file_name=f"{st.session_state.video_name}_summary.txt",
                        mime="text/plain"
                    )

            with col3:
                if st.session_state.files and st.session_state.files.get("srt"):
                    srt_file = st.session_state.files["srt"]
                    if srt_file.exists():
                        with open(srt_file, 'r', encoding='utf-8') as f:
                            srt_content = f.read()
                        st.download_button(
                            "📹 Subtitles (SRT)",
                            data=srt_content,
                            file_name=f"{st.session_state.video_name}.srt",
                            mime="text/plain"
                        )

            # Show output location
            output_dir = video_path.parent / "outputs"
            st.info(f"📁 All files saved to: {output_dir}")


if __name__ == "__main__":
    main()
