# Video Transcriber

Simple video transcription tool using Whisper (local) + Ollama (local AI) for summarization and chat.

**Supported Formats**: MP4, MOV, AVI, MKV

## What This Does

1. **Transcribes** your videos using OpenAI's Whisper (runs locally on your Mac)
2. **Summarizes** content using Ollama (runs locally - no cloud needed)
3. **Chat** with AI about your video - ask questions, get insights, explore topics
4. **Outputs**: transcript (TXT), subtitles (SRT, VTT), AI summary, and chat history

## Quick Start (Mac)

### Prerequisites

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install ffmpeg
brew install ffmpeg

# Install Python dependencies
pip3 install openai-whisper streamlit
```

### Install Ollama

```bash
# Download and install Ollama from https://ollama.ai
# Or use homebrew:
brew install ollama

# Start Ollama
ollama serve

# Pull a model (in another terminal)
ollama pull llama3
# OR for better summaries (larger model):
ollama pull gpt-oss:20b
```

### Run the App

```bash
# Clone this repo
git clone https://github.com/DataAnts-AI/VideoTranscriber.git
cd VideoTranscriber

# Run the simple UI
streamlit run app_simple.py
```

Then open your browser to `http://localhost:8501`

## Using the Simple UI

### Processing Videos

1. **Select Whisper Model**: Choose transcription quality
   - `tiny` - Fastest, least accurate
   - `base` - **Recommended** - Good balance
   - `small` - Better accuracy, slower
   - `medium` - High accuracy, slow
   - `large` - Best accuracy, very slow

2. **Select Ollama Model**: Choose your installed model
   - `llama3` - Fast, good quality
   - `gpt-oss:20b` - Slower, better summaries and chat

3. **Video Folder**: Point to where your videos are

4. **Click Process**: Wait for transcription and summary

### Chatting About Your Video

After processing, you can chat with AI about your video:

- **Ask Questions**: "What were the main topics discussed?"
- **Get Details**: "What did they say about the budget?"
- **Find Information**: "When did they mention the deadline?"
- **Explore Topics**: "Tell me more about the technical approach"

The AI has access to the full transcript and will answer based on the video content.

### Downloads

Get your transcript, summary, subtitles, and chat history from the Downloads tab.

## Command Line Usage

If you prefer command line, use the included script:

```bash
./process_local.sh
```

Edit the script to set:
- `VIDEO_FILE`: Path to your video
- `OUTPUT_DIR`: Where to save results
- `OLLAMA_MODEL`: Which model to use

## Outputs

All files are saved to the `outputs` folder:

- `{video_name}.txt` - Plain text transcript
- `{video_name}.srt` - Subtitle file (for video players)
- `{video_name}.vtt` - Web subtitle format
- `{video_name}.json` - Full Whisper output with timestamps
- `{video_name}_summary.txt` - AI-generated summary
- `{video_name}_full_report.txt` - Combined transcript + summary

## Tips

- **Large videos**: Use `tiny` or `base` Whisper model to save time
- **Best quality**: Use `medium` or `large` Whisper model (much slower)
- **Better summaries**: Use `gpt-oss:20b` instead of `llama3` (takes longer)
- **Mac only**: This setup is optimized for Mac. For Windows/Linux, see advanced setup below.

## Troubleshooting

### Whisper not found
```bash
pip3 install -U openai-whisper
```

### Ollama connection error
```bash
# Make sure Ollama is running
ollama serve
```

### Out of memory
- Use smaller Whisper model (`tiny` or `base`)
- Close other applications
- Restart your computer

## Advanced Setup (Docker - Optional)

If you want to run in Docker (not recommended for Mac):

```bash
# Build and run
docker-compose up -d

# Access at http://localhost:8501
```

**Note**: Docker setup has memory limitations. Native Mac setup (above) works better.

## Technical Documentation

Want to understand how everything works under the hood?

Read **[TECHNICAL.md](TECHNICAL.md)** for detailed explanations of:
- System architecture and data flow
- Why we chose each technology
- How Whisper transcription works
- How Ollama chat works
- Performance optimization tips
- Security and privacy details

Perfect for developers who want to customize or extend the system.

## Support

For help with AI solutions, workflows, or custom implementations:
- Email: support@dataants.org
- Issues: https://github.com/DataAnts-AI/VideoTranscriber/issues

---

## Advanced Features

The original [app.py](app.py) includes additional features like:
- Speaker diarization (identify different speakers)
- Translation to multiple languages
- Keyword extraction
- GPU acceleration
- Docker deployment

These require additional setup. See [INSTALLATION.md](INSTALLATION.md) for details.

**For most users, the simple version ([app_simple.py](app_simple.py)) is recommended.**
