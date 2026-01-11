# Quick Start Guide

Get up and running in 5 minutes.

## Step 1: Install Dependencies

Run the setup script:

```bash
chmod +x setup_mac.sh
./setup_mac.sh
```

This will install:
- ffmpeg (for audio extraction)
- Whisper (for transcription)
- Streamlit (for the UI)

## Step 2: Install and Start Ollama

```bash
# Install Ollama (if not already installed)
brew install ollama

# Start Ollama (keep this terminal open)
ollama serve
```

In another terminal, pull a model:

```bash
# Fast, good quality (recommended)
ollama pull llama3

# OR for better summaries (larger model, slower)
ollama pull gpt-oss:20b
```

## Step 3: Add Your Videos

Place your video files in the `videos` folder:

```bash
# Example
cp ~/Downloads/my-meeting.mp4 ./videos/
```

## Step 4: Run the App

```bash
streamlit run app_simple.py
```

Your browser will open to `http://localhost:8501`

## Step 5: Process a Video

1. Select **Whisper Model**: `base` (recommended)
2. Select **Ollama Model**: Choose from your installed models
3. Select your video from the dropdown
4. Click **Process Video**
5. Wait for processing (may take several minutes)
6. Download your transcript, summary, and subtitles

## Outputs

All files are saved to the `outputs` folder:

- **Transcript** - Full text of the video
- **Summary** - AI-generated summary of key points
- **Subtitles** - SRT format for video players

## Tips

- **Faster processing**: Use `tiny` Whisper model
- **Better quality**: Use `base` or `small` Whisper model
- **Better summaries**: Use `gpt-oss:20b` instead of `llama3`

## Troubleshooting

### "Ollama is not running"
```bash
# In a separate terminal
ollama serve
```

### "No models found"
```bash
ollama pull llama3
```

### "Whisper not found"
```bash
pip3 install -U openai-whisper
```

## Command Line Alternative

If you prefer command line, edit and run:

```bash
./process_local.sh
```

Edit the file to set:
- Your video path
- Output directory
- Ollama model

That's it! You're ready to transcribe videos.
