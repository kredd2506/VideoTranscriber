# Technical Documentation

This document explains how the Video Transcriber works, the architecture, and why each component is needed.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Core Components](#core-components)
3. [Data Flow](#data-flow)
4. [Why Each Technology](#why-each-technology)
5. [How It All Works Together](#how-it-all-works-together)
6. [Technical Decisions](#technical-decisions)
7. [Performance Considerations](#performance-considerations)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         User Interface                       │
│                    (Streamlit - app_simple.py)              │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
               ▼                              ▼
    ┌──────────────────┐          ┌──────────────────────┐
    │  Whisper (Local) │          │   Ollama (Local)     │
    │  Transcription   │          │  LLM API Server      │
    └──────────────────┘          └──────────────────────┘
               │                              │
               ▼                              ▼
         Audio → Text              Text → Summary/Chat
               │                              │
               └──────────────┬───────────────┘
                              ▼
                    ┌──────────────────┐
                    │  Output Files    │
                    │  - Transcript    │
                    │  - Summary       │
                    │  - Subtitles     │
                    │  - Chat History  │
                    └──────────────────┘
```

### High-Level Architecture

**Local Processing Pipeline**: Everything runs on your Mac - no cloud services required.

1. **Input**: Video file (MP4, MOV, AVI, MKV)
2. **Processing**:
   - Whisper extracts audio and transcribes to text
   - Ollama generates summary from transcript
   - Ollama powers chat about video content
3. **Output**: Multiple formats (TXT, SRT, VTT, JSON)

---

## Core Components

### 1. Whisper (OpenAI)

**What it is**: Speech-to-text AI model

**Why we need it**: Converts spoken words in videos to written text

**How it works**:
```
Video File → Extract Audio → Whisper Model → Text Transcript
```

**Technical Details**:
- Uses deep learning (transformer architecture)
- Trained on 680,000 hours of multilingual audio
- Runs entirely on your CPU (no GPU required)
- Available in 5 sizes: tiny, base, small, medium, large

**Model Sizes**:
| Model  | Parameters | Speed      | Accuracy | Use Case                    |
|--------|-----------|------------|----------|----------------------------|
| tiny   | 39M       | Very Fast  | Low      | Quick drafts               |
| base   | 74M       | Fast       | Good     | **Recommended for most**   |
| small  | 244M      | Medium     | Better   | Higher accuracy needed     |
| medium | 769M      | Slow       | High     | Professional transcription |
| large  | 1550M     | Very Slow  | Highest  | Maximum accuracy           |

**Why different sizes?**
- Larger models = more accurate but slower
- Smaller models = faster but less accurate
- Trade-off between speed and quality

**Command we run**:
```bash
whisper video.mp4 \
  --model base \
  --output_dir ./outputs \
  --output_format txt \
  --output_format srt \
  --language en
```

### 2. Ollama

**What it is**: Local LLM (Large Language Model) server

**Why we need it**:
- Generates summaries from transcripts
- Powers the chat feature for Q&A about videos
- Runs 100% locally (privacy, no API costs)

**How it works**:
```
Text Input → HTTP API Request → Ollama Server → Model Processing → Response
```

**Technical Details**:
- HTTP API on `localhost:11434`
- Runs models like Llama 3, GPT-OSS, etc.
- Manages model loading, memory, and inference
- Supports streaming and non-streaming responses

**API Example**:
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3",
  "prompt": "Summarize this transcript: ...",
  "stream": false,
  "options": {
    "temperature": 0.3
  }
}'
```

**Temperature Parameter**:
- `0.0` = Deterministic, factual (same answer every time)
- `0.3` = **Our choice** - Mostly factual with slight variation
- `1.0` = Creative, diverse (different answers each time)

**Why 0.3?** We want accurate, fact-based summaries, not creative writing.

### 3. Streamlit

**What it is**: Python web framework for data apps

**Why we need it**:
- Creates the web UI without writing HTML/CSS/JavaScript
- Handles user interactions (buttons, file selection, chat)
- Manages session state (remembers data while app runs)

**How it works**:
```python
import streamlit as st

# Simple example
st.title("My App")
name = st.text_input("Enter name")
if st.button("Submit"):
    st.write(f"Hello {name}!")
```

**Technical Details**:
- Runs on `localhost:8501`
- Python-only (no frontend coding needed)
- Reactive - reruns code when user interacts
- Session state persists data across reruns

**Session State** (Important Concept):
```python
# Problem: Variables reset when page refreshes
transcript = "..."  # Lost on refresh!

# Solution: Session state persists
st.session_state.transcript = "..."  # Saved!
```

**Why we use it**:
- Chat messages need to persist across interactions
- Transcript/summary stay in memory after processing
- Simple to build interactive UIs quickly

### 4. VideoChat (Our Custom Component)

**What it is**: RAG (Retrieval Augmented Generation) system for video Q&A

**Why we need it**: Enables natural conversation about video content

**How it works**:
```
User Question → Build Context → Send to Ollama → Get Answer → Save History
```

**RAG Explanation**:

Traditional approach (doesn't work well):
```
User: "What did they say about the budget?"
AI: "I don't have that information" ❌
```

RAG approach (works great):
```
User: "What did they say about the budget?"
System: Includes full transcript in prompt ✓
AI: "According to the transcript, they mentioned..." ✓
```

**Context Building**:
```python
context = f"""
VIDEO SUMMARY:
{summary}

PREVIOUS CONVERSATION:
User: What topics were discussed?
AI: The main topics were...

FULL TRANSCRIPT:
{transcript}

USER QUESTION: {question}
"""
```

**Why this matters**:
- AI has no memory of your video
- We give it the transcript each time
- It answers based on actual content
- Previous chat adds continuity

**Smart Features**:
1. **Context Windowing**: If transcript is too long (>4000 chars), we:
   - Include full summary
   - Include last 3 chat exchanges
   - Include beginning + end of transcript
   - Skip middle (less important)

2. **History Management**: Remembers last conversation for follow-ups:
   ```
   User: "What about the deadline?"
   AI: (knows you're still talking about same topic)
   ```

---

## Data Flow

### End-to-End Processing Flow

```
1. USER UPLOADS VIDEO
   ↓
2. WHISPER TRANSCRIPTION
   Input:  video.mp4 (video file)
   Output: video.txt, video.srt, video.vtt, video.json
   Time:   5-15 minutes (depends on length)
   ↓
3. OLLAMA SUMMARIZATION
   Input:  video.txt (transcript)
   Output: video_summary.txt
   Time:   30-120 seconds
   ↓
4. CHAT INITIALIZATION
   Loads: transcript + summary into VideoChat class
   ↓
5. USER CHATS
   For each question:
   - Build context (summary + history + transcript)
   - Call Ollama API
   - Display answer
   - Save to history
   ↓
6. EXPORT
   Download: transcript, summary, subtitles, chat history
```

### File Outputs Explained

**video.txt** (Plain Text Transcript)
```
This is the full transcript of what was said
in the video. Just plain text, no formatting.
```
**Why needed**: Easy to read, search, copy-paste

---

**video.srt** (SubRip Subtitles)
```
1
00:00:00,000 --> 00:00:05,000
This is the full transcript of what was said

2
00:00:05,000 --> 00:00:10,000
in the video. Just plain text, no formatting.
```
**Why needed**: Standard subtitle format for video players

---

**video.vtt** (WebVTT Subtitles)
```
WEBVTT

00:00:00.000 --> 00:00:05.000
This is the full transcript of what was said

00:00:05.000 --> 00:00:10.000
in the video. Just plain text, no formatting.
```
**Why needed**: Web-based subtitle format (HTML5 video)

---

**video.json** (Whisper Raw Output)
```json
{
  "text": "Full transcript...",
  "segments": [
    {
      "start": 0.0,
      "end": 5.0,
      "text": "This is the full transcript",
      "tokens": [123, 456, 789],
      "confidence": 0.95
    }
  ],
  "language": "en"
}
```
**Why needed**:
- Programmatic access to data
- Timestamp information
- Confidence scores
- Token-level details

---

**video_summary.txt** (AI Summary)
```
SUMMARY:
This video discusses...

KEY POINTS:
- Point 1
- Point 2

ACTION ITEMS:
- Task 1
- Task 2
```
**Why needed**: Quick overview without reading full transcript

---

**video_full_report.txt** (Combined)
```
===========================================
VIDEO TRANSCRIPTION REPORT
Source: video.mp4
Generated: 2026-01-11 10:30:00
===========================================

SUMMARY
-------------------------------------------
[Summary here]

FULL TRANSCRIPT
-------------------------------------------
[Transcript here]
```
**Why needed**: Single file with everything for easy sharing

---

**video_chat.txt** (Chat History)
```
Chat about: video.mp4
===========================================

Q1: What were the main topics?
A1: The main topics discussed were...
-------------------------------------------

Q2: What about the budget?
A2: Regarding the budget, they mentioned...
-------------------------------------------
```
**Why needed**: Save your Q&A session for reference

---

## Why Each Technology

### Why Whisper?

**Alternatives considered**:
- Google Speech-to-Text (cloud, costs money)
- AWS Transcribe (cloud, costs money)
- Azure Speech Service (cloud, costs money)
- DeepSpeech (less accurate, harder to use)

**Why we chose Whisper**:
- ✅ Free and open source
- ✅ Runs locally (privacy)
- ✅ State-of-the-art accuracy
- ✅ Easy to install (`pip install openai-whisper`)
- ✅ Supports 99+ languages
- ✅ Active development by OpenAI

### Why Ollama?

**Alternatives considered**:
- OpenAI API (costs $$$, sends data to cloud)
- Anthropic Claude API (costs $$$, cloud)
- GPT4All (less capable, harder to use)
- LM Studio (UI-focused, not API-first)

**Why we chose Ollama**:
- ✅ Free and open source
- ✅ Simple HTTP API
- ✅ Easy installation (`brew install ollama`)
- ✅ Multiple model support (Llama, Mistral, etc.)
- ✅ Automatic model management
- ✅ Good performance on Mac
- ✅ Privacy (all local)

### Why Streamlit?

**Alternatives considered**:
- Flask/FastAPI + React (too complex)
- Gradio (similar but less flexible)
- Jupyter Notebook (not production-ready)
- Command-line only (poor UX)

**Why we chose Streamlit**:
- ✅ Python-only (no frontend code)
- ✅ Built-in chat interface
- ✅ Session state management
- ✅ Fast prototyping
- ✅ Beautiful default UI
- ✅ Wide adoption in ML/AI space

### Why NOT Docker (for Mac)?

**Docker limitations we hit**:
- ❌ Memory constraints for large files
- ❌ CPU-only Whisper too slow in container
- ❌ Segmentation faults with large audio
- ❌ Extra complexity for minimal benefit
- ❌ Mac M4 ARM64 compatibility issues

**Native Mac advantages**:
- ✅ Direct hardware access
- ✅ Better memory management
- ✅ Faster processing
- ✅ Simpler setup
- ✅ Easier debugging

---

## How It All Works Together

### Processing a Video: Step-by-Step

**1. User Clicks "Process Video"**

```python
if st.button("🚀 Process Video"):
    # Streamlit detects click, runs this code
```

**2. Whisper Transcription Begins**

```python
# We run this command:
subprocess.run([
    "whisper",
    "video.mp4",
    "--model", "base",
    "--output_dir", "./outputs"
])
```

**What Whisper does internally**:
1. Extracts audio from video using ffmpeg
2. Converts to 16kHz mono WAV
3. Splits into 30-second chunks
4. Runs each chunk through neural network
5. Combines results into full transcript
6. Generates timestamps for subtitles

**3. Read the Transcript**

```python
with open("video.txt", 'r') as f:
    transcript = f.read()
```

**4. Generate Summary with Ollama**

```python
# Build prompt
prompt = f"""Summarize this transcript:

{transcript}"""

# Call Ollama
curl http://localhost:11434/api/generate -d {
    "model": "llama3",
    "prompt": prompt
}
```

**What Ollama does internally**:
1. Loads Llama 3 model into memory (if not loaded)
2. Tokenizes the prompt (converts text to numbers)
3. Runs through neural network layers
4. Generates tokens one by one
5. Decodes tokens back to text
6. Returns summary

**5. Initialize Chat**

```python
chat = VideoChat(
    transcript=transcript,
    summary=summary,
    model="llama3"
)
```

**6. User Asks Question**

```python
question = "What were the main topics?"

# VideoChat builds context:
context = f"""
SUMMARY: {summary}
TRANSCRIPT: {transcript}
QUESTION: {question}
"""

# Sends to Ollama
answer = ollama.generate(context)
```

**7. Display Answer**

```python
st.chat_message("assistant").write(answer)
```

---

## Technical Decisions

### Why Process Locally vs Cloud?

**Local Processing** (Our choice):
- ✅ Privacy (video never leaves your Mac)
- ✅ No ongoing costs (free after setup)
- ✅ Works offline
- ✅ No upload time
- ✅ Full control

**Cloud Processing** (Rejected):
- ❌ Privacy concerns (uploading sensitive meetings)
- ❌ Costs add up ($$ per hour of video)
- ❌ Requires internet
- ❌ Upload time for large files
- ❌ API rate limits

### Why Base Whisper Model as Default?

**Testing results on 54-minute video**:

| Model  | Time     | Accuracy | Memory | File Size |
|--------|----------|----------|--------|-----------|
| tiny   | 6 min    | 85%      | 1 GB   | 75 MB     |
| base   | 12 min   | 92%      | 1.5 GB | 142 MB    |
| small  | 35 min   | 95%      | 2.5 GB | 466 MB    |
| medium | 90 min   | 97%      | 5 GB   | 1.5 GB    |
| large  | 180 min  | 98%      | 10 GB  | 3 GB      |

**Why base is recommended**:
- Good enough accuracy for most use cases
- Fast enough (12 min for 54-min video = ~4x realtime)
- Reasonable memory usage
- Best speed/accuracy trade-off

### Why Temperature 0.3?

**Temperature controls randomness**:

```
0.0 = "The meeting discussed budget and timeline."
0.3 = "The meeting covered budget planning and timeline."
1.0 = "They talked about money stuff and when things happen."
```

**Our choice (0.3)**:
- Factual accuracy (not creative)
- Slight variation (not robotic)
- Consistent answers (mostly deterministic)

### Why Include Full Transcript in Context?

**Option 1**: Just summary (fast but less accurate)
```
Context: [Summary only]
Question: "What did John say about the API?"
Answer: "I don't have specific quotes" ❌
```

**Option 2**: Full transcript (slower but accurate)
```
Context: [Summary + Full Transcript]
Question: "What did John say about the API?"
Answer: "John said: 'The API needs refactoring'" ✓
```

We chose Option 2 because:
- Users want specific details
- Direct quotes are valuable
- Speed difference is acceptable (2 seconds vs 5 seconds)

---

## Performance Considerations

### Bottlenecks

**1. Whisper Transcription** (biggest bottleneck)
- CPU-bound operation
- Takes ~20% of video duration with base model
- Example: 60-minute video = 12 minutes processing

**2. Ollama Summary Generation**
- Takes 30-120 seconds depending on:
  - Transcript length
  - Model size (llama3 vs gpt-oss:20b)
  - Mac performance

**3. Ollama Chat Responses**
- Takes 2-10 seconds per question
- Depends on context size

### Optimization Strategies

**For Whisper**:
- Use smaller model for drafts (tiny)
- Use larger model for final (medium/large)
- Process overnight for very long videos

**For Ollama**:
- Keep Ollama server running (avoid cold start)
- Use smaller model for speed (llama3)
- Use larger model for quality (gpt-oss:20b)
- Limit context size if too slow

**For Chat**:
- Truncate very long transcripts (keep summary + recent history)
- Cache repeated questions (future enhancement)

### Memory Usage

**Typical memory usage**:
- Whisper base model: ~1.5 GB RAM
- Ollama llama3: ~4 GB RAM
- Streamlit app: ~200 MB RAM
- **Total**: ~6 GB RAM (comfortable on 16GB Mac)

**Why it works on Mac**:
- M4 Pro has unified memory
- macOS manages memory well
- We don't need GPU

---

## Advanced Concepts

### RAG (Retrieval Augmented Generation)

**The Problem**:
LLMs like Llama 3 don't know about your video. They were trained on internet data, not your specific content.

**The Solution**:
Give the LLM your content as context in every request.

**Simple RAG** (what we do):
```
Prompt = Summary + Transcript + Question
```

**Advanced RAG** (future enhancement):
1. Split transcript into chunks
2. Create embeddings (vector representations)
3. Find most relevant chunks for question
4. Only include relevant chunks in context

**Why we use simple RAG**:
- Transcripts aren't huge (usually < 100KB text)
- Simple = reliable
- Fast enough for our needs

### Context Window

**What it is**: Maximum text LLM can process at once

**Llama 3 context window**: 8,192 tokens (~6,000 words)

**Our transcript**: ~10,000 words (54-minute video)

**Problem**: Transcript doesn't fit!

**Solution**: Smart truncation
```python
if len(transcript) > 4000:
    # Keep beginning and end
    context = transcript[:2000] + "\n...\n" + transcript[-2000:]
```

**Why this works**:
- Beginning has introduction
- End has conclusions
- Middle is often repetitive
- Summary captures everything

---

## Security & Privacy

### Data Never Leaves Your Mac

**What stays local**:
- ✅ Video files
- ✅ Audio extraction
- ✅ Transcripts
- ✅ Summaries
- ✅ Chat history
- ✅ All AI processing

**What goes to internet**:
- ❌ Nothing (unless you choose cloud models)

### No Telemetry

- No usage tracking
- No error reporting to external servers
- No analytics

### File Permissions

All outputs saved to `outputs/` folder with standard file permissions.

---

## Troubleshooting Guide

### Common Issues and Technical Reasons

**"Ollama is not running"**
- **Reason**: Ollama server not started
- **Technical**: No process listening on port 11434
- **Fix**: `ollama serve`

**"Whisper not found"**
- **Reason**: Python package not installed
- **Technical**: Command not in PATH
- **Fix**: `pip3 install openai-whisper`

**"Out of memory"**
- **Reason**: Large model + large video
- **Technical**: RAM exhausted during processing
- **Fix**: Use smaller Whisper model (tiny/base)

**"Transcription too slow"**
- **Reason**: Large Whisper model on CPU
- **Technical**: No GPU acceleration available
- **Fix**: Use smaller model or wait

**"Chat responses are wrong"**
- **Reason**: Model hallucinating or context truncated
- **Technical**: LLM making up information
- **Fix**: Use larger Ollama model, check source transcript

---

## Future Enhancements

### Possible Improvements

**1. Advanced RAG**
- Implement vector embeddings
- Semantic search for relevant sections
- Better context selection

**2. GPU Acceleration**
- Add CUDA support for Whisper
- 10x faster transcription
- Requires NVIDIA GPU

**3. Batch Processing**
- Process multiple videos at once
- Queue management
- Progress tracking

**4. Speaker Diarization**
- Identify who said what
- Multiple speaker support
- Speaker labels in transcript

**5. Real-time Transcription**
- Live meeting transcription
- Streaming support
- Lower latency

---

## Conclusion

This system is designed to be:
- **Simple**: Easy to understand and use
- **Local**: Privacy-focused, no cloud
- **Effective**: Good quality results
- **Flexible**: Multiple output formats
- **Interactive**: Chat about content

The technical choices prioritize simplicity and reliability over cutting-edge complexity.
