#!/bin/bash
# Process video locally on Mac (outside Docker) then use Ollama for summarization

set -e

VIDEO_FILE="/Users/manishreddy/Documents/GitHub/VideoTranscriber/videos/Call with Andy K-20251019_060054-Meeting Recording.mp4"
OUTPUT_DIR="/Users/manishreddy/Documents/GitHub/VideoTranscriber/outputs"
OLLAMA_MODEL="gpt-oss:20b"

echo "=========================================="
echo "Local Video Processing Script"
echo "=========================================="
echo "Video: $VIDEO_FILE"
echo "Output: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Check if whisper is installed
if ! command -v whisper &> /dev/null; then
    echo "❌ Whisper not installed. Installing..."
    pip3 install -U openai-whisper
fi

# Step 1: Transcribe with Whisper (native Mac - will work!)
echo "Step 1/3: Transcribing with Whisper..."
whisper "$VIDEO_FILE" \
    --model base \
    --output_dir "$OUTPUT_DIR" \
    --output_format txt \
    --output_format srt \
    --output_format vtt \
    --output_format json \
    --language en

BASE_NAME=$(basename "$VIDEO_FILE" .mp4)
TRANSCRIPT_FILE="$OUTPUT_DIR/${BASE_NAME}.txt"

if [ ! -f "$TRANSCRIPT_FILE" ]; then
    echo "❌ Transcription failed!"
    exit 1
fi

echo "✓ Transcription complete!"
echo ""

# Step 2: Summarize with Ollama
echo "Step 2/3: Generating summary with Ollama $OLLAMA_MODEL..."

TRANSCRIPT=$(cat "$TRANSCRIPT_FILE")
SUMMARY_FILE="$OUTPUT_DIR/${BASE_NAME}_summary.txt"

curl -s http://localhost:11434/api/generate -d "{
  \"model\": \"$OLLAMA_MODEL\",
  \"prompt\": \"Please provide a comprehensive summary of this meeting transcript. Include key points, decisions made, and action items:\n\n$TRANSCRIPT\",
  \"stream\": false,
  \"options\": {
    \"temperature\": 0.3
  }
}" | python3 -c "import sys, json; print(json.load(sys.stdin)['response'])" > "$SUMMARY_FILE"

echo "✓ Summary complete!"
echo ""

# Step 3: Create combined report
echo "Step 3/3: Creating full report..."

REPORT_FILE="$OUTPUT_DIR/${BASE_NAME}_full_report.txt"

cat > "$REPORT_FILE" <<EOF
================================================================================
VIDEO TRANSCRIPTION REPORT
Source: $(basename "$VIDEO_FILE")
Generated: $(date)
================================================================================

SUMMARY
--------------------------------------------------------------------------------
$(cat "$SUMMARY_FILE")

FULL TRANSCRIPT
--------------------------------------------------------------------------------
$TRANSCRIPT
EOF

echo "✓ Report complete!"
echo ""
echo "=========================================="
echo "✅ PROCESSING COMPLETE!"
echo "=========================================="
echo "Output files in: $OUTPUT_DIR"
echo "  - ${BASE_NAME}.txt (transcript)"
echo "  - ${BASE_NAME}.srt (subtitles)"
echo "  - ${BASE_NAME}.vtt (web subtitles)"
echo "  - ${BASE_NAME}_summary.txt (AI summary)"
echo "  - ${BASE_NAME}_full_report.txt (combined)"
echo "=========================================="
