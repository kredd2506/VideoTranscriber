#!/bin/bash
# Simple setup script for Mac

set -e

echo "=========================================="
echo "Video Transcriber Setup for Mac"
echo "=========================================="
echo ""

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew not found. Installing..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "✓ Homebrew found"
fi

# Install ffmpeg
echo ""
echo "Checking ffmpeg..."
if ! command -v ffmpeg &> /dev/null; then
    echo "Installing ffmpeg..."
    brew install ffmpeg
else
    echo "✓ ffmpeg found"
fi

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip3 install -U openai-whisper streamlit

echo ""
echo "Checking Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama not found"
    echo ""
    echo "Please install Ollama manually:"
    echo "  1. Visit https://ollama.ai"
    echo "  2. Download and install Ollama for Mac"
    echo "  3. Run: ollama serve"
    echo "  4. Run: ollama pull llama3"
    echo ""
    echo "Or install with Homebrew:"
    echo "  brew install ollama"
else
    echo "✓ Ollama found"

    # Check if Ollama is running
    if curl -s http://localhost:11434/api/tags &> /dev/null; then
        echo "✓ Ollama is running"

        # List available models
        echo ""
        echo "Available Ollama models:"
        ollama list

        # Suggest pulling models if none found
        if ! ollama list | grep -q "llama3"; then
            echo ""
            echo "💡 Recommended: Pull llama3 model"
            echo "   Run: ollama pull llama3"
        fi
    else
        echo "⚠️ Ollama is installed but not running"
        echo "   Start it with: ollama serve"
    fi
fi

# Create directories
echo ""
echo "Creating directories..."
mkdir -p videos
mkdir -p outputs
mkdir -p cache

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. If Ollama is not running, start it:"
echo "     ollama serve"
echo ""
echo "  2. Pull an Ollama model (if not already done):"
echo "     ollama pull llama3"
echo "     OR for better summaries:"
echo "     ollama pull gpt-oss:20b"
echo ""
echo "  3. Place your videos in the 'videos' folder"
echo ""
echo "  4. Run the app:"
echo "     streamlit run app_simple.py"
echo ""
echo "  5. Open your browser to:"
echo "     http://localhost:8501"
echo ""
echo "=========================================="
