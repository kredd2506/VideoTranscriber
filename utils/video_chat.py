#!/usr/bin/env python3
"""
Video Chat Module - RAG-based conversation about video transcripts
Uses Ollama for local AI chat about video content
"""

import json
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class VideoChat:
    """Chat interface for asking questions about video transcripts."""

    def __init__(self, transcript: str, video_name: str, model: str = "llama3", summary: str = None):
        """
        Initialize video chat.

        Args:
            transcript: Full video transcript text
            video_name: Name of the video for context
            model: Ollama model to use
            summary: Optional summary of the video
        """
        self.transcript = transcript
        self.video_name = video_name
        self.model = model
        self.summary = summary
        self.chat_history: List[Dict[str, str]] = []

    def _build_context(self, question: str, max_context_length: int = 4000) -> str:
        """
        Build context for the question by finding relevant parts of transcript.

        Args:
            question: User's question
            max_context_length: Maximum characters to include in context

        Returns:
            Relevant context string
        """
        # For now, use simple approach: include summary (if available) and full transcript
        # In a more advanced version, we could use embeddings to find most relevant sections

        context_parts = []

        if self.summary:
            context_parts.append(f"VIDEO SUMMARY:\n{self.summary}\n")

        # Include recent chat history for context
        if self.chat_history:
            recent_history = self.chat_history[-3:]  # Last 3 exchanges
            history_text = "\n".join([
                f"User: {msg['question']}\nAssistant: {msg['answer']}"
                for msg in recent_history
            ])
            context_parts.append(f"PREVIOUS CONVERSATION:\n{history_text}\n")

        # Add transcript (truncate if too long)
        transcript_preview = self.transcript
        if len(transcript_preview) > max_context_length:
            # Take beginning and end of transcript
            half = max_context_length // 2
            transcript_preview = (
                transcript_preview[:half] +
                "\n\n[... middle section omitted for brevity ...]\n\n" +
                transcript_preview[-half:]
            )

        context_parts.append(f"FULL TRANSCRIPT:\n{transcript_preview}")

        return "\n\n".join(context_parts)

    def _call_ollama(self, prompt: str, temperature: float = 0.3) -> Optional[str]:
        """
        Call Ollama API with prompt.

        Args:
            prompt: The prompt to send
            temperature: Sampling temperature

        Returns:
            Response text or None if failed
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            }

            cmd = [
                "curl", "-s", "http://localhost:11434/api/generate",
                "-d", json.dumps(payload)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # 2 minutes max
            )

            if result.returncode == 0:
                response = json.loads(result.stdout)
                return response.get('response', '').strip()

            logger.error(f"Ollama API call failed: {result.stderr}")
            return None

        except subprocess.TimeoutExpired:
            logger.error("Ollama API call timed out")
            return None
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return None

    def ask(self, question: str) -> str:
        """
        Ask a question about the video.

        Args:
            question: User's question

        Returns:
            Answer from the AI
        """
        # Build context
        context = self._build_context(question)

        # Build prompt
        prompt = f"""You are a helpful AI assistant answering questions about a video titled "{self.video_name}".

You have access to the video's transcript and summary. Answer the user's question based ONLY on the information provided in the transcript. If the answer isn't in the transcript, say so.

Be concise, accurate, and helpful. Use specific quotes from the transcript when relevant.

{context}

USER QUESTION: {question}

ANSWER:"""

        # Get response from Ollama
        answer = self._call_ollama(prompt)

        if not answer:
            answer = "Sorry, I couldn't generate a response. Please make sure Ollama is running."

        # Save to history
        self.chat_history.append({
            "question": question,
            "answer": answer
        })

        return answer

    def get_history(self) -> List[Dict[str, str]]:
        """Get chat history."""
        return self.chat_history

    def clear_history(self):
        """Clear chat history."""
        self.chat_history = []

    def export_chat(self, output_path: Path):
        """
        Export chat history to a file.

        Args:
            output_path: Where to save the chat
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Chat about: {self.video_name}\n")
                f.write("=" * 80 + "\n\n")

                for i, msg in enumerate(self.chat_history, 1):
                    f.write(f"Q{i}: {msg['question']}\n")
                    f.write(f"A{i}: {msg['answer']}\n")
                    f.write("-" * 80 + "\n\n")

            logger.info(f"Chat exported to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export chat: {e}")
            return False


def check_ollama_available() -> bool:
    """Check if Ollama is running and available."""
    try:
        result = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/tags"],
            capture_output=True,
            timeout=2
        )
        return result.returncode == 0
    except:
        return False


def get_available_models() -> List[str]:
    """Get list of available Ollama models."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            models = [line.split()[0] for line in lines if line.strip()]
            return models
        return []
    except Exception as e:
        logger.error(f"Failed to get Ollama models: {e}")
        return []
