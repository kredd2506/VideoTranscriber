"""
Progress tracking that persists across Streamlit reruns and crashes.
"""
import json
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

PROGRESS_FILE = Path("/app/data/outputs/.progress.json")


def save_progress(status, progress_percent=0, error=None):
    """Save progress to file that persists across crashes."""
    try:
        data = {
            "status": status,
            "progress": progress_percent,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Progress saved: {status} ({progress_percent}%)")
    except Exception as e:
        logger.warning(f"Failed to save progress: {e}")


def load_progress():
    """Load progress from file."""
    try:
        if PROGRESS_FILE.exists():
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load progress: {e}")
    return None


def clear_progress():
    """Clear progress file."""
    try:
        if PROGRESS_FILE.exists():
            PROGRESS_FILE.unlink()
            logger.info("Progress cleared")
    except Exception as e:
        logger.warning(f"Failed to clear progress: {e}")
