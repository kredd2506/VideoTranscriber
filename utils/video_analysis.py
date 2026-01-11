#!/usr/bin/env python3
"""
Advanced Video Analysis Module
Multi-agent system for deep content analysis using 5 Whys and AI agents
"""

import json
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class OllamaAgent:
    """Base class for Ollama-powered analysis agents."""

    def __init__(self, model: str = "llama3", temperature: float = 0.3):
        self.model = model
        self.temperature = temperature

    def _call_ollama(self, prompt: str, temperature: float = None) -> Optional[str]:
        """Call Ollama API with prompt."""
        if temperature is None:
            temperature = self.temperature

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
                timeout=180  # 3 minutes max
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


class ThemeAnalysisAgent(OllamaAgent):
    """Agent for identifying main themes and topics."""

    def analyze(self, transcript: str) -> Dict[str, any]:
        """Identify main themes and topics in the transcript."""

        prompt = f"""Analyze this transcript and identify the main themes and topics discussed.

TRANSCRIPT:
{transcript}

Please provide:
1. Main Themes (3-5 overarching themes)
2. Key Topics (5-10 specific topics discussed)
3. Topic Distribution (which topics got most attention)
4. Recurring Patterns (concepts or ideas mentioned repeatedly)

Format your response as structured sections."""

        response = self._call_ollama(prompt)

        return {
            "agent": "Theme Analysis",
            "analysis": response,
            "timestamp": datetime.now().isoformat()
        }


class GoalIdentificationAgent(OllamaAgent):
    """Agent for identifying goals, objectives, and desired outcomes."""

    def analyze(self, transcript: str) -> Dict[str, any]:
        """Identify the goals and objectives discussed."""

        prompt = f"""Analyze this transcript to identify the goals, objectives, and desired outcomes.

TRANSCRIPT:
{transcript}

Please identify:
1. Stated Goals (explicitly mentioned objectives)
2. Implied Goals (objectives suggested but not directly stated)
3. Success Criteria (how they plan to measure success)
4. Constraints (limitations or challenges mentioned)
5. Priorities (what seems most important)

Format your response with clear sections."""

        response = self._call_ollama(prompt)

        return {
            "agent": "Goal Identification",
            "analysis": response,
            "timestamp": datetime.now().isoformat()
        }


class FiveWhysAgent(OllamaAgent):
    """Agent for performing 5 Whys root cause analysis."""

    def analyze(self, transcript: str, initial_statement: str = None) -> Dict[str, any]:
        """Perform 5 Whys analysis on the video content."""

        if initial_statement is None:
            # First, identify the main problem/topic
            identify_prompt = f"""Based on this transcript, identify the main problem, challenge, or central topic being discussed.

TRANSCRIPT:
{transcript}

Provide a single clear statement of the main issue or topic."""

            initial_statement = self._call_ollama(identify_prompt)

        # Now perform 5 Whys
        prompt = f"""Perform a "5 Whys" root cause analysis based on this transcript.

TRANSCRIPT:
{transcript}

INITIAL STATEMENT: {initial_statement}

Perform the 5 Whys analysis:
1. Why is this happening/important? (Based on transcript evidence)
2. Why is that? (Dig deeper using context from transcript)
3. Why is that? (Continue deeper)
4. Why is that? (Continue deeper)
5. Why is that? (Root cause/fundamental reason)

For each "Why", provide:
- The question
- The answer based on transcript evidence
- Supporting quotes or references from the transcript

Then conclude with:
- Root Cause identified
- Implications
- Potential solutions or actions (if mentioned in transcript)"""

        response = self._call_ollama(prompt, temperature=0.4)

        return {
            "agent": "5 Whys Analysis",
            "initial_statement": initial_statement,
            "analysis": response,
            "timestamp": datetime.now().isoformat()
        }


class StakeholderAgent(OllamaAgent):
    """Agent for identifying stakeholders and their perspectives."""

    def analyze(self, transcript: str) -> Dict[str, any]:
        """Identify stakeholders and their viewpoints."""

        prompt = f"""Analyze this transcript to identify stakeholders and their perspectives.

TRANSCRIPT:
{transcript}

Please identify:
1. Key Stakeholders (people or groups mentioned or implied)
2. Their Interests (what each stakeholder cares about)
3. Their Concerns (worries or objections mentioned)
4. Their Influence (who has decision-making power)
5. Alignment (where stakeholders agree or disagree)

Format your response with clear sections."""

        response = self._call_ollama(prompt)

        return {
            "agent": "Stakeholder Analysis",
            "analysis": response,
            "timestamp": datetime.now().isoformat()
        }


class ActionItemsAgent(OllamaAgent):
    """Agent for extracting action items, decisions, and next steps."""

    def analyze(self, transcript: str) -> Dict[str, any]:
        """Extract actionable items and decisions."""

        prompt = f"""Analyze this transcript to extract action items, decisions, and next steps.

TRANSCRIPT:
{transcript}

Please identify:
1. Decisions Made (concrete decisions reached)
2. Action Items (specific tasks to be done)
3. Owners (who is responsible for what)
4. Deadlines (any timeframes mentioned)
5. Dependencies (what needs to happen first)
6. Next Steps (immediate follow-up actions)

Format as a structured list."""

        response = self._call_ollama(prompt)

        return {
            "agent": "Action Items Extraction",
            "analysis": response,
            "timestamp": datetime.now().isoformat()
        }


class InsightsAgent(OllamaAgent):
    """Agent for generating strategic insights and recommendations."""

    def analyze(self, transcript: str, previous_analyses: List[Dict] = None) -> Dict[str, any]:
        """Generate strategic insights based on all analyses."""

        context = ""
        if previous_analyses:
            context = "\n\nPREVIOUS ANALYSES:\n"
            for analysis in previous_analyses:
                context += f"\n{analysis['agent']}:\n{analysis['analysis']}\n"

        prompt = f"""Based on this transcript and previous analyses, provide strategic insights and recommendations.

TRANSCRIPT:
{transcript}
{context}

Please provide:
1. Key Insights (3-5 strategic observations)
2. Hidden Patterns (things not immediately obvious)
3. Risks & Opportunities (potential issues and benefits)
4. Recommendations (actionable suggestions)
5. Questions to Consider (what's unclear or needs clarification)

Be thoughtful and strategic in your analysis."""

        response = self._call_ollama(prompt, temperature=0.5)

        return {
            "agent": "Strategic Insights",
            "analysis": response,
            "timestamp": datetime.now().isoformat()
        }


class VideoAnalyzer:
    """Orchestrates multiple agents for comprehensive video analysis."""

    def __init__(self, transcript: str, video_name: str, model: str = "llama3"):
        self.transcript = transcript
        self.video_name = video_name
        self.model = model
        self.analyses = []

    def run_full_analysis(self, progress_callback=None) -> Dict[str, any]:
        """Run all analysis agents and compile results."""

        logger.info("Starting comprehensive video analysis...")

        agents = [
            ("Theme Analysis", ThemeAnalysisAgent(self.model)),
            ("Goal Identification", GoalIdentificationAgent(self.model)),
            ("5 Whys Analysis", FiveWhysAgent(self.model)),
            ("Stakeholder Analysis", StakeholderAgent(self.model)),
            ("Action Items", ActionItemsAgent(self.model)),
        ]

        total_agents = len(agents) + 1  # +1 for insights agent
        current = 0

        # Run each agent
        for name, agent in agents:
            if progress_callback:
                progress_callback(current / total_agents, f"Running {name}...")

            logger.info(f"Running {name}...")
            result = agent.analyze(self.transcript)
            self.analyses.append(result)

            current += 1

        # Run insights agent with all previous analyses
        if progress_callback:
            progress_callback(current / total_agents, "Generating strategic insights...")

        insights_agent = InsightsAgent(self.model)
        insights = insights_agent.analyze(self.transcript, self.analyses)
        self.analyses.append(insights)

        if progress_callback:
            progress_callback(1.0, "Analysis complete!")

        logger.info("Analysis complete")

        return {
            "video_name": self.video_name,
            "model_used": self.model,
            "timestamp": datetime.now().isoformat(),
            "analyses": self.analyses
        }

    def generate_report(self, output_path: Path = None) -> str:
        """Generate a comprehensive analysis report."""

        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE VIDEO ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"\nVideo: {self.video_name}")
        report.append(f"Analysis Model: {self.model}")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n" + "=" * 80)

        # Add each analysis section
        for analysis in self.analyses:
            report.append(f"\n\n{'#' * 80}")
            report.append(f"## {analysis['agent'].upper()}")
            report.append(f"{'#' * 80}\n")
            report.append(analysis['analysis'])

        # Add footer
        report.append("\n\n" + "=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)

        report_text = "\n".join(report)

        # Save to file if path provided
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report_text)
                logger.info(f"Report saved to {output_path}")
            except Exception as e:
                logger.error(f"Failed to save report: {e}")

        return report_text

    def get_summary(self) -> str:
        """Get a brief summary of all analyses."""

        summary_parts = []

        for analysis in self.analyses:
            summary_parts.append(f"**{analysis['agent']}**")
            # Take first 200 characters of analysis
            preview = analysis['analysis'][:200] + "..."
            summary_parts.append(preview)
            summary_parts.append("")

        return "\n".join(summary_parts)


def run_analysis(transcript: str, video_name: str, model: str = "llama3",
                 output_dir: Path = None, progress_callback=None) -> Dict[str, any]:
    """
    Convenience function to run full analysis.

    Args:
        transcript: Video transcript text
        video_name: Name of the video
        model: Ollama model to use
        output_dir: Directory to save report
        progress_callback: Optional callback for progress updates

    Returns:
        Dictionary with analysis results and report path
    """

    analyzer = VideoAnalyzer(transcript, video_name, model)
    results = analyzer.run_full_analysis(progress_callback)

    # Generate and save report
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / f"{video_name}_analysis_report.txt"
        report_text = analyzer.generate_report(report_path)
    else:
        report_path = None
        report_text = analyzer.generate_report()

    return {
        "results": results,
        "report_text": report_text,
        "report_path": report_path,
        "analyzer": analyzer
    }
