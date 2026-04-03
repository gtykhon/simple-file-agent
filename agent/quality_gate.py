"""Two-stage code quality gate.

Stage 1 — Generate: fast model produces code.
Stage 2 — Review:   quality model scores it on 5 axes.

If the review rejects the code, the rejection feedback is appended to the
generation prompt and the cycle repeats (up to max_retries).

Scoring axes and weights (must match the JSON keys the reviewer returns):
  syntax         25%   — valid, parseable code
  logic          25%   — correct reasoning and flow
  error_handling 20%   — handles edge cases and failures
  best_practices 15%   — idiomatic, readable, maintainable
  security       15%   — no obvious vulnerabilities
"""

import ast
import json
import logging
import re
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

SCORE_WEIGHTS = {
    "syntax":         0.25,
    "logic":          0.25,
    "error_handling": 0.20,
    "best_practices": 0.15,
    "security":       0.15,
}

APPROVAL_THRESHOLD = 6  # all axes must score >= this out of 10

REVIEWER_SYSTEM = f"""You are a strict code reviewer. Score the code on five axes (0–10 each).
Respond with ONLY valid JSON — no extra text:
{{
  "syntax":         <0-10>,
  "logic":          <0-10>,
  "error_handling": <0-10>,
  "best_practices": <0-10>,
  "security":       <0-10>,
  "feedback":       "<one concise sentence explaining the most important issue>",
  "approved":       <true if ALL scores >= {APPROVAL_THRESHOLD}, else false>
}}"""


# --------------------------------------------------------------------------- #
# Result dataclass                                                               #
# --------------------------------------------------------------------------- #

@dataclass
class ReviewResult:
    approved: bool
    score: float          # weighted 0.0–1.0
    feedback: str
    syntax_valid: bool    # AST check (Python only)
    raw_scores: dict      # {"syntax": 8, "logic": 7, ...}


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

def _python_syntax_ok(code: str) -> bool:
    """Return True if *code* is valid Python (AST parse succeeds)."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _extract_json(text: str) -> Optional[dict]:
    """Try direct parse, then extract from a ```json ... ``` block."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Last-ditch: find first {...} block
    match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def _weighted_score(scores: dict) -> float:
    total = 0.0
    for axis, weight in SCORE_WEIGHTS.items():
        total += (scores.get(axis, 0) / 10) * weight
    return round(total, 3)


# --------------------------------------------------------------------------- #
# Quality gate                                                                  #
# --------------------------------------------------------------------------- #

class QualityGate:
    """
    Two-stage gate: generate → review → (retry if rejected).

    Args:
        client: A MultiModelClient instance.
        max_retries: How many regeneration attempts to make after rejection.
    """

    def __init__(self, client, max_retries: int = 2):
        self.client = client
        self.max_retries = max_retries

    def review(self, code: str, context: str = "") -> ReviewResult:
        """
        Ask the quality model to review *code*.
        Always uses cloud (force_cloud=True) for consistent review quality.
        Falls back to syntax-only result if JSON parsing fails after all attempts.
        """
        syntax_ok = _python_syntax_ok(code)
        prompt = f"Context: {context}\n\nCode to review:\n```python\n{code}\n```"

        for attempt in range(self.max_retries + 1):
            response, model = self.client.generate(
                prompt, system=REVIEWER_SYSTEM, force_cloud=True
            )
            data = _extract_json(response)
            if data:
                raw_scores = {k: data.get(k, 0) for k in SCORE_WEIGHTS}
                return ReviewResult(
                    approved=bool(data.get("approved", False)) and syntax_ok,
                    score=_weighted_score(raw_scores),
                    feedback=data.get("feedback", "No feedback provided"),
                    syntax_valid=syntax_ok,
                    raw_scores=raw_scores,
                )
            logger.debug("Attempt %d/%d: could not parse reviewer JSON", attempt + 1, self.max_retries + 1)

        # Fallback — syntax check only
        logger.warning("Quality gate: JSON parsing failed on all attempts, using syntax-only result")
        return ReviewResult(
            approved=syntax_ok,
            score=0.5,
            feedback="Reviewer returned unparseable output — syntax check only",
            syntax_valid=syntax_ok,
            raw_scores={},
        )

    def generate_and_review(
        self, generation_prompt: str, context: str = ""
    ) -> Tuple[str, ReviewResult]:
        """
        Full two-stage pipeline:
          1. Generate code (fast/local model preferred)
          2. Review it (cloud model, consistent quality)
          3. If rejected, enrich prompt with feedback and regenerate
          4. Return final (code, ReviewResult)

        The last attempt is always returned, even if still rejected.
        """
        prompt = generation_prompt

        for attempt in range(self.max_retries + 1):
            code, model = self.client.generate(prompt)
            logger.debug("Generated by %s (attempt %d)", model, attempt + 1)

            result = self.review(code, context)

            if result.approved:
                logger.info("Quality gate PASSED — score=%.2f", result.score)
                return code, result

            logger.info(
                "Quality gate REJECTED (attempt %d/%d) — score=%.2f — %s",
                attempt + 1, self.max_retries + 1, result.score, result.feedback,
            )
            # Enrich prompt so the next generation has the reviewer's feedback
            prompt = (
                f"{generation_prompt}\n\n"
                f"[Previous attempt rejected]\n"
                f"Feedback: {result.feedback}\n"
                f"Fix those issues in your next response."
            )

        # Final attempt — return as-is
        code, _ = self.client.generate(prompt)
        result = self.review(code, context)
        logger.info("Quality gate final result — approved=%s score=%.2f", result.approved, result.score)
        return code, result
