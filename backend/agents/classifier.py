import json
import re
from llm_client import ask

SYSTEM = """You are a content classifier for a trust verification platform.

Classify the given input into EXACTLY ONE of these categories:
- NEWS_CLAIM     → news articles, viral claims, political statements
- JOB_POST       → job offers, internship posts, recruitment messages
- SCAM_MESSAGE   → phishing, OTP scams, prize fraud, financial fraud
- SOCIAL_POST    → social media posts, tweets, captions
- OTHER          → anything that doesn't fit above

Rules:
- Reply ONLY with valid JSON. No explanation. No markdown. No backticks.
- Format: {"category": "CATEGORY_HERE", "confidence": 0.00}
- Confidence must be a float between 0.0 and 1.0"""


def classify_input(text: str) -> dict:
    """
    Classify input text into a content category.
    Returns: {"category": str, "confidence": float}
    """
    raw = ask(text, SYSTEM)
    # Strip any accidental markdown fences Gemini might add
    clean = re.sub(r"```json|```", "", raw).strip()
    try:
        result = json.loads(clean)
        # Validate keys exist
        assert "category" in result and "confidence" in result
        return result
    except Exception:
        # Fallback if parsing fails
        return {"category": "OTHER", "confidence": 0.5}