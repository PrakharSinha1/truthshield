import json
import re
from llm_client import ask

SYSTEM = """You are the Critic Agent in a trust verification pipeline.

Your job is to CHALLENGE the specialist's analysis. Act as a devil's advocate.
You are NOT trying to defend the content — you are ensuring the analysis is rigorous.

Ask yourself:
- Is the evidence sufficient to make this conclusion?
- Are we making unfair assumptions?
- Could there be a legitimate explanation?
- Is the confidence level appropriate given what we know?
- Are any flags weak or circumstantial?

Be critical but fair. Your goal is accuracy, not leniency.

Reply ONLY with valid JSON. No markdown. No explanation outside JSON.
Format:
{
  "challenges": ["challenge1", "challenge2"],
  "alternative_explanation": "possible innocent explanation if any, or 'none'",
  "confidence_penalty": 0.0
}

confidence_penalty rules:
- 0.0  → specialist analysis is solid, no penalty
- 0.05 → minor uncertainty
- 0.10 → some assumptions made
- 0.20 → significant gaps in evidence
- 0.30 → analysis is largely speculative"""


def run_critic(text: str, specialist_result: dict) -> dict:
    """
    Challenge the specialist's analysis.
    Returns: {"challenges": list, "alternative_explanation": str, "confidence_penalty": float}
    """
    prompt = f"""Original content:
\"\"\"{text}\"\"\"

Specialist concluded:
{json.dumps(specialist_result, indent=2)}

Challenge this analysis."""

    raw = ask(prompt, SYSTEM)
    clean = re.sub(r"```json|```", "", raw).strip()

    try:
        result = json.loads(clean)
        assert "confidence_penalty" in result
        # Clamp penalty between 0 and 0.3
        result["confidence_penalty"] = max(0.0, min(0.3, float(result["confidence_penalty"])))
        return result
    except Exception:
        return {
            "challenges": ["critic agent parsing error"],
            "alternative_explanation": "unknown",
            "confidence_penalty": 0.05
        }