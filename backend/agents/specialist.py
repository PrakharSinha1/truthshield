import json
import re
from llm_client import ask

SYSTEM_PROMPTS = {
    "JOB_POST": """You are a job scam detection specialist.

Analyze the given job post for red flags. Check for:
- Unrealistic salary promises (e.g. "earn ₹50,000/day")
- Vague company name or no company mentioned
- Requests for personal documents upfront
- Suspicious urgency ("apply in 24 hours")
- Work-from-home with guaranteed income
- Requests for registration fees or deposits

Reply ONLY with valid JSON. No markdown. No explanation outside JSON.
Format:
{
  "risk_level": "HIGH" | "MEDIUM" | "LOW",
  "flags": ["flag1", "flag2"],
  "reasoning": "brief explanation"
}""",

    "SCAM_MESSAGE": """You are a fraud and scam detection specialist.

Analyze the given message for scam indicators. Check for:
- OTP requests (you will never be asked for OTP by legit organizations)
- Prize/lottery claims ("you have won")
- Urgency and fear tactics ("your account will be blocked")
- Impersonation of banks, govt, companies
- Requests for money transfer or card details
- Suspicious links or phone numbers

Reply ONLY with valid JSON. No markdown. No explanation outside JSON.
Format:
{
  "risk_level": "HIGH" | "MEDIUM" | "LOW",
  "flags": ["flag1", "flag2"],
  "reasoning": "brief explanation"
}""",

    "NEWS_CLAIM": """You are a fact-checking and misinformation detection specialist.

Analyze the given news claim or article. Check for:
- Extraordinary claims without evidence
- Emotionally charged or sensational language
- Missing source attribution
- Logical contradictions within the text
- Known misinformation patterns
- Clickbait-style framing

Reply ONLY with valid JSON. No markdown. No explanation outside JSON.
Format:
{
  "risk_level": "HIGH" | "MEDIUM" | "LOW",
  "flags": ["flag1", "flag2"],
  "reasoning": "brief explanation"
}""",

    "SOCIAL_POST": """You are a social media misinformation specialist.

Analyze the social media post for:
- Unverified viral claims
- Manipulation or emotional baiting
- Hate speech or divisive content
- Impersonation of public figures
- Misleading statistics or out-of-context media

Reply ONLY with valid JSON. No markdown. No explanation outside JSON.
Format:
{
  "risk_level": "HIGH" | "MEDIUM" | "LOW",
  "flags": ["flag1", "flag2"],
  "reasoning": "brief explanation"
}""",

    "OTHER": """You are a general content trust analyst.

Analyze the given content for any trust or safety concerns:
- Deceptive intent
- Harmful information
- Misleading claims

Reply ONLY with valid JSON. No markdown. No explanation outside JSON.
Format:
{
  "risk_level": "HIGH" | "MEDIUM" | "LOW",
  "flags": ["flag1", "flag2"],
  "reasoning": "brief explanation"
}"""
}


def run_specialist(text: str, category: str, evidence_text: str) -> dict:
    """
    Run the appropriate specialist agent based on content category.
    Returns: {"risk_level": str, "flags": list, "reasoning": str}
    """
    system = SYSTEM_PROMPTS.get(category, SYSTEM_PROMPTS["OTHER"])

    prompt = f"""Content to analyze:
\"\"\"{text}\"\"\"

Similar known cases from our database:
{evidence_text}

Analyze the content above."""

    raw = ask(prompt, system)
    clean = re.sub(r"```json|```", "", raw).strip()

    try:
        result = json.loads(clean)
        assert "risk_level" in result
        return result
    except Exception:
        return {
            "risk_level": "MEDIUM",
            "flags": ["parsing error — manual review recommended"],
            "reasoning": raw[:300]
        }