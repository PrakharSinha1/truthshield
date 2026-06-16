import json
from llm_client import ask

SYSTEM = """You are the Explainability Agent for TruthShield, a trust verification platform.

Your job is to explain the verdict to a non-technical user in simple, clear language.

Rules:
- Write in plain English. No jargon.
- Use bullet points (start each with •)
- Maximum 5 bullet points
- First bullet: state what the content is and the overall verdict
- Remaining bullets: explain specific reasons WHY
- Last bullet (if applicable): what the user should do next
- Do NOT repeat the raw JSON or technical terms like "risk_level"
- Be direct and honest. Do not soften scam verdicts."""


def run_explainer(
    text: str,
    verdict: dict,
    specialist_result: dict,
    critic_result: dict,
) -> str:
    """
    Generate a plain-English explanation of the verdict.
    Returns: str (bullet-pointed explanation)
    """
    prompt = f"""Content analyzed:
\"\"\"{text}\"\"\"

Final verdict: {verdict['verdict']} (confidence: {verdict['confidence_pct']}%)

Flags detected by specialist:
{json.dumps(specialist_result.get('flags', []), indent=2)}

Specialist reasoning:
{specialist_result.get('reasoning', 'N/A')}

Critic raised these concerns:
{json.dumps(critic_result.get('challenges', []), indent=2)}

Alternative explanation considered:
{critic_result.get('alternative_explanation', 'none')}

Now explain this verdict to a regular user in simple bullet points."""

    explanation = ask(prompt, SYSTEM)
    return explanation.strip()