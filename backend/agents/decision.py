# Decision Agent
# Pure logic — no LLM needed here.
# Combines specialist risk + critic penalty → final verdict + confidence score.

RISK_BASE_SCORES = {
    "HIGH":   0.90,
    "MEDIUM": 0.55,
    "LOW":    0.15,
}

VERDICT_THRESHOLDS = [
    (0.80, "Likely Scam / Fake"),
    (0.55, "Suspicious — Verify Before Trusting"),
    (0.30, "Possibly Misleading — Proceed with Caution"),
    (0.00, "Appears Legitimate"),
]


def run_decision(specialist_result: dict, critic_result: dict) -> dict:
    """
    Produce a final verdict and confidence score.
    Returns: {"verdict": str, "confidence": float, "confidence_pct": int}
    """
    risk_level = specialist_result.get("risk_level", "MEDIUM").upper()
    base_score = RISK_BASE_SCORES.get(risk_level, 0.55)

    penalty = float(critic_result.get("confidence_penalty", 0.0))
    final_score = round(max(0.0, min(1.0, base_score - penalty)), 3)

    verdict = "Unknown"
    for threshold, label in VERDICT_THRESHOLDS:
        if final_score >= threshold:
            verdict = label
            break

    return {
        "verdict": verdict,
        "confidence": final_score,
        "confidence_pct": int(final_score * 100),
        "risk_level": risk_level,
    }