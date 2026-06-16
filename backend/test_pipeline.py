"""
test_pipeline.py
Run this BEFORE starting the server to verify all agents work.

Usage:
    python test_pipeline.py
"""

import json
from agents.classifier import classify_input
from agents.evidence   import collect_evidence, format_evidence_for_prompt
from agents.specialist import run_specialist
from agents.critic     import run_critic
from agents.decision   import run_decision
from agents.explainer  import run_explainer

TEST_INPUTS = [
    "Congratulations! You won Rs 10 lakh in SBI lottery. Share your OTP to claim your prize now.",
    "Urgent hiring! Data entry job from home. Earn Rs 80,000 per month. No interview. Apply now.",
    "Government bans all private banks in India starting next month. Withdraw your money immediately.",
]


def run_pipeline(text: str):
    print("\n" + "="*60)
    print(f"INPUT: {text[:80]}...")
    print("="*60)

    print("→ Classifying...")
    category = classify_input(text)
    print(f"  Category: {category}")

    print("→ Collecting evidence...")
    evidence = collect_evidence(text)
    evidence_text = format_evidence_for_prompt(evidence)
    print(f"  Found {len(evidence)} similar cases")

    print("→ Running specialist...")
    specialist = run_specialist(text, category["category"], evidence_text)
    print(f"  Risk: {specialist.get('risk_level')} | Flags: {specialist.get('flags')}")

    print("→ Running critic...")
    critic = run_critic(text, specialist)
    print(f"  Penalty: {critic.get('confidence_penalty')} | Challenges: {len(critic.get('challenges', []))}")

    print("→ Running decision...")
    verdict = run_decision(specialist, critic)
    print(f"  Verdict: {verdict['verdict']} ({verdict['confidence_pct']}%)")

    print("→ Generating explanation...")
    explanation = run_explainer(text, verdict, specialist, critic)
    print(f"\nEXPLANATION:\n{explanation}")

    return verdict


if __name__ == "__main__":
    for text in TEST_INPUTS:
        try:
            run_pipeline(text)
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n✅ Pipeline test complete.")