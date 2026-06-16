from vector_store.qdrant_client import search_similar


def collect_evidence(text: str, top_k: int = 5) -> list:
    """
    Retrieve semantically similar known scams / fake news from vector DB.
    Returns a list of {"text": str, "score": float, "label": str}
    """
    try:
        results = search_similar(text, top_k=top_k)
        return results
    except Exception:
        return []


def format_evidence_for_prompt(evidence: list) -> str:
    """
    Convert evidence list into a readable string for agent prompts.
    """
    if not evidence:
        return "No similar cases found in database."

    lines = []
    for i, e in enumerate(evidence, 1):
        score_pct = round(e.get("score", 0) * 100, 1)
        label = e.get("label", "unknown")
        text = e.get("text", "")
        lines.append(f"{i}. [{label}] (similarity: {score_pct}%) — {text}")

    return "\n".join(lines)