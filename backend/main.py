from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time

from agents.classifier import classify_input
from agents.evidence   import collect_evidence, format_evidence_for_prompt
from agents.specialist import run_specialist
from agents.critic     import run_critic
from agents.decision   import run_decision
from agents.explainer  import run_explainer
from vector_store.qdrant_client import seed_database

app = FastAPI(
    title="TruthShield API",
    description="AI-powered trust verification platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    try:
        seed_database()
        print("✅ Qdrant seeded and ready")
    except Exception as e:
        print(f"⚠️ Seed failed: {e} — continuing anyway")


class AnalyzeRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {"status": "TruthShield API is running", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    if len(text) > 5000:
        raise HTTPException(status_code=400, detail="Input too long. Max 5000 characters.")

    start = time.time()

    try:
        category      = classify_input(text)
        evidence      = collect_evidence(text, top_k=5)
        evidence_text = format_evidence_for_prompt(evidence)
        specialist    = run_specialist(text, category["category"], evidence_text)
        critic        = run_critic(text, specialist)
        verdict       = run_decision(specialist, critic)
        explanation   = run_explainer(text, verdict, specialist, critic)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    duration_ms = int((time.time() - start) * 1000)

    return {
        "input_text":  text,
        "category":    category,
        "verdict":     verdict,
        "explanation": explanation,
        "flags":       specialist.get("flags", []),
        "evidence":    evidence[:3],
        "duration_ms": duration_ms,
    }