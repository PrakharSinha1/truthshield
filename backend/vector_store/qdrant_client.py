from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import hashlib
import struct
import uuid

COLLECTION  = "truthshield_evidence"
VECTOR_SIZE = 128  # small vector, no ML model needed

_client = QdrantClient(location=":memory:")


def _text_to_vector(text: str) -> list:
    """
    Lightweight deterministic vector from text using hashing.
    No ML model, no torch, no RAM issues.
    """
    vector = []
    text_lower = text.lower()
    for i in range(VECTOR_SIZE):
        chunk = text_lower[i::VECTOR_SIZE] or str(i)
        h = hashlib.md5((chunk + str(i)).encode()).digest()
        val = struct.unpack('f', h[:4])[0]
        # normalize to -1 to 1
        val = (val % 2.0) - 1.0
        vector.append(val)
    return vector


def init_collection(reset: bool = False):
    existing = [c.name for c in _client.get_collections().collections]
    if COLLECTION in existing:
        if reset:
            _client.delete_collection(COLLECTION)
        else:
            return
    _client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )


def add_evidence(text: str, label: str, source: str = "manual"):
    vector = _text_to_vector(text)
    _client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": text, "label": label, "source": source},
            )
        ],
    )


def search_similar(query: str, top_k: int = 5) -> list:
    try:
        vector = _text_to_vector(query)
        results = _client.search(
            collection_name=COLLECTION,
            query_vector=vector,
            limit=top_k,
        )
        return [
            {
                "text":  r.payload.get("text", ""),
                "score": round(r.score, 4),
                "label": r.payload.get("label", "unknown"),
            }
            for r in results
        ]
    except Exception:
        return []


SEED_DATA = [
    ("Earn 50000 daily working from home. No experience needed. Apply now!", "job_scam"),
    ("Urgent hiring! Data entry job from home. Salary 80000 per month. No interview required.", "job_scam"),
    ("We are hiring remote workers. Guaranteed 5000 per day. Send your Aadhaar to apply.", "job_scam"),
    ("Fake internship at Google India. Pay 2000 registration fee to secure your slot.", "job_scam"),
    ("Part-time job offer: Like YouTube videos and earn 500 per video. WhatsApp us.", "job_scam"),
    ("Congratulations! You have won 10 lakh in SBI lottery. Share your OTP to claim.", "scam"),
    ("Your KYC is expired. Click the link to update immediately or your account will be blocked.", "scam"),
    ("TRAI is blocking your number in 2 hours. Call this number immediately to appeal.", "scam"),
    ("You have received a refund of 9800. Enter your UPI PIN to receive the amount.", "scam"),
    ("Dear customer, your Amazon order is stuck. Pay 49 customs fee via this link.", "scam"),
    ("Government bans all private banks in India starting next month.", "fake_news"),
    ("Scientists confirm 5G towers are spreading COVID-19 in urban areas.", "fake_news"),
    ("India to switch to 4-day work week from January. Official notification issued.", "fake_news"),
    ("WhatsApp will start charging users 99 per month from next week.", "fake_news"),
    ("NASA confirms asteroid will hit Earth in 2025. Evacuation plans underway.", "fake_news"),
    ("RBI announces 0.25% repo rate cut to boost economic growth.", "legitimate"),
    ("ISRO successfully launches communication satellite into geostationary orbit.", "legitimate"),
    ("Apply for the Infosys campus placement drive. Eligible: BE/BTech 2025 batch.", "legitimate"),
]


def seed_database():
    init_collection()
    for text, label in SEED_DATA:
        add_evidence(text, label, source="seed")
    print(f"Seeded {len(SEED_DATA)} entries into Qdrant collection '{COLLECTION}'")


if __name__ == "__main__":
    seed_database()