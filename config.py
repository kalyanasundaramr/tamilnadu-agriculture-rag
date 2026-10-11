"""Shared configuration for the Tamil Nadu Agriculture Schemes RAG app."""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
EXTRACTED_DIR = DATA_DIR / "extracted"
MEDIA_DIR = DATA_DIR / "media"
VECTOR_DIR = BASE_DIR / "vectorstore"
LOG_DIR = BASE_DIR / "logs"
MANIFEST_PATH = DATA_DIR / "manifest.json"
LOG_PATH = LOG_DIR / "ingest.log"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip()
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip()

# Add or remove official source pages here. Ingestion follows only approved domains.
START_URLS = [
    "https://www.tnagriculture.in/",
    "https://www.tnhorticulture.tn.gov.in/",
    "https://www.pmkisan.gov.in/",
    "https://pmfby.gov.in/",
]
ALLOWED_DOMAINS = {
    "tnagriculture.in", "www.tnagriculture.in",
    "tnhorticulture.tn.gov.in", "www.tnhorticulture.tn.gov.in",
    "pmkisan.gov.in", "www.pmkisan.gov.in",
    "pmfby.gov.in", "www.pmfby.gov.in",
    "tn.gov.in", "www.tn.gov.in",
}
MAX_PAGES_PER_DOMAIN = int(os.getenv("MAX_PAGES_PER_DOMAIN", "25"))
REQUEST_TIMEOUT_MS = int(os.getenv("REQUEST_TIMEOUT_MS", "25000"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "5"))
# FAISS similarity scores are distance-like for this setup; use only if calibrated on your data.
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "1.5"))

def ensure_directories():
    for path in (PDF_DIR, EXTRACTED_DIR, MEDIA_DIR, VECTOR_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)
