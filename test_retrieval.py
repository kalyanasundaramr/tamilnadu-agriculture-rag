
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load API key from .env
load_dotenv()

# Project and vector database paths
PROJECT_DIR = Path(__file__).resolve().parent
VECTORSTORE_DIR = PROJECT_DIR / "vectorstore"

# Check the index file exists
index_file = VECTORSTORE_DIR / "faiss_index.faiss"

if not index_file.exists():
    raise FileNotFoundError(
        f"FAISS index not found: {index_file}"
    )

# Use the same embedding model/configuration as ingest.py
embeddings = OpenAIEmbeddings()

# Load the existing FAISS index
db = FAISS.load_local(
    str(VECTORSTORE_DIR),
    embeddings,
    index_name="faiss_index",
    allow_dangerous_deserialization=True,
)

questions = [
    "List the agriculture schemes available for farmers in Tamil Nadu",
    "PM-KISAN farmer eligibility and benefits",
    "Tamil Nadu agriculture subsidy schemes",
]

for question in questions:
    print("\n" + "=" * 70)
    print("QUESTION:", question)

    docs = db.similarity_search(question, k=8)

    print("Retrieved chunks:", len(docs))

    for i, doc in enumerate(docs, start=1):
        print(f"\n--- Chunk {i} ---")
        print("Source:", doc.metadata.get("source", "Unknown"))
        print(doc.page_content[:1000])