import os
import json
import logging
import warnings
import shutil

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- CONFIG ---
INDEX_PATH = "../vector_db/math_index"
CONTENT_FILE = "../data/math_content.jsonl"   # 🔥 CHANGED (IMPORTANT)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


# ----------------------------
# PARSER (NEW JSON SCHEMA)
# ----------------------------
def parse_jsonl(file_path: str) -> list[Document]:
    """
    Parses new structured KB:
    Each line = 1 JSON object
    """
    documents = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)

                # --- CORE TEXT BUILDING ---
                text_parts = []

                text_parts.append(f"Concept: {obj.get('concept', '')}")
                text_parts.append(f"Type: {obj.get('type', '')}")
                text_parts.append(f"Grade: {obj.get('grade', '')}")
                text_parts.append(f"Topic: {obj.get('topic', '')}")

                if obj.get("content"):
                    text_parts.append(f"Content: {obj['content']}")

                if obj.get("solution"):
                    text_parts.append(f"Solution: {obj['solution']}")

                content = "\n".join(text_parts)

                # --- METADATA (VERY IMPORTANT FOR RAG FILTERING) ---
                metadata = {
                    "id": obj.get("id", ""),
                    "type": obj.get("type", ""),
                    "grade": obj.get("grade", 0),
                    "topic": obj.get("topic", ""),
                    "concept": obj.get("concept", "")
                }

                documents.append(Document(
                    page_content=content,
                    metadata=metadata
                ))

            except json.JSONDecodeError:
                print(f"⚠️ Skipping invalid JSON line: {line[:50]}...")

    return documents


# ----------------------------
# INGESTION PIPELINE
# ----------------------------
def run_ingestion():
    print("\n" + "=" * 60)
    print("🚀 BUILDING OPTIMIZED MATH RAG BRAIN (GRADE 1–6)")
    print("=" * 60)

    if not os.path.exists(CONTENT_FILE):
        print(f"❌ Error: {CONTENT_FILE} not found.")
        return

    # Clean old index
    if os.path.exists(INDEX_PATH):
        shutil.rmtree(INDEX_PATH)

    # 1. Parse structured KB
    print("\n[1/3] Parsing JSONL knowledge base...")
    documents = parse_jsonl(CONTENT_FILE)

    print(f"   ✔ Loaded {len(documents)} atomic knowledge nodes")

    # 2. Embeddings (optimized for CPU)
    print("\n[2/3] Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
        model_kwargs={"device": "cpu"}
    )

    # 3. FAISS indexing
    print("\n[3/3] Building FAISS index...")

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

    vector_db = FAISS.from_documents(documents, embeddings)
    vector_db.save_local(INDEX_PATH)

    print("\n" + "=" * 60)
    print(f"✅ SUCCESS: Indexed {vector_db.index.ntotal} reasoning nodes")
    print("🚀 System ready for RAG reasoning + storytelling mode")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_ingestion()