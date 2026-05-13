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

# Parse the json file, read every object, convert it into a string with all the metadata and add it to a list of Documents.
# Each Document has a page_content (the string that will be embedded) and metadata (the structured fields that will be used 
# for filtering during retrieval and evaluation).
def parse_jsonl(file_path: str) -> list[Document]:
    """
    Parses new structured KB:
    Each line = 1 JSON object
    """
    # List to hold all documents for FAISS ingestion
    documents = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)

                # --- CONTENT (COMBINE ALL FIELDS INTO ONE STRING FOR EMBEDDING) ---
                text_parts = []
                text_parts.append(f"Concept: {obj.get('concept', '')}")
                text_parts.append(f"Type: {obj.get('type', '')}")
                text_parts.append(f"Grade: {obj.get('grade', '')}")
                text_parts.append(f"Topic: {obj.get('topic', '')}")

                # Optional fields
                if obj.get("content"):
                    text_parts.append(f"Content: {obj['content']}")
                if obj.get("solution"):
                    text_parts.append(f"Solution: {obj['solution']}")

                # Combine all parts into one string for embedding
                content = "\n".join(text_parts)

                # --- METADATA (KEEP STRUCTURED FOR RAG) ---
                metadata = {
                    "id": obj.get("id", ""),
                    "type": obj.get("type", ""),
                    "grade": obj.get("grade", 0),
                    "topic": obj.get("topic", ""),
                    "concept": obj.get("concept", "")
                }

                # --- CREATE DOCUMENT ---
                documents.append(Document(
                    page_content=content, # This is the raw string of text that the BGE model will embed
                    metadata=metadata # Used for filtering. Not embedded but stored in FAS for retrieval and evaluation
                ))

            except json.JSONDecodeError:
                print(f"⚠️ Skipping invalid JSON line: {line[:50]}...")

    # This function returns a list of Document objects, each containing the combined content for embedding
    #  and structured metadata for filtering. This allows us to build a FAISS index that can be efficiently
    #  queried based on both semantic similarity and metadata filters during RAG retrieval.
    return documents


# ----------------------------
# INGESTION PIPELINE
# ----------------------------
def run_ingestion():
    print("\n" + "=" * 60)
    print("🚀 BUILDING OPTIMIZED MATH RAG BRAIN (GRADE 1–6)")
    print("=" * 60)

    # Check if content file exists
    if not os.path.exists(CONTENT_FILE):
        print(f"❌ Error: {CONTENT_FILE} not found.")
        return

    # Clear existing index if it exists (for fresh ingestion)
    if os.path.exists(INDEX_PATH):
        shutil.rmtree(INDEX_PATH)

    # 1. Parse structured KB
    print("\n[1/3] Parsing JSONL knowledge base...")

    # This function reads the JSONL file line by line, converts each JSON object into a Document with 
    # combined content for embedding and structured metadata for filtering. It also handles any malformed lines gracefully.
    documents = parse_jsonl(CONTENT_FILE)

    print(f"   ✔ Loaded {len(documents)} atomic knowledge nodes")

    # 2. Generate embeddings with BGE (Initializing)
    print("\n[2/3] Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
        model_kwargs={"device": "cpu"}
    )

    # 3. Build FAISS index
    print("\n[3/3] Building FAISS index...")

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

    # This creates a FAISS vector store from the list of Documents and their embeddings. It then saves the index
    #  locally for use in RAG retrieval. The metadata is stored in FAISS but not embedded, allowing
    #  for efficient filtering during retrieval.
    vector_db = FAISS.from_documents(documents, embeddings)
    vector_db.save_local(INDEX_PATH)

    print("\n" + "=" * 60)
    print(f"✅ SUCCESS: Indexed {vector_db.index.ntotal} reasoning nodes")
    print("🚀 System ready for RAG reasoning + storytelling mode")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_ingestion()