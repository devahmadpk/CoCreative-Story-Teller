import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from concept_map import normalize_concept

# =========================================================
# CONFIG
# =========================================================

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
INDEX_PATH = "../vector_db/math_index"


# =========================================================
# LOAD DB
# =========================================================

def load_db():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
        model_kwargs={"device": "cpu"}
    )

    return FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


# =========================================================
# TEST QUERIES
# =========================================================

TEST_QUERIES = [
    ("I have 2 apples and get 3 more", "addition"),
    ("I had 10 candies and gave 4 away", "subtraction"),
    ("Share 12 candies among 3 kids", "division"),
    ("What is half of 8", "fractions"),
    ("What is value of 3 in 354", "place value"),
    ("Which is bigger -3 or -1", "integers"),
]


# =========================================================
# RETRIEVAL TEST
# =========================================================

def test_retrieval(db):

    print("\n🔍 RETRIEVAL TEST\n")

    correct = 0

    for q, expected in TEST_QUERIES:

        results = db.similarity_search(q, k=3)

        top = results[0]
        raw = top.metadata.get("concept", "")

        predicted = normalize_concept(raw)
        expected_norm = normalize_concept(expected)

        match = predicted == expected_norm

        print(f"Q: {q}")
        print(f"Expected: {expected_norm} | Got: {predicted}")
        print("✔" if match else "❌", "\n")

        if match:
            correct += 1

    acc = correct / len(TEST_QUERIES)
    print(f"📊 Retrieval Accuracy: {acc:.2f}")


# =========================================================
# EMBEDDING CLUSTER QUALITY (FIXED SAMPLING)
# =========================================================

def test_embedding_similarity(db):

    print("\n📐 EMBEDDING CLUSTER TEST\n")

    seed_queries = [
        "addition", "subtraction", "division",
        "fractions", "integers", "place value"
    ]

    all_docs = []

    for q in seed_queries:
        all_docs.extend(db.similarity_search(q, k=10))

    embeddings = db.embedding_function

    concept_vectors = {}

    for doc in all_docs:

        concept = normalize_concept(doc.metadata.get("concept", "unknown"))

        vec = embeddings.embed_query(doc.page_content)

        concept_vectors.setdefault(concept, []).append(vec)

    # average vectors
    def avg(vecs):
        return np.mean(vecs, axis=0)

    keys = list(concept_vectors.keys())

    print("Cross Concept Similarity (lower is better)\n")

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):

            c1, c2 = keys[i], keys[j]

            v1 = avg(concept_vectors[c1])
            v2 = avg(concept_vectors[c2])

            sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

            print(f"{c1} vs {c2}: {sim:.4f}")


# =========================================================
# NOISE TEST (FIXED VISIBILITY)
# =========================================================

def test_noise(db):

    print("\n🧪 NOISE TEST\n")

    noise = [
        "cricket match today",
        "who is Messi",
        "best phone under 100k",
        "weather tomorrow"
    ]

    for q in noise:

        results = db.similarity_search(q, k=3)

        concepts = [
            normalize_concept(r.metadata.get("concept", ""))
            for r in results
        ]

        print(f"Query: {q}")
        print(f"Retrieved: {concepts}")
        print("-")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print("\n==============================")
    print(" EMBEDDING TEST SUITE FIXED")
    print("==============================")

    db = load_db()

    test_retrieval(db)
    test_embedding_similarity(db)
    test_noise(db)

    print("\nDONE\n")