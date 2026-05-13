import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from concept_map import normalize_concept

# =========================================================
# CONFIG
# =========================================================

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
INDEX_PATH = "./vector_db/math_index"


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
        # Perform similarity search on the FAISS index for the query, retrieving the top 3 results. We will check if the top result's concept matches
        #  the expected concept for the query.
        results = db.similarity_search(q, k=3)

        # Get the concept from the metadata of the top retrieved document and normalize it using the same function we use for evaluation. 
        # This ensures that we are comparing
        top = results[0]
        raw = top.metadata.get("concept", "")

        predicted = normalize_concept(raw)
        expected_norm = normalize_concept(expected)

        # Check if the predicted concept matches the expected concept
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
# This test checks the quality of the embedding clusters for each concept. It retrieves a sample of documents for each concept, computes their embeddings,
#  and then calculates the average cosine similarity between the embeddings of different concepts.
#  Ideally, we want to see higher similarity within the same concept and lower similarity across different concepts,
#  which would indicate that the embedding model is effectively capturing the semantic differences between concepts.
def test_embedding_similarity(db):

    print("\n📐 EMBEDDING CLUSTER TEST\n")

    seed_queries = [
        "addition", "subtraction", "division",
        "fractions", "integers", "place value"
    ]

    all_docs = []
    # For each seed query (which corresponds to a concept), we perform a similarity search to retrieve a sample of documents related to that concept.
    #  We then compute the embeddings for the content of those documents and group them by their normalized concept.
    #  Finally, we calculate the average cosine similarity between the embedding clusters of different concepts to assess how well the embedding model is
    #  distinguishing between them.
    for q in seed_queries:
        all_docs.extend(db.similarity_search(q, k=10)) # Higher k for a more robust sample of the embedding space for each concept

    embeddings = db.embedding_function

    concept_vectors = {}

    # For each retrieved document, we get its content and metadata, normalize the concept, compute the embedding vector for the content, and group the vectors
    #  by their normalized concept name. This allows us to analyze the embedding clusters for each concept and compare them to see if they are well-separated
    #  in the embedding space.
    for doc in all_docs:

        concept = normalize_concept(doc.metadata.get("concept", "unknown"))

        vec = embeddings.embed_query(doc.page_content)

        concept_vectors.setdefault(concept, []).append(vec)

    # average vectors because we want to see the overall cluster similarity, not just individual points. This gives us a better sense of how well the embedding model
    #  is grouping similar concepts together and separating different concepts in the vector space.
    def avg(vecs):
        return np.mean(vecs, axis=0)

    keys = list(concept_vectors.keys())

    print("Cross Concept Similarity (lower is better)\n")

    # We calculate the cosine similarity between the average embedding vectors of different concepts. Ideally, we want to see lower similarity scores
    #  between different concepts,
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):

            c1, c2 = keys[i], keys[j]

            v1 = avg(concept_vectors[c1])
            v2 = avg(concept_vectors[c2])

            # Cosine similarity is calculated as the dot product of the two vectors divided by the product of their magnitudes. A lower cosine similarity score 
            # indicates that the vectors are more orthogonal, which in this context would suggest that the embedding model is effectively distinguishing between
            #  the two concepts.
            sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

            print(f"{c1} vs {c2}: {sim:.4f}")


# =========================================================
# NOISE TEST (FIXED VISIBILITY)
# =========================================================
# This test checks how the FAISS index handles queries that are unrelated to the math concepts in the knowledge base. We want to see that such queries do not 
# retrieve relevant concept chunks, which would indicate that the index is not producing false positives for irrelevant queries. This helps us understand the 
# precision of the retrieval system and its ability to filter out noise.
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