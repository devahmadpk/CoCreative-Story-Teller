import os
import ollama

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from concept_map import normalize_concept


# =========================================================
# CONFIG
# =========================================================

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(BASE_DIR, "vector_db", "math_index")


# =========================================================
# PROMPT ENGINE
# =========================================================

class PromptEngine:

    def build(self, conversation, query, context_docs):

        context_text = "\n\n".join([doc.page_content for doc in context_docs[:3]])

        return f"""
You are a friendly math tutor for kids (Grade 1–6).

Your job:
- Understand the conversation
- Use previous messages when needed
- Solve the latest question correctly
- Explain simply using a short story if helpful

---

CONVERSATION:
{conversation}

---

KNOWLEDGE (if useful):
{context_text}

---

LATEST USER MESSAGE:
{query}

---

RULES:

1. ALWAYS focus on the latest message
2. USE previous conversation if it continues the same situation
3. DO NOT repeat the question
4. DO NOT ask unnecessary questions
5. DO NOT invent numbers
6. Give correct math answer
7. Keep explanation simple and short

---

Now give the answer:
"""


# =========================================================
# MAIN ENGINE
# =========================================================

class MathRAGEngine:

    def __init__(self):

        # embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
            model_kwargs={"device": "cpu"}
        )

        # vector db
        self.db = FAISS.load_local(
            INDEX_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        self.prompt_engine = PromptEngine()

        # conversation memory
        self.chat_history = []
        self.max_history = 12


    # =====================================================
    # MEMORY
    # =====================================================

    def build_conversation(self):

        history = self.chat_history[-self.max_history:]

        text = ""
        for turn in history:
            text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"

        return text.strip()


    # =====================================================
    # LLM
    # =====================================================

    def llm(self, prompt):

        response = ollama.chat(
            model="gemma:2b",
            messages=[{"role": "user", "content": prompt}]
        )

        return response["message"]["content"]


    # =====================================================
    # MAIN ASK
    # =====================================================

    def ask(self, query):

        # 1. retrieve context (simple, no filtering)
        docs = self.db.similarity_search(query, k=4)

        # 2. build conversation
        conversation = self.build_conversation()

        # 3. build prompt
        prompt = self.prompt_engine.build(
            conversation=conversation,
            query=query,
            context_docs=docs
        )

        # 4. get response
        response = self.llm(prompt)

        # 5. store memory
        self.chat_history.append({
            "user": query,
            "assistant": response
        })

        if len(self.chat_history) > 20:
            self.chat_history.pop(0)

        # 6. format output
        return f"\nRAG:\n{'*'*40}\n{response.strip()}\n{'*'*40}\n"


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    rag = MathRAGEngine()

    while True:
        q = input("\nAsk: ")
        if q.lower() in ["exit", "quit"]:
            break

        print(rag.ask(q))