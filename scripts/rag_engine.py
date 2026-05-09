import os
import re
import ollama

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# =========================================================
# CONFIG
# =========================================================

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL       = "gemma2:2b"

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(BASE_DIR, "vector_db", "math_index")

MAX_HISTORY = 12
RAG_K       = 2

VAGUE_TRIGGERS = {
    "yes", "yeah", "yep", "ok", "okay", "sure", "go", "more",
    "again", "next", "continue", "another", "yup", "alright",
    "cool", "great", "nice", "fine", "do it", "let's go"
}

# Declarative openers — inputs starting with these are statements, not questions
DECLARATIVE_OPENERS = (
    "i have", "i got", "i found", "i picked", "i start",
    "i own", "i bought", "i collected", "i earned", "i received",
    "there are", "there is", "we have", "she has", "he has",
)

NUMBER_WORDS = {
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "fifteen", "twenty", "hundred"
}


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """You are a friendly math tutor for children in Grade 1 to 6.
Answer the child's latest question correctly using simple language.
If the question continues a situation from earlier in the conversation, use those facts.
Never invent numbers. Never repeat the question back. Build short 1 - 2 sentence responses while answering questions."""


# =========================================================
# NUMERIC STATE TRACKER
# =========================================================

class NumericStateTracker:
    """
    Tracks named quantities explicitly so the LLM never has to re-parse
    prose to find a running total.

    Example across turns:
        Turn 1: "I have 3 apples"        → apple = 3
        Turn 2: "teacher gave me 3 more" → apple = 6   (extracted from response)
        Turn 3: "dad gave me 5 more"     → prompt says [FACT] apple = 6
                                           so model computes 6+5=11, not 3+5=8
    """

    def __init__(self):
        self.quantities: dict[str, int] = {}

    def update_from_response(self, response: str):
        """Scan LLM response for the final computed total and store it."""
        patterns = [
            r"you have (?:a total of )?(\d+)\s+([a-z]+)",
            r"total of (\d+)\s+([a-z]+)",
            r"(\d+)\s+([a-z]+)\s+(?:now|total|altogether|in all)",
            r"=\s*(\d+)\s+([a-z]+)",
        ]
        for pattern in patterns:
            for count_str, item in re.findall(pattern, response.lower()):
                item_clean = item.rstrip("s") if len(item) > 3 else item
                try:
                    self.quantities[item_clean] = int(count_str)
                except ValueError:
                    continue

    def update_from_query(self, query: str):
        """Scan user message for initial quantity declarations."""
        pattern = (
            r"(?:have|got|found|picked up|start with)\s+"
            r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+([a-z]+)"
        )
        word_to_num = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
        }
        for count_str, item in re.findall(pattern, query.lower()):
            item_clean = item.rstrip("s") if len(item) > 3 else item
            if item_clean not in self.quantities:
                count = word_to_num.get(count_str) or (
                    int(count_str) if count_str.isdigit() else None
                )
                if count is not None:
                    self.quantities[item_clean] = count

    def get_context_string(self) -> str:
        """
        Returns state as a plain natural-language sentence.
        e.g. "The child currently has: 8 apples, 4 sticks."
        Gemma ignores [TAG] formats but reliably reads plain sentences.
        """
        if not self.quantities:
            return ""
        parts = ", ".join(f"{val} {item}s" for item, val in self.quantities.items())
        return f"The child currently has: {parts}."

    def reset(self):
        self.quantities.clear()


# =========================================================
# PROMPT ENGINE
# =========================================================

class PromptEngine:

    def build_user_turn(self, query: str, context_docs: list, state_context: str) -> str:
        """
        Assembles: [FACT numeric state] + [RAG background] + [Question]

        State is injected first so the model's attention anchors to the
        correct numbers before reading anything else.
        RAG background is a soft hint — framed as optional so the model
        reasons with it rather than copy-pasting it verbatim.
        """
        parts = []

        # State injected as plain sentence — NOT a [TAG] format.
        # Gemma ignores unfamiliar tags but reliably reads plain sentences.
        if state_context:
            parts.append(state_context)

        if context_docs:
            context_text = "\n".join(
                f"- {doc.page_content[:200]}" for doc in context_docs
            )
            parts.append(f"Background (use only if relevant):\n{context_text}")

        parts.append(f"Question: {query}")
        return "\n\n".join(parts)

    def build_user_turn_clean(self, query: str) -> str:
        """Clean version for history — no RAG blob, no state block."""
        return query


# =========================================================
# MAIN ENGINE
# =========================================================

class MathRAGEngine:

    def __init__(self):

        print(f"Loading embeddings ({EMBEDDING_MODEL})...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
            model_kwargs={"device": "cpu"},
        )

        print(f"Loading FAISS index from {INDEX_PATH}...")
        self.db = FAISS.load_local(
            INDEX_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

        self.prompt_engine = PromptEngine()
        self.state_tracker = NumericStateTracker()
        self.chat_history: list[dict] = []

        print(f"Engine ready. Model: {LLM_MODEL}\n")


    # =====================================================
    # QUERY CLASSIFIER
    # =====================================================

    def _is_declarative(self, query: str) -> bool:
        """
        Returns True if the input is a plain fact statement, not a question.
        RAG retrieval is SKIPPED for declarative inputs.

        WHY THIS MATTERS:
        "I have three apples" → FAISS retrieves a multiplication chunk
        (semantic overlap on words like "groups", "each", "altogether")
        → model reads that chunk and hallucinates "3 bags × 3 = 9 apples"
        even though the user just stated a simple fact.

        A declarative statement needs NO concept lookup. It just gets stored
        in numeric state and history, and the model acknowledges it simply.

        Detection rules (no extra LLM call needed):
          1. No question signals (how/what/why/which/when/where/?)
          2. Starts with a known declarative opener ("I have", "there are"...)
          3. Contains at least one number (digit or word)
        """
        q = query.strip().lower()

        # Rule 1: bail out immediately if it looks like a question
        question_signals = {"how", "what", "why", "which", "when", "where", "?"}
        if any(sig in q for sig in question_signals):
            return False

        # Rule 2: must start with a declarative opener
        if not any(q.startswith(op) for op in DECLARATIVE_OPENERS):
            return False

        # Rule 3: must contain a number
        has_digit = bool(re.search(r'\d', q))
        has_word_num = any(w in q.split() for w in NUMBER_WORDS)
        if not (has_digit or has_word_num):
            return False

        return True


    # =====================================================
    # VAGUE QUERY RESOLVER
    # =====================================================

    def _resolve_query(self, query: str) -> str:
        """
        Rewrites vague one-word replies ("yes", "ok", "more") into
        self-contained questions using the last exchange as context.
        """
        is_vague = (
            query.strip().lower() in VAGUE_TRIGGERS
            or len(query.strip().split()) <= 2
        )

        if not is_vague or not self.chat_history:
            return query

        last_user = next(
            (m["content"] for m in reversed(self.chat_history) if m["role"] == "user"),
            ""
        )
        last_assistant = next(
            (m["content"] for m in reversed(self.chat_history) if m["role"] == "assistant"),
            ""
        )

        expand_messages = [
            {
                "role": "system",
                "content": (
                    "You rewrite vague follow-up messages into clear, self-contained "
                    "questions based on the conversation so far. "
                    "Output ONLY the rewritten question. No explanation, no preamble."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Previous question: {last_user}\n"
                    f"Previous answer: {last_assistant}\n"
                    f"Child's follow-up: \"{query}\"\n\n"
                    "Rewrite the follow-up as a clear, specific question:"
                )
            }
        ]

        try:
            result = ollama.chat(
                model=LLM_MODEL,
                messages=expand_messages,
                options={"temperature": 0.1, "num_predict": 60}
            )
            expanded = result["message"]["content"].strip()
            if expanded and len(expanded) < 200:
                return expanded
        except Exception:
            pass

        return query


    # =====================================================
    # MEMORY
    # =====================================================

    def _get_messages(self, user_turn_with_context: str) -> list[dict]:
        """[system] + [clean history] + [current turn with context + state]"""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        recent   = self.chat_history[-(MAX_HISTORY * 2):]
        messages.extend(recent)
        messages.append({"role": "user", "content": user_turn_with_context})
        return messages


    # =====================================================
    # LLM CALL
    # =====================================================

    def _llm(self, messages: list[dict]) -> str:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=messages,
            options={"temperature": 0.3, "num_predict": 256}
        )
        return response["message"]["content"].strip()


    # =====================================================
    # RETRIEVAL
    # =====================================================

    def _retrieve(self, query: str) -> list:
        """Concept chunks only — filters out example scenarios."""
        def concept_only(meta: dict) -> bool:
            return meta.get("type") != "example"
        try:
            return self.db.similarity_search(query, k=RAG_K, filter=concept_only)
        except Exception:
            return self.db.similarity_search(query, k=RAG_K)


    # =====================================================
    # MAIN ASK
    # =====================================================

    def ask(self, query: str) -> str:

        # 1. Expand vague queries ("yes", "ok") into real questions
        resolved_query = self._resolve_query(query)

        # 2. Update numeric state from the user's message
        self.state_tracker.update_from_query(resolved_query)

        # 3. Classify: statement vs question
        #    Declarative statements ("I have 3 apples") skip RAG entirely
        #    to prevent irrelevant chunks from causing hallucination
        is_statement = self._is_declarative(resolved_query)
        docs = [] if is_statement else self._retrieve(resolved_query)

        # 4. Build user turn: [FACT state] + [RAG context] + [question]
        user_turn_with_context = self.prompt_engine.build_user_turn(
            query=resolved_query,
            context_docs=docs,
            state_context=self.state_tracker.get_context_string(),
        )

        # 5. Assemble full messages array
        messages = self._get_messages(user_turn_with_context)

        # 6. Call LLM
        response = self._llm(messages)

        # 7. Extract new totals from response and update numeric state
        self.state_tracker.update_from_response(response)

        # 8. Store CLEAN query + response (no RAG blob — prevents hallucination)
        self.chat_history.append({
            "role": "user",
            "content": self.prompt_engine.build_user_turn_clean(resolved_query)
        })
        self.chat_history.append({"role": "assistant", "content": response})

        if len(self.chat_history) > MAX_HISTORY * 2:
            self.chat_history = self.chat_history[-(MAX_HISTORY * 2):]

        return response


    def reset(self):
        """Clears conversation and numeric state for a fresh session."""
        self.chat_history.clear()
        self.state_tracker.reset()


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    rag = MathRAGEngine()

    print("=" * 50)
    print("  Math Tutor — type 'exit' to quit, 'reset' to start over")
    print("=" * 50)

    while True:
        try:
            q = input("\nAsk: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not q:
            continue
        if q.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        if q.lower() == "reset":
            rag.reset()
            print("Session reset.\n")
            continue

        print(f"\n{rag.ask(q)}\n")