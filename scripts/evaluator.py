import matplotlib.pyplot as plt
import json
import numpy as np
from pathlib import Path
from rag_engine import MathRAGEngine

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# =========================================================
# NON-RAG BASELINE (DIRECT GEMMA)
# =========================================================

import ollama

class BaselineModel:
    def __init__(self, model="gemma:2b"):
        self.model = model

    def ask(self, query):
        res = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": query}]
        )
        return res["message"]["content"]


# =========================================================
# TEST CASES (GRADE 1–6 MIX)
# =========================================================

TEST_CASES = [
    ("Ariel has 2 stones and gets 3 more. How many now?", "addition"),
    ("I had 10 candies and gave 4 away. What is left?", "subtraction"),
    ("Share 12 candies among 3 kids.", "division"),
    ("What is half of 8?", "fractions"),
    ("What is value of 3 in 354?", "place value"),
    ("Which is bigger -3 or -1?", "integers"),
]


# =========================================================
# SIMPLE SCORING FUNCTION (HEURISTIC)
# =========================================================

def score_answer(answer: str, concept: str):

    answer = answer.lower()

    score = 0

    # correctness keywords
    concept_keywords = {
        "addition": ["plus", "+", "add", "total", "sum"],
        "subtraction": ["minus", "-", "left", "take away"],
        "division": ["divide", "share", "each"],
        "fractions": ["half", "quarter", "fraction"],
        "place value": ["hundreds", "tens", "ones"],
        "integers": ["negative", "below zero", "less than"],
    }

    if concept in concept_keywords:
        for kw in concept_keywords[concept]:
            if kw in answer:
                score += 1

    # reasoning depth bonus
    if "step" in answer or "first" in answer:
        score += 1

    # penalty for very short answers
    if len(answer) < 40:
        score -= 1

    return max(score, 0)


# =========================================================
# SINGLE TURN EVALUATION
# =========================================================

def evaluate_single_turn(rag, baseline):

    rag_scores = []
    base_scores = []

    print("\n🔍 SINGLE TURN EVALUATION\n")

    for q, concept in TEST_CASES:

        rag_ans = rag.ask(q)
        base_ans = baseline.ask(q)

        rag_score = score_answer(rag_ans, concept)
        base_score = score_answer(base_ans, concept)

        rag_scores.append(rag_score)
        base_scores.append(base_score)

        print(f"\nQ: {q}")
        print(f"[RAG]: {rag_score} | {rag_ans[:80]}...")
        print(f"[BASE]: {base_score} | {base_ans[:80]}...")

    return rag_scores, base_scores


# =========================================================
# MULTI-TURN TEST (CORE FEATURE)
# =========================================================

MULTI_TURN = [
    "I have 2 apples",
    "I got 3 more apples",
    "How many apples now?",
    "I gave 2 apples away",
    "How many left?"
]


def evaluate_multi_turn(rag):

    print("\n🧠 MULTI-TURN TEST (RAG ONLY)\n")

    memory = ""
    scores = []

    for i, q in enumerate(MULTI_TURN):

        query = memory + "\nUser: " + q

        ans = rag.ask(query)

        print(f"\nTurn {i+1}")
        print("Q:", q)
        print("A:", ans[:120])

        score = len(ans) / 100  # proxy for reasoning depth
        scores.append(score)

        memory += f"\nUser: {q}\nAssistant: {ans}"

    return scores


# =========================================================
# REPORT GENERATION
# =========================================================

def generate_report(rag_scores, base_scores, multi_scores):

    report = {
        "rag_avg": float(np.mean(rag_scores)),
        "baseline_avg": float(np.mean(base_scores)),
        "multi_turn_avg": float(np.mean(multi_scores)),
    }

    with open("evaluation_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print("\n📊 FINAL REPORT")
    print(json.dumps(report, indent=4))

    return report


# =========================================================
# CHART GENERATION
# =========================================================

def plot_results(rag_scores, base_scores, multi_scores):

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rag_plot_path = RESULTS_DIR / "rag_vs_baseline.png"
    multi_plot_path = RESULTS_DIR / "multi_turn.png"

    # 1. RAG vs BASELINE
    plt.figure()
    plt.plot(rag_scores, label="RAG")
    plt.plot(base_scores, label="Baseline")
    plt.title("RAG vs Baseline Performance")
    plt.xlabel("Test Case #")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(rag_plot_path)

    # 2. Multi-turn reasoning stability
    plt.figure()
    plt.plot(multi_scores, label="Multi-Turn RAG")
    plt.title("Multi-Turn Reasoning Stability")
    plt.xlabel("Conversation Turn")
    plt.ylabel("Reasoning Depth Score")
    plt.legend()
    plt.savefig(multi_plot_path)

    print(f"\n📈 Charts saved: {rag_plot_path}, {multi_plot_path}")


# =========================================================
# MAIN EXECUTION
# =========================================================

if __name__ == "__main__":

    print("\n===================================")
    print("🧪 RAG EVALUATION FRAMEWORK")
    print("===================================\n")

    rag = MathRAGEngine()
    baseline = BaselineModel()

    # 1. Single turn comparison
    rag_scores, base_scores = evaluate_single_turn(rag, baseline)

    # 2. Multi-turn reasoning
    multi_scores = evaluate_multi_turn(rag)

    # 3. Report
    report = generate_report(rag_scores, base_scores, multi_scores)

    # 4. Charts
    plot_results(rag_scores, base_scores, multi_scores)

    print("\n✅ Evaluation Complete\n")