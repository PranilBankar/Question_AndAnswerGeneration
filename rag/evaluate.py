"""
evaluate.py — RAG Pipeline Evaluation Script

Runs all questions from eval_data.json through the full pipeline and computes:
  - Chapter Classification Accuracy
  - Recall@5     (graded: 0.0 / 0.7 / 1.0)
  - Precision@3  (top-3 retrieved chunks vs. expected topics)
  - Faithfulness / Verification Rate
  - Average RAG Confidence
  - Rejection Rate  (BERT confidence too low)
  - End-to-End Latency

Results are saved to rag/eval_results/evaluation_results.csv and 5 PNG graphs.
"""

import json
import time
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from rag.pipeline import run_pipeline

# ─── Visualization style ────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (10, 6)

# ─── Paths ───────────────────────────────────────────────────────────────────
DATA_FILE   = os.path.join(os.path.dirname(__file__), "eval_data.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "eval_results")
GRAPHS_DIR  = os.path.join(RESULTS_DIR, "graphs")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR,  exist_ok=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

import re

def normalize(text: str) -> str:
    """Strict normalization: lowercase, strip, replace hyphens/underscores with spaces."""
    # Remove leading numbers and decorators (e.g. "9.8.6 - ")
    text = re.sub(r'^[\d\.\s\-]+', '', text)
    return text.lower().strip().replace("-", " ").replace("_", " ")


def is_hit(retrieved_topic: str, expected_topics: list[str]) -> bool:
    """
    Strict normalized exact match.
    A retrieved topic is a hit only if its normalized form exactly equals
    a normalized expected topic.
    """
    rt = normalize(retrieved_topic)
    for et in expected_topics:
        if rt == normalize(et):
            return True
    return False


def graded_recall(hits: list) -> float:
    """
    Graded Recall@5:
      0 hits  → 0.0
      1 hit   → 0.7
      2+ hits → 1.0
    """
    n = len(hits)
    if n == 0:
        return 0.0
    elif n == 1:
        return 0.7
    else:
        return 1.0


# ─── Main evaluation ─────────────────────────────────────────────────────────

def evaluate_pipeline():
    with open(DATA_FILE, "r") as f:
        eval_data = json.load(f)

    results = []
    print(f"Starting evaluation of {len(eval_data)} questions...\n")

    for item in tqdm(eval_data, desc="Evaluating RAG Pipeline"):
        question        = item["question"]
        expected_chapter = item["expected_chapter"]
        expected_topics  = item.get("expected_topics", [])

        start_time = time.time()
        res        = run_pipeline(question, top_k=5, use_classifier=False)
        latency    = time.time() - start_time

        # ── Rejection check ──────────────────────────────────────────────────
        rejected = res.get("rejected", False)

        # ── Chapter classification accuracy ──────────────────────────────────
        cls = res.get("classification")
        predicted_chapter = cls.get("chapter", "") if cls else ""
        chapter_correct   = (
            normalize(predicted_chapter) == normalize(expected_chapter)
            if predicted_chapter else False
        )

        if rejected:
            # Don't compute retrieval metrics for rejected samples
            print(f"  [REJECTED] Q{item['id']}: '{question[:60]}'")
            results.append({
                "id"               : item["id"],
                "question"         : question,
                "expected_chapter" : expected_chapter,
                "predicted_chapter": predicted_chapter,
                "chapter_correct"  : chapter_correct,
                "recall_at_5"      : None,
                "precision_at_3"   : None,
                "avg_similarity"   : None,
                "is_verified"      : False,
                "confidence"       : 0.0,
                "latency_sec"      : latency,
                "rejected"         : True,
            })
            continue

        # ── Retrieval: collect section_titles from sources ───────────────────
        raw_sources = res.get("sources", [])
        retrieved_topics = []
        for src in raw_sources:
            title = src.get("section_title", "")
            if title and title.strip():
                retrieved_topics.append(title)
            else:
                print(f"  [WARN] Q{item['id']}: Retrieved chunk has no section_title — skipping.")

        # ── Recall@5 (graded) ────────────────────────────────────────────────
        hits   = [t for t in retrieved_topics if is_hit(t, expected_topics)]
        recall = graded_recall(hits)

        # ── Precision@3 ──────────────────────────────────────────────────────
        # Only top-3 retrieved chunks vs. up to 3 expected topics
        top3_retrieved = retrieved_topics[:3]
        top3_hits      = [t for t in top3_retrieved if is_hit(t, expected_topics)]
        precision      = len(top3_hits) / len(top3_retrieved) if top3_retrieved else 0.0

        # ── Average vector similarity ─────────────────────────────────────────
        similarities   = [s.get("similarity", 0.0) for s in raw_sources]
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

        # ── Faithfulness & confidence ─────────────────────────────────────────
        is_verified = res.get("verified", False)
        confidence  = res.get("confidence", 0.0)

        results.append({
            "id"               : item["id"],
            "question"         : question,
            "expected_chapter" : expected_chapter,
            "predicted_chapter": predicted_chapter,
            "chapter_correct"  : chapter_correct,
            "recall_at_5"      : recall,
            "precision_at_3"   : precision,
            "avg_similarity"   : avg_similarity,
            "is_verified"      : is_verified,
            "confidence"       : confidence,
            "latency_sec"      : latency,
            "rejected"         : False,
        })

    # ── Save raw results ──────────────────────────────────────────────────────
    df       = pd.DataFrame(results)
    csv_path = os.path.join(RESULTS_DIR, "evaluation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Evaluation complete. Results saved to {csv_path}")

    # ── Summary metrics ───────────────────────────────────────────────────────
    non_rejected = df[df["rejected"] == False]
    total        = len(df)
    rejected_n   = df["rejected"].sum()

    print("\n" + "=" * 55)
    print("           EVALUATION SUMMARY")
    print("=" * 55)
    print(f"  Total Questions      : {total}")
    print(f"  Rejected (low conf.) : {rejected_n} ({rejected_n/total:.1%})")
    print(f"  Chapter Accuracy     : {non_rejected['chapter_correct'].mean():.1%}")
    print(f"  Recall@5  (graded)   : {non_rejected['recall_at_5'].mean():.3f}")
    print(f"  Precision@3          : {non_rejected['precision_at_3'].mean():.3f}")
    print(f"  Verification Rate    : {non_rejected['is_verified'].mean():.1%}")
    print(f"  Avg Confidence       : {non_rejected['confidence'].mean():.3f}")
    print(f"  Avg Latency          : {non_rejected['latency_sec'].mean():.1f}s")
    print("=" * 55)

    # ── Generate graphs ───────────────────────────────────────────────────────
    print("\nGenerating evaluation graphs...")
    generate_graphs(df, non_rejected)


# ─── Graph generation ─────────────────────────────────────────────────────────

def generate_graphs(df: pd.DataFrame, non_rejected: pd.DataFrame):
    colors = sns.color_palette("husl", 8)

    # 1. Retrieval Effectiveness (Recall@5 and Precision@3)
    plt.figure()
    metrics_df = pd.DataFrame({
        "Metric"        : ["Recall@5", "Precision@3"],
        "Average Score" : [
            non_rejected["recall_at_5"].mean(),
            non_rejected["precision_at_3"].mean(),
        ],
    })
    ax = sns.barplot(x="Metric", y="Average Score", data=metrics_df, palette="viridis")
    plt.title("Retrieval Effectiveness", pad=20)
    plt.ylim(0, 1.05)
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2%}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center", va="center", xytext=(0, 10),
            textcoords="offset points", fontweight="bold",
        )
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, "1_retrieval_effectiveness.png"), dpi=300)
    plt.close()

    # 2. Verification Rate (Faithfulness)
    plt.figure(figsize=(8, 8))
    verified_counts = non_rejected["is_verified"].value_counts()
    plt.pie(
        verified_counts,
        labels=[f"{bool(k)} ({v})" for k, v in verified_counts.items()],
        autopct="%1.1f%%",
        colors=["#2ecc71", "#e74c3c"],
        startangle=90,
        textprops={"fontsize": 14, "weight": "bold"},
    )
    plt.title("Faithfulness: LLM Answer Verification Rate", pad=20)
    plt.savefig(os.path.join(GRAPHS_DIR, "2_verification_rate.png"), dpi=300)
    plt.close()

    # 3. Overall Pipeline Confidence Distribution
    plt.figure()
    sns.histplot(non_rejected["confidence"], bins=10, kde=True, color=colors[4])
    plt.title("Distribution of Final RAG Confidence Scores", pad=20)
    plt.xlabel("Confidence Score (0.0 to 1.0)")
    plt.ylabel("Number of Questions")
    plt.xlim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, "3_confidence_distribution.png"), dpi=300)
    plt.close()

    # 4. Latency Distribution
    plt.figure()
    sns.boxplot(y=non_rejected["latency_sec"], color=colors[2])
    sns.stripplot(y=non_rejected["latency_sec"], color="black", alpha=0.5, jitter=True)
    plt.title("Pipeline End-to-End Latency", pad=20)
    plt.ylabel("Latency (Seconds)")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, "4_latency_distribution.png"), dpi=300)
    plt.close()

    # 5. Average Retrieval Similarity
    plt.figure()
    sns.histplot(non_rejected["avg_similarity"], bins=10, kde=True, color=colors[0])
    plt.title("Average Vector Similarity of Retrieved Chunks", pad=20)
    plt.xlabel("Cosine Similarity Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, "5_retrieval_similarity.png"), dpi=300)
    plt.close()

    # 6. Chapter Classification Accuracy (new)
    plt.figure(figsize=(8, 8))
    acc_counts = non_rejected["chapter_correct"].value_counts()
    plt.pie(
        acc_counts,
        labels=[f"{'Correct' if bool(k) else 'Wrong'} ({v})" for k, v in acc_counts.items()],
        autopct="%1.1f%%",
        colors=["#3498db", "#e67e22"],
        startangle=90,
        textprops={"fontsize": 14, "weight": "bold"},
    )
    plt.title("Chapter Classification Accuracy (BERT)", pad=20)
    plt.savefig(os.path.join(GRAPHS_DIR, "6_chapter_accuracy.png"), dpi=300)
    plt.close()

    print(f"✅ Generated 6 graphs in {GRAPHS_DIR}")


if __name__ == "__main__":
    evaluate_pipeline()
