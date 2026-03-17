import json
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from rag.pipeline import run_pipeline

# Configure visualization style
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['figure.figsize'] = (10, 6)

DATA_FILE = os.path.join(os.path.dirname(__file__), 'eval_data.json')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'eval_results')
GRAPHS_DIR = os.path.join(RESULTS_DIR, 'graphs')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)

def evaluate_pipeline():
    with open(DATA_FILE, 'r') as f:
        eval_data = json.load(f)

    results = []
    
    print(f"Starting evaluation of {len(eval_data)} questions...\n")
    
    for item in tqdm(eval_data, desc="Evaluating RAG Pipeline"):
        question = item["question"]
        expected_chapter = item["expected_chapter"]
        expected_topics = item.get("expected_topics", [])
        
        start_time = time.time()
        
        # Run the full pipeline (including BERT classifier)
        res = run_pipeline(question, top_k=5, use_classifier=True)
        
        latency = time.time() - start_time
        
        # === Compute Metrics ===
        
        # 1. Retrieval Metrics (Precision@K and Recall@K)
        # Check the 'sources' list for the expected topics
        retrieved_topics = [src.get("section_title", "") for src in res.get("sources", [])]
        
        # A retrieved chunk is a "hit" if its section_title matches any of the expected topics
        def is_hit(retrieved_topic, exp_topics):
            rt = retrieved_topic.lower().strip()
            for et in exp_topics:
                et_lower = et.lower().strip()
                if rt == et_lower or et_lower in rt or rt in et_lower:
                    return True
            return False
            
        hits = [t for t in retrieved_topics if is_hit(t, expected_topics)]
        
        # Recall@K: Did AT LEAST ONE of the expected topics appear in the top K?
        recall = 1.0 if len(hits) > 0 else 0.0
        
        # Precision@K: What proportion of the retrieved chunks belong to the expected topics?
        if len(retrieved_topics) > 0:
            precision = len(hits) / len(retrieved_topics)
        else:
            precision = 0.0
            
        # Average vector similarity
        similarities = [src.get("similarity", 0.0) for src in res.get("sources", [])]
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        
        # 2. Faithfulness / Verification
        is_verified = res.get("verified", False)
        
        # 3. Overall Confidence
        confidence = res.get("confidence", 0.0)
        
        results.append({
            "id": item["id"],
            "question": question,
            "expected_chapter": expected_chapter,
            "predicted_chapter": res.get("classification", {}).get("chapter", "None") if res.get("classification") else "None",
            "recall_at_5": recall,
            "precision_at_5": precision,
            "avg_similarity": avg_similarity,
            "is_verified": is_verified,
            "confidence": confidence,
            "latency_sec": latency
        })
        
        # Optional: Print a brief summary per question
        # print(f"  Q{item['id']}: Ver: {is_verified} | Conf: {confidence:.2f} | Latency: {latency:.2f}s")

    # === Save Raw Results ===
    df = pd.DataFrame(results)
    csv_path = os.path.join(RESULTS_DIR, 'evaluation_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Evaluation complete. Results saved to {csv_path}")
    
    # === Generate Graphs ===
    print("Generating evaluation graphs...")
    generate_graphs(df)

def generate_graphs(df):
    colors = sns.color_palette("husl", 8)
    
    # 1. Retrieval Effectiveness (Recall and Precision)
    plt.figure()
    metrics_df = pd.DataFrame({
        "Metric": ["Recall@5", "Precision@5"],
        "Average Score": [df["recall_at_5"].mean(), df["precision_at_5"].mean()]
    })
    ax = sns.barplot(x="Metric", y="Average Score", data=metrics_df, palette="viridis")
    plt.title("Retrieval Effectiveness (Top 5 Chunks)", pad=20)
    plt.ylim(0, 1.05)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2%}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, "1_retrieval_effectiveness.png"), dpi=300)
    plt.close()

    # 2. Verification Rate (Faithfulness)
    plt.figure(figsize=(8, 8))
    verified_counts = df["is_verified"].value_counts()
    plt.pie(verified_counts, labels=[f"{bool(k)} ({v})" for k, v in verified_counts.items()], 
            autopct='%1.1f%%', colors=["#2ecc71", "#e74c3c"], startangle=90, textprops={'fontsize': 14, 'weight': 'bold'})
    plt.title("Faithfulness: LLM Answer Verification Rate", pad=20)
    plt.savefig(os.path.join(GRAPHS_DIR, "2_verification_rate.png"), dpi=300)
    plt.close()

    # 3. Overall Pipeline Confidence Distribution
    plt.figure()
    sns.histplot(df["confidence"], bins=10, kde=True, color=colors[4])
    plt.title("Distribution of Final RAG Confidence Scores", pad=20)
    plt.xlabel("Confidence Score (0.0 to 1.0)")
    plt.ylabel("Number of Questions")
    plt.xlim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, "3_confidence_distribution.png"), dpi=300)
    plt.close()

    # 4. Latency Distribution
    plt.figure()
    sns.boxplot(y=df["latency_sec"], color=colors[2])
    sns.stripplot(y=df["latency_sec"], color="black", alpha=0.5, jitter=True)
    plt.title("Pipeline End-to-End Latency", pad=20)
    plt.ylabel("Latency (Seconds)")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, "4_latency_distribution.png"), dpi=300)
    plt.close()

    # 5. Average Retrieval Similarity
    plt.figure()
    sns.histplot(df["avg_similarity"], bins=10, kde=True, color=colors[0])
    plt.title("Average Vector Similarity of Retrieved Chunks", pad=20)
    plt.xlabel("Cosine Similarity Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, "5_retrieval_similarity.png"), dpi=300)
    plt.close()

    print(f"✅ Generated 5 graphs in {GRAPHS_DIR}")

if __name__ == "__main__":
    evaluate_pipeline()
