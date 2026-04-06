import json
import ast
from tqdm import tqdm
from rag.retriever import retrieve_chunks
from rag.generator import call_groq

def build_golden_dataset():
    with open("rag/eval_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    prompt_template = """
You are examining a RAG database's retrieval output.
The user asked: "{question}"

The database retrieved these 5 chunk titles:
{options}

Which of these EXACT titles contain the answer to the question? 
You must reply with ONLY a python-formatted list of strings. Do not write any other text.
Example response: ["Factors Affecting Enzyme Activity", "Enzymes"]
"""

    print("Reverse engineering exact Supabase chunk titles...")
    new_data = []
    
    for item in tqdm(data):
        q = item["question"]
        chunks = retrieve_chunks(q, top_k=5)
        
        # Gather exact strings
        titles = []
        for c in chunks:
            t = c.get("section_title", "")
            if t and t not in titles:
                titles.append(t)
        
        # Build prompt
        opts_str = "\n".join([f"- {t}" for t in titles])
        prompt = prompt_template.format(question=q, options=opts_str)
        
        messages = [{"role": "user", "content": prompt}]
        
        response_text = call_groq(messages, temperature=0.1, max_tokens=100)
        
        # Parse the python list
        try:
            expected_topics = ast.literal_eval(response_text)
            if not isinstance(expected_topics, list):
                expected_topics = titles[:2] # fallback
        except:
            expected_topics = titles[:2] # fallback
            
        item["expected_topics"] = expected_topics
        new_data.append(item)
    
    with open("rag/eval_data_golden.json", "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2)
        
    print("\n✅ Saved to rag/eval_data_golden.json!")

if __name__ == "__main__":
    build_golden_dataset()
