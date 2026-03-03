def build_answer_prompt(
    question: str,
    chapter: str,
    section_title: str,
    chunks: list[dict],
) -> list[dict]:
    """
    Builds the Groq chat messages for the ANSWER generation call.

    Returns a list of message dicts in OpenAI/Groq format:
      [{"role": "system", ...}, {"role": "user", ...}]
    """
    # Format retrieved chunks as numbered context blocks
    context_blocks = ""
    for i, chunk in enumerate(chunks, 1):
        similarity_pct = int(chunk.get("similarity", 0) * 100)
        context_blocks += (
            f"[{i}] (relevance: {similarity_pct}%) "
            f"{chunk.get('chapter', '')} > {chunk.get('section_title', '')}\n"
            f"{chunk.get('text_content', '').strip()}\n\n"
        )

    system_prompt = (
        "You are an expert NEET Biology teacher with deep knowledge of NCERT textbooks.\n"
        "Your job is to answer the student's question using ONLY the provided NCERT context below.\n"
        "Rules:\n"
        "  1. Base your answer strictly on the NCERT context provided.\n"
        "  2. Do NOT introduce facts not present in the context.\n"
        "  3. If the context does not contain enough information, say: "
        "'The provided NCERT context does not fully address this question.'\n"
        "  4. Keep the answer concise, factually accurate, and at NEET exam level.\n"
        f"  5. The question belongs to the chapter: '{chapter}', topic: '{section_title}'."
    )

    user_prompt = (
        f"Question: {question}\n\n"
        f"NCERT Context:\n{context_blocks}"
        "Answer:"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]


def build_justification_prompt(
    question: str,
    answer: str,
    chunks: list[dict],
) -> list[dict]:
    """
    Builds the Groq chat messages for the JUSTIFICATION generation call.
    The LLM must explain WHY the answer is correct, citing context points.
    """
    context_blocks = "\n".join(
        f"[{i}] {c.get('text_content', '').strip()}"
        for i, c in enumerate(chunks, 1)
    )

    system_prompt = (
        "You are an expert NEET Biology teacher.\n"
        "Given the student's question, the answer, and NCERT context passages, "
        "provide a step-by-step biological justification for why the answer is correct.\n"
        "Rules:\n"
        "  1. Use ONLY the NCERT context to justify.\n"
        "  2. Number each reasoning step.\n"
        "  3. Be specific — reference biological terms and mechanisms.\n"
        "  4. Do NOT add facts not found in the context."
    )

    user_prompt = (
        f"Question: {question}\n\n"
        f"Answer Given: {answer}\n\n"
        f"NCERT Context:\n{context_blocks}\n\n"
        "Step-by-step justification:"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]


def build_verifier_prompt(
    question: str,
    answer: str,
    chunks: list[dict],
) -> list[dict]:
    """
    Builds the Groq chat messages for the VERIFIER call.
    The LLM must output YES or NO on whether the answer is supported by context.
    """
    context_blocks = "\n".join(
        f"[{i}] {c.get('text_content', '').strip()}"
        for i, c in enumerate(chunks, 1)
    )

    system_prompt = (
        "You are a strict NEET Biology answer verifier.\n"
        "Given a question, an answer, and NCERT context passages, "
        "determine if the answer is fully supported by the context.\n"
        "Respond with EXACTLY one of:\n"
        "  YES - if the answer is clearly supported by the NCERT context.\n"
        "  NO  - if the answer contains facts not present in the context, or contradicts it.\n"
        "Then on a new line, give a one-sentence explanation."
    )

    user_prompt = (
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        f"NCERT Context:\n{context_blocks}\n\n"
        "Verdict (YES/NO):"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
