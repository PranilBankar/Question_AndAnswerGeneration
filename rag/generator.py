import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', 'pdf_to_embedding', '.env'))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

_groq_client: Groq = None


def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise ValueError("Missing GROQ_API_KEY in .env file.")
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


def call_groq(messages: list[dict], temperature: float = 0.2, max_tokens: int = 512) -> str:
    """
    Makes a single Groq chat completion call.
    Returns the raw text response from the LLM.
    """
    client = get_groq_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def generate_answer(messages: list[dict]) -> str:
    """
    Calls the Groq LLM to generate the main answer.
    Uses low temperature (0.2) for factual consistency.
    """
    return call_groq(messages, temperature=0.2, max_tokens=512)


def generate_justification(messages: list[dict]) -> list[str]:
    """
    Calls the Groq LLM to generate a step-by-step justification.
    Parses numbered lines into a list of strings.
    Returns a list like: ["Step 1: ...", "Step 2: ..."]
    """
    raw = call_groq(messages, temperature=0.3, max_tokens=512)

    lines = raw.strip().split("\n")
    steps = [l.strip() for l in lines if l.strip() and (
        l.strip()[0].isdigit() or l.strip().lower().startswith("step")
    )]

    # Fallback: if no numbered lines, return original text split by newlines
    return steps if steps else [l.strip() for l in lines if l.strip()]


def verify_answer(messages: list[dict]) -> dict:
    """
    Calls the Groq LLM to verify if the answer is supported by the NCERT context.
    Returns: {"verified": bool, "explanation": str}
    """
    raw = call_groq(messages, temperature=0.0, max_tokens=100)

    lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
    verdict_line = lines[0].upper() if lines else ""
    explanation  = lines[1] if len(lines) > 1 else ""

    verified = verdict_line.startswith("YES")
    return {"verified": verified, "explanation": explanation}
