import os
from groq import Groq
from dotenv import load_dotenv

# Load .env for local development (Railway injects env vars directly)
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(dotenv_path=os.path.join(_root, '.env'), override=False)
load_dotenv(dotenv_path=os.path.join(_root, 'pdf_to_embedding', '.env'), override=False)

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


def call_groq(messages: list[dict], temperature: float = 0.2, max_tokens: int = 512, max_retries: int = 5) -> str:
    """
    Makes a single Groq chat completion call with exponential backoff for rate limits.
    Returns the raw text response from the LLM.
    """
    import time
    client = get_groq_client()
    delay = 5.0
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "rate limit" in error_msg or "too many requests" in error_msg:
                if attempt == max_retries - 1:
                    print(f"[Groq API] Max retries reached. Failing.")
                    raise e
                print(f"[Groq API] Rate limit hit. Waiting {delay}s before attempt {attempt+2}/{max_retries}...")
                time.sleep(delay)
                delay *= 1.5  # Exponential backoff
            else:
                raise e


def generate_answer(messages: list[dict]) -> str:
    """
    Calls the Groq LLM to generate the main answer.
    Uses low temperature (0.2) for factual consistency.
    """
    return call_groq(messages, temperature=0.2, max_tokens=768)


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

def verify_answer_llm(messages: list[dict]) -> dict:
    """
    Calls Groq to act as a Faithfulness/NLI judge.
    Expects the prompt to dictate a YES/NO verdict followed by an explanation.
    """
    raw = call_groq(messages, temperature=0.1, max_tokens=150)
    lines = raw.strip().split("\n")
    
    first_line = lines[0].strip().upper()
    verified = "YES" in first_line and "NO" not in first_line
    
    explanation = " ".join(lines[1:]).strip() if len(lines) > 1 else first_line
    
    return {
        "verified": verified,
        "explanation": f"LLM Judge: {explanation}"
    }

