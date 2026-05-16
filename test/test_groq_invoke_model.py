import os
#import httpx
from groq import Groq
from dotenv import load_dotenv

#"llama-3.1-8b-instant"
# 1. Define model + behavior
MODEL_NAME =  "meta-llama/llama-4-scout-17b-16e-instruct"
SYSTEM_PROMPT = "Answer concisely"
# groq/compound llama-3.1-8b-instant


def build_client() -> Groq:
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Set GROQ_API_KEY in .env or environment variables.")

    ssl_verify = os.getenv("GROQ_SSL_VERIFY", "true").strip().lower() not in {"0", "false", "no"}
    #http_client = httpx.Client(verify=ssl_verify, timeout=30.0)
    return Groq(api_key=api_key, http_client=None)


client = build_client()


# 2. Invoke model with prompt
def invoke_model(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=1000
    )
    return response.choices[0].message.content.strip()


# --- Run ---
question = """
What is AgenticAI?
"""

answer = invoke_model(question)
print("\nAnswer:")
print(answer)
