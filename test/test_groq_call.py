import os
from groq import Groq
from dotenv import load_dotenv
import certifi
import httpx


def main() -> None:
    load_dotenv()

    # Force Python HTTP clients to use certifi CA bundle.
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Set GROQ_API_KEY in your environment before running this script.")

    ssl_verify = os.getenv("GROQ_SSL_VERIFY", "true").strip().lower() not in {"0", "false", "no"}
    client = Groq(api_key=api_key, http_client=httpx.Client(verify=ssl_verify, timeout=30.0))

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "user", "content": "what is JCL"}
        ]
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
