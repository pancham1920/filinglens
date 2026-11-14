"""Call the local answer-generation model through Ollama."""

import httpx


MODEL = "qwen2.5:3b"
# Ollama runs the model locally and exposes this HTTP endpoint.
GENERATE_URL = "http://127.0.0.1:11434/api/generate"


def generate_text(prompt: str) -> str:
    """Send a prompt to Ollama and return the generated answer text."""
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    try:
        response = httpx.post(
            GENERATE_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,  # Receive one complete JSON response.
                "options": {
                    "temperature": 0,  # Reduce randomness in the answer.
                    "num_ctx": 8192,  # Token window for input and output.
                    "num_predict": 256,  # Maximum tokens to generate.
                },
            },
            timeout=180,  # Allow time for model loading and generation.
            trust_env=False,  # Bypass proxy settings for this local request.
        )
        # Turn HTTP errors, such as a missing model, into exceptions.
        response.raise_for_status()
    except httpx.RequestError as error:
        # Includes connection failures and request timeouts.
        raise RuntimeError(
            "Could not reach Ollama or generation timed out. "
            "Check that Ollama is running."
        ) from error
    except httpx.HTTPStatusError as error:
        raise RuntimeError(
            f"Ollama returned HTTP {error.response.status_code}. "
            f"Check that {MODEL} is installed."
        ) from error

    # Ollama includes timing and model metadata; callers only need the text.
    data = response.json()

    # A response stopped at the output limit may be incomplete.
    if data.get("done_reason") == "length":
        raise RuntimeError("The model reached its output limit")

    answer = data["response"].strip()

    if not answer:
        raise RuntimeError("The model returned an empty answer")

    return answer


def main() -> int:
    # Smoke check with fixed evidence. Retrieval is connected in later steps.
    prompt = """
Answer the question using only the evidence below.
Keep the answer to one sentence and preserve the units.

Evidence:
Apple's consolidated statement reports net income of
$93,736 million for the year ended September 28, 2024.

Question:
What was Apple's net income in fiscal 2024?
"""

    print("Model:", MODEL)
    try:
        answer = generate_text(prompt)
    except (ValueError, RuntimeError) as error:
        print("FAIL:", error)
        return 1

    print("Answer:", answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
