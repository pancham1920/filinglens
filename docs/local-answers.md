# Local answer model: section 3.3

FilingLens calls `qwen2.5:3b` through Ollama on `127.0.0.1:11434`.
The model runs locally without a paid API key. The model download is about
1.9 GB; running it also requires memory for inference and context.

## Setup on macOS

Install the Ollama application from https://ollama.com/download/mac and open it.
Alternatively, install the command-line service with Homebrew:

```bash
brew install ollama
brew services start ollama
```

Use either the application or the Homebrew service, so only one server runs.
Then download the model:

```bash
ollama pull qwen2.5:3b
ollama list
```

## Verify the Python client

From `backend`:

```bash
uv run --offline python -m app.llm
```

The output should report fiscal 2024 net income of $93,736 million.
The first request can take longer while the model loads. The client allows
180 seconds, uses a 4,096-token context window and caps generation at 256 tokens.
It bypasses environment proxy settings for this localhost request.

This smoke check supplies evidence directly. Retrieval context construction
and grounded answer integration are sections 3.4 and 3.5. Existing BGE
embeddings and PostgreSQL retrieval remain the evidence source.

If the service is unavailable, start Ollama. If the model is missing, run the
pull command above. No additional Python dependency is needed: `httpx` is
already declared in `backend/pyproject.toml`.
