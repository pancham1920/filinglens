"""Prepare retrieved filing chunks as numbered evidence."""


def build_context(
    results: list[dict],
    max_chars: int = 8000,
) -> tuple[str, list[dict]]:
    if max_chars <= 0:
        raise ValueError("Context character limit must be positive")

    blocks = []
    sources = []
    used_chars = 0

    for result in results:
        citation_id = len(sources) + 1

        block = (
            f"[{citation_id}]\n"
            f"Company: {result['company']} ({result['ticker']})\n"
            f"Filing fiscal year: {result['fiscal_year']}\n"
            f"Section: {result['section']}\n"
            f"Content type: {result['content_type']}\n"
            f"Evidence:\n{result['text']}"
        )

        # Count the separator between evidence blocks too.
        added_chars = len(block) + (2 if blocks else 0)

        # Keep each chunk intact so table rows and units stay together.
        if used_chars + added_chars > max_chars:
            break

        blocks.append(block)
        used_chars += added_chars

        # Preserve the original result for citations and the evidence panel.
        sources.append({
            **result,
            "citation_id": citation_id,
        })

    return "\n\n".join(blocks), sources