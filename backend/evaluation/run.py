"""Run the evaluation questions against the local /ask API."""

import csv
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import httpx


EVALUATION_DIR = Path(__file__).resolve().parent
API_URL = "http://127.0.0.1:8001/ask"


def main() -> int:
    questions = json.loads(
        (EVALUATION_DIR / "questions.json").read_text()
    )
    results = []

    with httpx.Client(timeout=240, trust_env=False) as client:
        for case in questions:
            started = perf_counter()

            try:
                response = client.post(
                    API_URL,
                    json={
                        "question": case["question"],
                        **case["filters"],
                    },
                )
                response.raise_for_status()
                data = response.json()

            except httpx.HTTPError as error:
                # Record the failure and continue with the other questions.
                data = {
                    "status": "error",
                    "answer": str(error),
                    "sources": [],
                }

                if isinstance(error, httpx.HTTPStatusError):
                    data["answer"] = error.response.text

            seconds = round(perf_counter() - started, 2)

            results.append({
                "case": case,
                "response": data,
                "seconds": seconds,
            })

            print(
                f"{case['id']}: {data['status']} "
                f"({seconds:.2f}s)",
                flush=True,
            )

    # Keep complete source text so citations can be checked afterward.
    results_path = EVALUATION_DIR / "results.json"
    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Leave judgement columns blank for manual review.
    review_path = EVALUATION_DIR / "review.csv"
    fields = [
        "id",
        "question",
        "expected_answer",
        "status",
        "actual_answer",
        "seconds",
        "correctness",
        "citation_support",
        "notes",
    ]

    with review_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for result in results:
            case = result["case"]
            data = result["response"]

            writer.writerow({
                "id": case["id"],
                "question": case["question"],
                "expected_answer": case["expected_answer"],
                "status": data["status"],
                "actual_answer": data["answer"],
                "seconds": result["seconds"],
                "correctness": "",
                "citation_support": "",
                "notes": "",
            })

    answered_times = [
        result["seconds"]
        for result in results
        if result["response"]["status"] == "answered"
    ]
    errors = sum(
        result["response"]["status"] == "error"
        for result in results
    )

    print(f"\nQuestions run: {len(results)}")
    print(f"Request errors: {errors}")

    if answered_times:
        print(
            "Median answered-request time: "
            f"{median(answered_times):.2f}s"
        )
        print(f"Slowest answered request: {max(answered_times):.2f}s")

    print("Saved:", results_path)
    print("Review:", review_path)

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())