"""This module runs the full RAG pipeline from retrieval to grounded generation.
It keeps the orchestration thin so each stage stays easy to explain and test."""

from __future__ import annotations

import argparse
import logging
from typing import Any

from src.generator import generate
from src.retriever import search

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure logging for command-line use."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for manual pipeline testing."""
    parser = argparse.ArgumentParser(description="Run the CyberSecRAG pipeline.")
    parser.add_argument("--query", required=True, help="User question to send through the pipeline.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of CVE records to retrieve before generation.",
    )
    return parser.parse_args()


def query(user_query: str, top_k: int = 5) -> dict[str, Any]:
    """Run the full RAG pipeline: embed query → retrieve CVEs → generate answer."""
    LOGGER.info("Running pipeline for query with top_k=%s.", top_k)

    # Retrieve the most relevant CVEs first so the generator only sees grounded context.
    retrieved_cves = search(user_query, top_k=top_k)

    # Pass the retrieved evidence into the generator to keep the answer tied to source records.
    answer = generate(user_query, retrieved_cves)

    return {
        "query": user_query,
        "retrieved_cves": retrieved_cves,
        "answer": answer,
    }


def display_pipeline_result(result: dict[str, Any]) -> None:
    """Print the pipeline response and the supporting CVEs for CLI use."""
    print(result["answer"])
    print("-" * 60)
    print("Retrieved CVEs:")

    for retrieved_cve in result["retrieved_cves"]:
        print(
            f"{retrieved_cve.get('year')} | "
            f"{retrieved_cve.get('cve_id')} | "
            f"{retrieved_cve.get('severity')} | "
            f"CVSS {retrieved_cve.get('cvss')} | "
            f"Score {retrieved_cve.get('score'):.4f} | "
            f"{retrieved_cve.get('match_reason')}"
        )


def main() -> None:
    """Run the pipeline CLI for manual testing."""
    configure_logging()
    args = parse_args()
    result = query(args.query, top_k=args.top_k)
    display_pipeline_result(result)


if __name__ == "__main__":
    main()
