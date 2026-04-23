"""This module loads retrieval assets once and serves semantic search queries.
It enriches FAISS hits with full CVE details loaded from JSONL files in the data folder."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

LOGGER = logging.getLogger(__name__)
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64
EMBEDDING_DIMENSION = 384
NOT_AVAILABLE = "Not available"
# Resolve the project root from this file's location so paths work from notebooks, CLI, and Flask.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_PATH = PROJECT_ROOT / "index" / "faiss_combined.index"
METADATA_PATH = PROJECT_ROOT / "index" / "metadata_combined.json"
_retriever: Retriever | None = None
STOP_WORDS = {
    "about",
    "affected",
    "applications",
    "ask",
    "critical",
    "high",
    "in",
    "kernel",
    "low",
    "medium",
    "remote",
    "the",
    "vulnerabilities",
    "vulnerability",
    "web",
}
INTENT_PATTERNS = {
    "remote code execution": [
        "remote code execution",
        "rce",
        "arbitrary code execution",
        "code execution",
    ],
    "sql injection": ["sql injection", "sqli"],
    "buffer overflow": ["buffer overflow", "stack overflow", "heap overflow"],
    "authentication bypass": [
        "authentication bypass",
        "auth bypass",
        "bypass authentication",
        "improper authentication",
    ],
}
SEVERITY_LEVELS = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
MIN_BASE_SCORE = 0.38
MIN_REFINED_SCORE = 0.48
MIN_CONSTRAINED_SCORE = 0.50


def configure_logging() -> None:
    """Configure logging for command-line and import-time use."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for manual search testing."""
    parser = argparse.ArgumentParser(description="Search the combined CVE FAISS index.")
    parser.add_argument("--query", required=True, help="Natural-language query to search for.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many nearest CVE matches to return.",
    )
    return parser.parse_args()


def normalize_text_value(value: Any, fallback: str) -> str:
    """Convert empty values into a readable fallback string."""
    if value is None:
        return fallback

    text_value = str(value).strip()
    return text_value or fallback


def normalize_severity(value: Any) -> str:
    """Normalize severity values so filtering works reliably."""
    return normalize_text_value(value, "UNKNOWN").upper()


def format_affected_products(value: Any) -> str:
    """Convert affected product data into one readable string."""
    if value is None:
        return NOT_AVAILABLE

    # The processed data stores affected products as a list, so join it for display.
    cleaned_items = [str(item).strip() for item in value if str(item).strip()]
    return ", ".join(cleaned_items) if cleaned_items else NOT_AVAILABLE


def parse_query_constraints(query: str) -> dict[str, Any]:
    """Extract simple year, severity, phrase, and keyword hints from the query."""
    lowered_query = query.lower()
    year_match = re.search(r"\b(20\d{2})\b", lowered_query)
    year = int(year_match.group(1)) if year_match else None

    # Pick up explicit severity words so we can prefer matching CVEs later.
    severities = {level for level in SEVERITY_LEVELS if level.lower() in lowered_query}
    phrases = [
        phrase
        for phrase, aliases in INTENT_PATTERNS.items()
        if any(alias in lowered_query for alias in aliases)
    ]

    # Keep a few non-trivial keywords to help rerank products and descriptions.
    keywords = [
        token
        for token in re.findall(r"[a-z0-9]+", lowered_query)
        if len(token) >= 4 and token not in STOP_WORDS and not token.isdigit()
    ]
    return {"year": year, "severities": severities, "phrases": phrases, "keywords": keywords}


def build_search_blob(result: dict[str, Any]) -> str:
    """Create one lowercase text blob used for lightweight reranking rules."""
    affected_products = result.get("affected_products")
    if isinstance(affected_products, list):
        affected_text = " ".join(affected_products)
    else:
        affected_text = str(affected_products)

    searchable_parts = [
        result.get("cve_id", ""),
        result.get("description", ""),
        result.get("severity", ""),
        result.get("problem_type", ""),
        result.get("cwe_id", ""),
        affected_text,
        str(result.get("year", "")),
    ]
    return " ".join(str(part) for part in searchable_parts).lower()


def collect_match_reasons(result: dict[str, Any], constraints: dict[str, Any]) -> list[str]:
    """Explain why a result matched, using the parsed query constraints."""
    reasons: list[str] = []
    blob = build_search_blob(result)
    year = constraints["year"]

    if year is not None and result.get("year") == year:
        reasons.append(f"year {year}")

    severity = normalize_severity(result.get("severity"))
    if constraints["severities"] and severity in constraints["severities"]:
        reasons.append(f"severity {severity}")

    for phrase in constraints["phrases"]:
        if any(alias in blob for alias in INTENT_PATTERNS[phrase]):
            reasons.append(f"{phrase} terms")

    keyword_hits = [keyword for keyword in constraints["keywords"] if keyword in blob]
    if keyword_hits:
        reasons.append(f"keywords: {', '.join(keyword_hits[:3])}")

    if not reasons:
        reasons.append("semantic similarity")

    return reasons


def score_result(result: dict[str, Any], constraints: dict[str, Any]) -> float:
    """Combine FAISS similarity with lightweight query-aware bonuses."""
    score = float(result["score"])
    blob = build_search_blob(result)
    year = constraints["year"]
    severity = normalize_severity(result.get("severity"))

    # Reward exact year matches because demo questions often ask for a specific year.
    if year is not None:
        score += 0.20 if result.get("year") == year else -0.15

    # Reward severity matches, but do not hard-fail if severity is missing.
    if constraints["severities"]:
        if severity in constraints["severities"]:
            score += 0.12
        elif severity != "UNKNOWN":
            score -= 0.04

    # Reward attack-type phrases such as RCE or SQL injection when they appear in the record.
    for phrase in constraints["phrases"]:
        if any(alias in blob for alias in INTENT_PATTERNS[phrase]):
            score += 0.10

    # Reward a few matching keywords to make products like Apache or Linux rank better.
    keyword_hits = sum(1 for keyword in constraints["keywords"] if keyword in blob)
    score += min(keyword_hits, 4) * 0.02
    return score


def has_required_constraints(result: dict[str, Any], constraints: dict[str, Any]) -> bool:
    """Check whether a result satisfies the strongest explicit user constraints."""
    year = constraints["year"]
    if year is not None and result.get("year") != year:
        return False

    severities = constraints["severities"]
    if severities and normalize_severity(result.get("severity")) not in severities:
        return False

    return True


def count_match_signals(result: dict[str, Any], constraints: dict[str, Any]) -> int:
    """Count how many meaningful query signals a result satisfies."""
    signals = 0
    blob = build_search_blob(result)

    if constraints["year"] is not None and result.get("year") == constraints["year"]:
        signals += 1

    if constraints["severities"]:
        if normalize_severity(result.get("severity")) in constraints["severities"]:
            signals += 1

    if constraints["phrases"]:
        if any(
            any(alias in blob for alias in INTENT_PATTERNS[phrase])
            for phrase in constraints["phrases"]
        ):
            signals += 1

    if any(keyword in blob for keyword in constraints["keywords"]):
        signals += 1

    return signals


def is_strong_match(
    result: dict[str, Any], constraints: dict[str, Any], refined_score: float
) -> bool:
    """Decide whether a result is strong enough to show to the user."""
    base_score = float(result["score"])
    has_constraints = bool(
        constraints["year"]
        or constraints["severities"]
        or constraints["phrases"]
        or constraints["keywords"]
    )

    # Reject obviously weak semantic matches before any extra logic runs.
    if base_score < MIN_BASE_SCORE:
        return False

    if not has_constraints:
        return refined_score >= MIN_REFINED_SCORE

    if not has_required_constraints(result, constraints):
        return False

    signal_count = count_match_signals(result, constraints)
    return refined_score >= MIN_CONSTRAINED_SCORE and signal_count >= 2


def choose_candidate_pool(
    results: list[dict[str, Any]], constraints: dict[str, Any], top_k: int
) -> list[dict[str, Any]]:
    """Apply strict filters only when they still leave us enough useful candidates."""
    pool = results
    year = constraints["year"]
    if year is not None:
        year_matches = [result for result in pool if result.get("year") == year]
        if year_matches:
            LOGGER.info("Applied year filter and kept %s candidates.", len(year_matches))
            pool = year_matches

    if constraints["severities"]:
        severity_matches = [
            result
            for result in pool
            if normalize_severity(result.get("severity")) in constraints["severities"]
        ]
        if len(severity_matches) >= max(2, top_k // 2):
            LOGGER.info("Applied severity filter and kept %s candidates.", len(severity_matches))
            pool = severity_matches

    return pool


def finalize_results(
    results: list[dict[str, Any]], constraints: dict[str, Any], top_k: int
) -> list[dict[str, Any]]:
    """Rerank candidates, attach match reasons, and trim to the final top_k."""
    candidate_pool = choose_candidate_pool(results, constraints, top_k)
    ranked_pool = sorted(
        candidate_pool,
        key=lambda result: score_result(result, constraints),
        reverse=True,
    )

    final_results: list[dict[str, Any]] = []
    for result in ranked_pool:
        refined_score = score_result(result, constraints)
        if not is_strong_match(result, constraints, refined_score):
            continue

        result["match_reason"] = "; ".join(collect_match_reasons(result, constraints))
        result["score_percent"] = round(float(result["score"]) * 100, 2)
        final_results.append(result)
        if len(final_results) >= top_k:
            break

    if not final_results:
        LOGGER.info("No strong CVE matches passed the final retrieval threshold.")

    return final_results


def load_jsonl_record_lookup() -> dict[str, dict[str, Any]]:
    """Load all JSONL files in the data folder into a lookup keyed by cve_id."""
    lookup: dict[str, dict[str, Any]] = {}
    jsonl_paths = sorted((DATA_DIR / "processed").glob("*.jsonl"))

    if not jsonl_paths:
        LOGGER.warning("No JSONL files were found in %s.", DATA_DIR / "processed")
        return lookup

    for jsonl_path in jsonl_paths:
        LOGGER.info("Loading full CVE records from %s", jsonl_path)
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped_line = line.strip()
                if not stripped_line:
                    continue

                record = json.loads(stripped_line)
                cve_id = record.get("cve_id")
                if cve_id is None:
                    continue

                # Store the most recent copy for a CVE ID so later lookup is straightforward.
                lookup[str(cve_id)] = record
                if line_number % 1000 == 0:
                    LOGGER.info("Loaded %s lines from %s", line_number, jsonl_path.name)

    LOGGER.info("Loaded %s CVE records into the retrieval lookup.", len(lookup))
    return lookup


class Retriever:
    """Load retrieval assets once and serve repeated search requests."""

    def __init__(self) -> None:
        """Initialize the retriever with one shared model, index, metadata list, and lookup."""
        self.index = self.load_index()
        self.metadata = self.load_metadata()
        self.record_lookup = load_jsonl_record_lookup()
        self.model = self.load_model()
        self.validate_alignment()

    def load_index(self) -> faiss.Index:
        """Load the prebuilt FAISS index from disk."""
        if not INDEX_PATH.exists():
            LOGGER.error("Run python -m src.indexer --years 2023 2024 first.")
            raise SystemExit(1)

        LOGGER.info("Loading FAISS index from %s", INDEX_PATH)
        return faiss.read_index(str(INDEX_PATH))

    def load_metadata(self) -> list[dict[str, Any]]:
        """Load the combined metadata list from disk."""
        if not METADATA_PATH.exists():
            LOGGER.error("Combined metadata file not found at %s.", METADATA_PATH)
            raise SystemExit(1)

        LOGGER.info("Loading metadata from %s", METADATA_PATH)
        with METADATA_PATH.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        return metadata

    def load_model(self) -> SentenceTransformer:
        """Load the same sentence-transformer model used during embedding."""
        LOGGER.info("Loading embedding model: %s", MODEL_NAME)
        return SentenceTransformer(MODEL_NAME)

    def validate_alignment(self) -> None:
        """Check that FAISS vector count matches metadata length exactly."""
        if self.index.ntotal != len(self.metadata):
            LOGGER.error(
                "Index vectors (%s) do not match metadata entries (%s).",
                self.index.ntotal,
                len(self.metadata),
            )
            raise SystemExit(1)

    def encode_query(self, query: str) -> np.ndarray:
        """Embed and normalize one query string for FAISS search."""
        # Encode as a one-item batch so the returned shape is compatible with FAISS search.
        embedding = self.model.encode([query], batch_size=BATCH_SIZE, show_progress_bar=False)

        # Convert to float32 because FAISS expects query vectors in this data type.
        query_vector = np.asarray(embedding, dtype="float32")
        if query_vector.ndim != 2 or query_vector.shape[1] != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Expected query embedding dimension {EMBEDDING_DIMENSION}, "
                f"got shape {query_vector.shape}."
            )

        # Normalize the query the same way as document embeddings so scores are cosine similarity.
        faiss.normalize_L2(query_vector)
        return query_vector

    def build_result(self, metadata_item: dict[str, Any], score: float) -> dict[str, Any]:
        """Merge metadata with full-record lookup details for one search hit."""
        cve_id = metadata_item.get("cve_id")
        lookup_record = self.record_lookup.get(str(cve_id))

        if lookup_record is None:
            description = NOT_AVAILABLE
            affected_products = NOT_AVAILABLE
            cwe_id = NOT_AVAILABLE
            problem_type = NOT_AVAILABLE
        else:
            description = normalize_text_value(
                lookup_record.get("description"), NOT_AVAILABLE
            )
            raw_products = lookup_record.get("affected_products")
            if isinstance(raw_products, list):
                affected_products = raw_products
            elif raw_products:
                affected_products = [str(raw_products)]
            else:
                affected_products = NOT_AVAILABLE

            cwe_id = normalize_text_value(lookup_record.get("cwe_id"), NOT_AVAILABLE)
            problem_type = normalize_text_value(
                lookup_record.get("problem_type"), NOT_AVAILABLE
            )

        return {
            "cve_id": cve_id,
            "description": description,
            "cvss": metadata_item.get("cvss_score"),
            "severity": normalize_severity(metadata_item.get("severity")),
            "affected_products": affected_products,
            "year": metadata_item.get("year"),
            "cwe_id": cwe_id,
            "problem_type": problem_type,
            "match_reason": "semantic similarity",
            "score": float(score),
        }

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search the FAISS index and return enriched CVE results."""
        if top_k <= 0 or self.index.ntotal == 0:
            return []

        constraints = parse_query_constraints(query)
        query_vector = self.encode_query(query)

        # Pull a bigger candidate set first so we can rerank with year and attack-type hints.
        requested_results = min(max(top_k * 8, 25), self.index.ntotal)
        scores, indices = self.index.search(query_vector, requested_results)

        results: list[dict[str, Any]] = []
        for score, index_position in zip(scores[0], indices[0]):
            # FAISS uses -1 when there is no valid match for a position.
            if index_position < 0 or index_position >= len(self.metadata):
                continue

            metadata_item = self.metadata[index_position]
            results.append(self.build_result(metadata_item, float(score)))

        return finalize_results(results, constraints, top_k)


def search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Search the shared retriever instance without reloading assets."""
    return get_retriever().search(query, top_k=top_k)


def get_retriever() -> Retriever:
    """Create the shared retriever only when it is first needed."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def display_results(results: list[dict[str, Any]]) -> None:
    """Print retrieval results in a compact human-readable CLI format."""
    if not results:
        print("No results found.")
        return

    for result in results:
        description = normalize_text_value(result.get("description"), NOT_AVAILABLE)
        short_description = description[:150]
        print(f"CVE ID: {result.get('cve_id')}")
        print(f"Year: {result.get('year')}")
        print(f"Severity: {normalize_text_value(result.get('severity'), 'UNKNOWN')}")
        print(f"CVSS: {result.get('cvss')}")
        print(f"Score: {result.get('score'):.4f} ({result.get('score_percent')}%)")
        print(f"Why it matched: {result.get('match_reason')}")
        print(f"Description: {short_description}")
        print()


def main() -> None:
    """Run the retriever CLI for manual testing."""
    configure_logging()
    args = parse_args()
    results = search(args.query, top_k=args.top_k)
    display_results(results)


if __name__ == "__main__":
    main()
