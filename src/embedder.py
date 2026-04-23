"""This module builds normalized sentence embeddings from CVE JSONL files.
It supports multiple years so each year's embeddings and metadata can be saved separately."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

LOGGER = logging.getLogger(__name__)
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64
EMBEDDING_DIMENSION = 384
DATA_DIR = Path("data")
EMBEDDINGS_DIR = Path("embeddings")


def configure_logging() -> None:
    """Configure logging for command-line use."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the embedder."""
    parser = argparse.ArgumentParser(
        description="Generate normalized embeddings for one or more CVE years."
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        required=True,
        help="One or more CVE years, for example: --years 2023 2024",
    )
    return parser.parse_args()


def data_path_for_year(year: int) -> Path:
    """Return the input JSONL path for a given year."""
    return DATA_DIR / "processed" / f"{year}.jsonl"


def embeddings_path_for_year(year: int) -> Path:
    """Return the output NumPy path for a given year."""
    return EMBEDDINGS_DIR / f"embeddings_{year}.npy"


def metadata_path_for_year(year: int) -> Path:
    """Return the output metadata path for a given year."""
    return EMBEDDINGS_DIR / f"metadata_{year}.json"


def load_records(year: int) -> list[dict[str, Any]] | None:
    """Load all JSONL records for a year, or return None when the file is missing."""
    input_path = data_path_for_year(year)
    if not input_path.exists():
        LOGGER.error(
            "data/processed/%s.jsonl not found. Run notebooks/01_data_exploration.ipynb first.",
            year,
        )
        return None

    records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue

            records.append(json.loads(stripped_line))
            if line_number % 1000 == 0:
                LOGGER.info("Loaded %s lines for year %s.", line_number, year)

    return records


def build_metadata(record: dict[str, Any], year: int) -> dict[str, Any]:
    """Build the lean metadata payload stored alongside the embeddings."""
    # Fall back to the CLI year so metadata stays complete even if the field is missing.
    record_year = record.get("year")
    return {
        "cve_id": record.get("cve_id"),
        "severity": record.get("severity"),
        "cvss_score": record.get("cvss_score"),
        "year": year if record_year is None else record_year,
    }


def prepare_embedding_inputs(
    records: list[dict[str, Any]], year: int
) -> tuple[list[str], list[dict[str, Any]]]:
    """Create formatted texts and matching metadata for one year."""
    texts: list[str] = []
    metadata: list[dict[str, Any]] = []

    for index, record in enumerate(records, start=1):
        # The notebook already prepares the exact embedding text, so reuse it directly.
        embedding_text = str(record.get("embedding_text", "")).strip()
        if not embedding_text:
            continue

        texts.append(embedding_text)
        metadata.append(build_metadata(record, year))

        if index % 1000 == 0:
            LOGGER.info("Prepared %s records for year %s.", index, year)

    return texts, metadata


def load_model() -> SentenceTransformer:
    """Load the sentence-transformer model used for CVE embeddings."""
    LOGGER.info("Loading embedding model: %s", MODEL_NAME)
    return SentenceTransformer(MODEL_NAME)


def encode_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Encode text into normalized float32 embeddings."""
    # The model encodes texts in batches so memory usage stays predictable.
    embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)

    # Convert to float32 because FAISS expects NumPy arrays in this format.
    embedding_array = np.asarray(embeddings, dtype="float32")

    if embedding_array.ndim != 2 or embedding_array.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected embedding dimension {EMBEDDING_DIMENSION}, "
            f"got shape {embedding_array.shape}."
        )

    # Normalize vectors so later FAISS inner-product search behaves like cosine similarity.
    faiss.normalize_L2(embedding_array)
    return embedding_array


def save_outputs(year: int, embeddings: np.ndarray, metadata: list[dict[str, Any]]) -> None:
    """Save embeddings and metadata for one year."""
    # Create the output folder automatically so the first run does not fail.
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    np.save(embeddings_path_for_year(year), embeddings)
    with metadata_path_for_year(year).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    LOGGER.info("Processed %s records for year %s.", len(metadata), year)
    LOGGER.info("Saved embedding array for year %s with shape %s.", year, embeddings.shape)


def process_year(model: SentenceTransformer, year: int) -> None:
    """Load, format, embed, and save one year's CVE records."""
    records = load_records(year)
    if records is None:
        return

    if not records:
        LOGGER.warning("No records found in %s. Skipping year %s.", data_path_for_year(year), year)
        return

    texts, metadata = prepare_embedding_inputs(records, year)
    embeddings = encode_texts(model, texts)
    save_outputs(year, embeddings, metadata)


def main() -> None:
    """Run the embedder CLI."""
    configure_logging()
    args = parse_args()
    model = load_model()

    for year in args.years:
        try:
            process_year(model, year)
        except Exception:
            # Log the stack trace so debugging is easier, then continue to the next year.
            LOGGER.exception("Failed to process year %s.", year)


if __name__ == "__main__":
    main()
