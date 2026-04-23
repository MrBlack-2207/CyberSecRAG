"""This module builds one combined FAISS index from yearly embedding files.
It keeps metadata in the same order as vectors so retrieval can map results correctly."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np

LOGGER = logging.getLogger(__name__)
EMBEDDINGS_DIR = Path("embeddings")
INDEX_DIR = Path("index")
INDEX_PATH = INDEX_DIR / "faiss_combined.index"
COMPRESSED_EMBEDDINGS_PATH = EMBEDDINGS_DIR / "embeddings_compressed.npy"
COMPRESSED_INDEX_PATH = INDEX_DIR / "faiss_compressed.index"
METADATA_PATH = INDEX_DIR / "metadata_combined.json"
STANDARD_DIMENSION = 384
COMPRESSED_DIMENSION = 64
COMPRESSED_YEARS = [2023, 2024]


def configure_logging() -> None:
    """Configure logging for command-line use."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the indexer."""
    parser = argparse.ArgumentParser(
        description="Build one combined FAISS index from yearly embedding files."
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        help="One or more CVE years, for example: --years 2023 2024",
    )
    parser.add_argument(
        "--compressed",
        action="store_true",
        help="Build a FAISS index from compressed 64-dimensional embeddings.",
    )
    return parser.parse_args()


def embeddings_path_for_year(year: int) -> Path:
    """Return the embedding file path for a year."""
    return EMBEDDINGS_DIR / f"embeddings_{year}.npy"


def metadata_path_for_year(year: int) -> Path:
    """Return the metadata file path for a year."""
    return EMBEDDINGS_DIR / f"metadata_{year}.json"


def load_year_data(year: int) -> tuple[np.ndarray, list[dict[str, Any]]] | None:
    """Load one year's embeddings and metadata, or return None when unavailable."""
    embedding_path = embeddings_path_for_year(year)
    metadata_path = metadata_path_for_year(year)

    if not embedding_path.exists():
        LOGGER.warning("Missing %s. Skipping year %s.", embedding_path, year)
        return None

    if not metadata_path.exists():
        LOGGER.warning("Missing %s. Skipping year %s.", metadata_path, year)
        return None

    embeddings = np.load(embedding_path)
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    if embeddings.ndim != 2:
        LOGGER.warning("Embeddings in %s are not 2D. Skipping year %s.", embedding_path, year)
        return None

    if len(metadata) != embeddings.shape[0]:
        LOGGER.warning(
            "Metadata count does not match vector count for year %s. Skipping year %s.",
            year,
            year,
        )
        return None

    # Convert to float32 because FAISS expects vectors in this NumPy data type.
    return np.asarray(embeddings, dtype="float32"), metadata


def combine_years(years: list[int]) -> tuple[np.ndarray, list[dict[str, Any]]] | None:
    """Combine all available yearly arrays and metadata into one ordered dataset."""
    arrays: list[np.ndarray] = []
    combined_metadata: list[dict[str, Any]] = []

    for year in years:
        year_data = load_year_data(year)
        if year_data is None:
            continue

        embeddings, metadata = year_data
        arrays.append(embeddings)

        # Extend in the same loop so metadata order matches vector order exactly.
        combined_metadata.extend(metadata)
        LOGGER.info("Loaded %s vectors for year %s.", embeddings.shape[0], year)

    if not arrays:
        return None

    # Stack yearly arrays vertically so row N still lines up with metadata item N.
    combined_embeddings = np.vstack(arrays)
    return combined_embeddings, combined_metadata


def validate_vector_dimension(embeddings: np.ndarray, expected_dimension: int) -> bool:
    """Check that embeddings are 2D and have the expected vector width."""
    if embeddings.ndim != 2:
        LOGGER.error("Expected a 2D embedding array, got shape %s.", embeddings.shape)
        return False

    if embeddings.shape[1] != expected_dimension:
        LOGGER.error(
            "Expected embeddings with dimension %s, got shape %s.",
            expected_dimension,
            embeddings.shape,
        )
        return False

    return True


def build_index(embeddings: np.ndarray, expected_dimension: int) -> faiss.IndexFlatIP:
    """Build an IndexFlatIP FAISS index from pre-normalized embeddings."""
    if not validate_vector_dimension(embeddings, expected_dimension):
        raise ValueError(f"Invalid embedding shape: {embeddings.shape}")

    # IndexFlatIP performs inner-product search, which is cosine similarity for normalized vectors.
    index = faiss.IndexFlatIP(expected_dimension)
    index.add(embeddings)
    return index


def save_outputs(
    index: faiss.IndexFlatIP, metadata: list[dict[str, Any]], index_path: Path
) -> None:
    """Save the FAISS index and combined metadata to disk."""
    # Create the output folder automatically so a clean repo can run the indexer first try.
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_path))
    with METADATA_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    index_size_bytes = index_path.stat().st_size
    LOGGER.info("Total vectors added: %s", index.ntotal)
    LOGGER.info("Index size on disk: %s bytes", index_size_bytes)
    LOGGER.info("Saved FAISS index to %s", index_path)
    LOGGER.info("Saved combined metadata to %s", METADATA_PATH)


def load_compressed_embeddings() -> np.ndarray | None:
    """Load the compressed 64-dimensional embeddings saved by the autoencoder step."""
    if not COMPRESSED_EMBEDDINGS_PATH.exists():
        LOGGER.error("Run python -m src.autoencoder --train first.")
        return None

    compressed_embeddings = np.load(COMPRESSED_EMBEDDINGS_PATH)
    return np.asarray(compressed_embeddings, dtype="float32")


def load_combined_metadata_for_compressed_index() -> list[dict[str, Any]] | None:
    """Load the same combined metadata order used by the baseline combined index."""
    combined_data = combine_years(COMPRESSED_YEARS)
    if combined_data is None:
        LOGGER.error("No valid metadata files were found for compressed indexing.")
        return None

    _, metadata = combined_data
    return metadata


def main() -> None:
    """Run the indexer CLI."""
    configure_logging()
    args = parse_args()

    if args.compressed:
        compressed_embeddings = load_compressed_embeddings()
        if compressed_embeddings is None:
            return

        metadata = load_combined_metadata_for_compressed_index()
        if metadata is None:
            return

        if len(metadata) != compressed_embeddings.shape[0]:
            LOGGER.error(
                "Metadata count does not match compressed vector count: %s vs %s.",
                len(metadata),
                compressed_embeddings.shape[0],
            )
            return

        index = build_index(compressed_embeddings, COMPRESSED_DIMENSION)
        save_outputs(index, metadata, COMPRESSED_INDEX_PATH)
        LOGGER.info("Built compressed index with 64-d vectors")
        return

    if not args.years:
        raise SystemExit("Use --years for the standard index or --compressed for the 64-d index.")

    combined_data = combine_years(args.years)

    if combined_data is None:
        LOGGER.error("No valid embedding files were found for the requested years.")
        return

    embeddings, metadata = combined_data
    index = build_index(embeddings, STANDARD_DIMENSION)
    save_outputs(index, metadata, INDEX_PATH)


if __name__ == "__main__":
    main()
