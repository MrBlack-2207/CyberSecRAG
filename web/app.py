"""This module serves the Flask web demo for the CyberSecRAG project.
It exposes one page for the UI and one JSON endpoint that runs the full pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Add the project root to Python's import path so `python web/app.py` can find `src/`.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

PIPELINE_ERROR: str | None = None
pipeline_query = None

try:
    # Import the pipeline once at startup so the web app can reuse it for every request.
    from src.pipeline import query as pipeline_query
except Exception as error:  # pragma: no cover - startup safety for local demo
    PIPELINE_ERROR = str(error)
    LOGGER.exception("Pipeline failed to load during startup.")

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)


def build_nvd_url(cve_id: Any) -> str | None:
    """Build the public NVD detail URL for a CVE ID."""
    if not cve_id:
        return None

    # NVD uses the CVE ID directly in the vulnerability detail page URL.
    return f"https://nvd.nist.gov/vuln/detail/{cve_id}"


def serialize_cve(cve: dict[str, Any]) -> dict[str, Any]:
    """Reduce pipeline results to the fields the frontend needs."""
    return {
        "cve_id": cve.get("cve_id"),
        "severity": cve.get("severity"),
        "cvss": cve.get("cvss"),
        "description": cve.get("description"),
        "affected_products": cve.get("affected_products"),
        "score": cve.get("score"),
        "nvd_url": build_nvd_url(cve.get("cve_id")),
    }


def build_error_response(message: str, status_code: int) -> tuple[Any, int]:
    """Return a JSON error response with the requested HTTP status code."""
    return jsonify({"error": message}), status_code


@app.get("/")
def index() -> str:
    """Serve the single-page CyberSecRAG demo interface."""
    return render_template("index.html")


@app.post("/query")
def run_query() -> tuple[Any, int] | Any:
    """Run the RAG pipeline for one user query and return JSON output."""
    if PIPELINE_ERROR is not None or pipeline_query is None:
        return build_error_response(
            "Pipeline is not ready. Make sure the index exists and the environment is set up.",
            500,
        )

    payload = request.get_json(silent=True) or {}
    user_query = str(payload.get("query", "")).strip()
    if not user_query:
        return build_error_response("Query text is required.", 400)

    try:
        # Call the shared pipeline function so retrieval and generation stay in one place.
        result = pipeline_query(user_query)
        cves = [serialize_cve(cve) for cve in result.get("retrieved_cves", [])]
        return jsonify({"answer": result.get("answer", ""), "cves": cves})
    except Exception as error:  # pragma: no cover - runtime safety for local demo
        LOGGER.exception("Query handling failed.")
        return build_error_response(str(error), 500)


def main() -> None:
    """Run the Flask development server for the professor demo."""
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
