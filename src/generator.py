"""This module turns retrieved CVE context into a grounded Groq response.
It loads the API key from the environment and returns safe fallback text on API failures."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from dotenv import load_dotenv
from groq import Groq

LOGGER = logging.getLogger(__name__)
MODEL_NAME = "llama-3.1-8b-instant"
SYSTEM_PROMPT = (
    "You are a cybersecurity assistant. Answer ONLY using the CVE records provided. "
    "Do not guess or invent information. If the answer is not in the context, say so clearly. "
    "Write a polished professor-demo answer in plain text only. Do not use markdown, bold text, "
    "asterisks, bullet symbols, or headings like Summary. Start with a short opening paragraph, "
    "then describe each strong CVE in compact numbered sentences. Keep each CVE explanation to 1-2 "
    "sentences using year, severity, CVSS, affected products, problem type, and why it matches the "
    "query. If some retrieved CVEs are weaker matches, say that briefly and honestly. If the user "
    "asked for a year or severity, prioritize only CVEs that match those constraints when they are "
    "present in context. Keep the full answer concise enough to finish cleanly."
)
API_KEY_ERROR = (
    "GROQ_API_KEY not set. Add it to your .env file. "
    "Get a free key at https://console.groq.com"
)
GENERATION_ERROR = "Sorry, I was unable to generate a response. Please try again."

# Load the local .env file so the API key is available during normal project runs.
load_dotenv()


def get_api_key() -> str:
    """Return the Groq API key or raise a clear error when it is missing."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(API_KEY_ERROR)
    return api_key


def create_client() -> Groq:
    """Create a Groq client using the API key from the environment."""
    return Groq(api_key=get_api_key())


def format_context_entry(retrieved_cve: dict[str, Any]) -> str:
    """Format one retrieved CVE into the context block expected by the generator."""
    cve_id = retrieved_cve.get("cve_id", "Unknown CVE")
    year = retrieved_cve.get("year", "Unknown")
    severity = retrieved_cve.get("severity", "Unknown")
    cvss = retrieved_cve.get("cvss")
    description = retrieved_cve.get("description", "Not available")
    problem_type = retrieved_cve.get("problem_type", "Not available")
    cwe_id = retrieved_cve.get("cwe_id", "Not available")
    match_reason = retrieved_cve.get("match_reason", "semantic similarity")
    affected = retrieved_cve.get("affected_products", [])

    # Join product lists so the prompt stays readable for the model.
    if isinstance(affected, list):
        affected_products = ", ".join(affected) if affected else "Not available"
    else:
        affected_products = str(affected) if affected else "Not available"

    # Show N/A in the prompt when CVSS is missing so the model sees explicit missing data.
    cvss_text = "N/A" if cvss is None else str(cvss)
    return (
        f"[{cve_id}] Year: {year} | Severity: {severity} | CVSS: {cvss_text}\n"
        f"Problem Type: {problem_type}\n"
        f"CWE: {cwe_id}\n"
        f"Affected: {affected_products}"
        f"\nWhy it matched: {match_reason}\n"
        f"Description: {description}"
    )


def build_context_block(retrieved_cves: list[dict[str, Any]]) -> str:
    """Combine all retrieved CVEs into one context block for the model."""
    # Separate CVEs with blank lines so the model can distinguish one source from another.
    return "\n\n".join(format_context_entry(retrieved_cve) for retrieved_cve in retrieved_cves)


def build_user_message(user_query: str, retrieved_cves: list[dict[str, Any]]) -> str:
    """Build the user message that contains context followed by the question."""
    context_block = build_context_block(retrieved_cves)
    return (
        f"Context:\n{context_block}\n\n"
        f"Question: {user_query}\n\n"
        "Answer based only on the above CVEs. Use plain text only, stay descriptive but compact, "
        "and avoid markdown formatting."
    )


def clean_response_text(response_text: str) -> str:
    """Remove markdown-style formatting so the demo answer looks clean in the UI."""
    cleaned_text = response_text.replace("**", "")

    # Remove generic markdown heading lines that do not add much value in the UI.
    cleaned_text = re.sub(r"(?im)^\s*summary\s*$", "", cleaned_text)
    cleaned_text = re.sub(r"(?im)^\s*most relevant cves\s*$", "", cleaned_text)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    return cleaned_text.strip()


def generate(user_query: str, retrieved_cves: list[dict[str, Any]]) -> str:
    """Generate a grounded answer from retrieved CVE context using the Groq API."""
    if not retrieved_cves:
        return "No relevant CVEs were found for your query."

    client = create_client()
    user_message = build_user_message(user_query, retrieved_cves)

    try:
        # Keep temperature low so the answer stays factual and less creative.
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=800,
            temperature=0.2,
        )
    except Exception as error:
        # Log the API error so it can be debugged without exposing a traceback to end users.
        LOGGER.error("Groq generation failed: %s", error)
        return GENERATION_ERROR

    response_text = response.choices[0].message.content or GENERATION_ERROR
    return clean_response_text(response_text)
