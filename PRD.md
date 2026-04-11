# CyberSecRAG — Product Requirements Document (PRD)
**Version:** 1.0 | **Student Project** | **Last Updated:** April 2026

---

## 1. Project Summary

CyberSecRAG is an AI-powered cybersecurity assistant that helps users understand software vulnerabilities. A user types a natural language question like *"What are critical Apache vulnerabilities in 2024?"* and the system:

1. Searches a database of ~35,000 real CVE (Common Vulnerabilities and Exposures) records
2. Retrieves the most relevant ones using semantic similarity
3. Generates a clear, factual explanation — grounded only in what it found

The project also experiments with **autoencoders** (a type of neural network) to see whether compressing the search vectors makes retrieval faster or better — a research angle that distinguishes it academically.

---

## 2. Problem Being Solved

Security professionals and students need to quickly understand vulnerability data. The official CVE database has 200,000+ entries — it's impossible to manually search. Current tools:
- Return keyword matches (miss semantically related results)
- Give raw JSON data with no explanation
- Don't connect related vulnerabilities

CyberSecRAG solves this by combining **semantic search** with **AI-generated explanations**, always citing the source CVEs.

---

## 3. Target User

- **Primary**: CS/cybersecurity students and researchers
- **Secondary**: Professors evaluating the project
- **Not for**: Security professionals in production environments (this is a research prototype)

---

## 4. System Architecture

```
User Query
    │
    ▼
[Embedding Layer]          ← sentence-transformers (all-MiniLM-L6-v2)
    │                         Converts text to 384-dimensional vector
    ▼
[Vector Search]            ← FAISS (cosine similarity)
    │                         Finds top-5 most similar CVE vectors
    ▼
[Context Assembly]         ← Formats CVE data into a prompt
    │
    ▼
[LLM Generation]           ← Anthropic Claude (claude-haiku)
    │                         Answers ONLY using retrieved CVEs
    ▼
Answer + Source CVEs
```

**Phase 2 enhancement:**
```
Embeddings (384-d)
    │
    ▼
[Autoencoder Encoder]      ← 384 → 128 → 64 dimensions
    │                         Trained to compress without losing meaning
    ▼
Compressed Vectors (64-d)  ← Smaller FAISS index, potentially better retrieval
```

---

## 5. Features

### Phase 1 — Core RAG System (Must Have)

| Feature | Description |
|--------|-------------|
| CVE Extraction | Convert 35k CVE JSON files into a single searchable JSONL file |
| Embedding Generation | Embed all CVEs using sentence-transformers |
| FAISS Index | Build a fast similarity search index |
| Retrieval | Return top-5 relevant CVEs for any query |
| Grounded Generation | LLM answers only from retrieved context, never hallucinating |
| Web Demo | Simple Flask website for professor showcase |

### Phase 2 — Autoencoder Research (Should Have)

| Feature | Description |
|--------|-------------|
| Autoencoder Training | Train 384→64→384 autoencoder on CVE embeddings |
| Compressed Index | Build second FAISS index using 64-d vectors |
| Comparison Evaluation | Measure Recall@5, latency, memory for both systems |
| Evaluation Notebook | Side-by-side comparison notebook |

---

## 6. Data

- **Source**: MITRE CVEListV5 (official, publicly available)
- **Dataset size**: ~334,000 files, 3GB on disk
- **Initial scope**: 2024 CVEs only (~35,000 records) to keep iteration fast
- **Expansion**: Add 2023 later with `--year 2023` flag

**Fields extracted per CVE:**
- `cve_id` — e.g., CVE-2024-12345
- `description` — plain English description of the vulnerability
- `cvss` — severity score (0.0–10.0)
- `severity` — Low / Medium / High / Critical
- `affected_products` — vendor + product names
- `references` — source URLs

---

## 7. Technology Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Language | Python 3.10+ | Student familiar, rich ML ecosystem |
| Embedding model | sentence-transformers (all-MiniLM-L6-v2) | Fast, free, good quality, runs on CPU/GPU |
| Vector database | FAISS (Facebook AI Similarity Search) | Industry standard, runs locally, no cloud needed |
| LLM | Groq API — `llama-3.1-8b-instant` | Completely free, no credit card, very fast, OpenAI-compatible SDK |
| Autoencoder | PyTorch | Standard deep learning framework |
| Web framework | Flask | Lightweight, no build tools needed |
| Frontend | Vanilla HTML/CSS/JS | No npm, no frameworks, easy to explain |
| Notebooks | Jupyter | Standard for academic demonstration |

---

## 8. Non-Requirements (Out of Scope)

- ❌ No real-time CVE ingestion (static dataset only)
- ❌ No user authentication
- ❌ No production deployment (localhost demo only)
- ❌ No exploitation guidance (explanation and mitigation only)
- ❌ No multi-user support
- ❌ No cloud database

---

## 9. Evaluation Metrics

### Phase 1 Baseline
- **Retrieval relevance**: Manual check — do top-5 results make sense for the query?
- **Answer quality**: Does the LLM answer match the retrieved CVE content?
- **Coverage**: % of CVEs with CVSS score, severity, affected products

### Phase 2 Autoencoder Comparison
| Metric | How to Measure |
|--------|---------------|
| Recall@5 | For known query-CVE pairs, is the correct CVE in top-5? |
| Index size | `os.path.getsize()` on .index files |
| Query latency | `time.time()` before/after FAISS search |
| Similarity score distribution | Histogram of cosine similarity scores |

---

## 10. File Organization Rules

- `data/` — generated, gitignored, never committed
- `embeddings/` — generated, gitignored
- `index/` — generated, gitignored
- `src/` — all reusable Python modules
- `scripts/` — one-off CLI scripts (extract_cves.py lives here)
- `web/` — Flask app for demo
- `notebooks/` — Jupyter notebooks for exploration and presentation
- `tests/` — unit tests

---

## 11. Implementation Order

```
Step 1 ✅  scripts/extract_cves.py        DONE
Step 2 🔲  src/embedder.py               Next
Step 3 🔲  src/indexer.py
Step 4 🔲  src/retriever.py
Step 5 🔲  src/generator.py
Step 6 🔲  src/pipeline.py
Step 7 🔲  web/app.py + templates         Demo website
Step 8 🔲  notebooks/ (01–03)
--- Phase 1 complete — test end-to-end ---
Step 9 🔲  src/autoencoder.py             Phase 2 research
Step 10 🔲 notebooks/04_autoencoder_eval  Comparison
```

---

## 12. How to Run (Complete Commands)

```bash
# 1. Setup
git clone https://github.com/MrBlack-2207/CyberSecRAG
cd CyberSecRAG
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Extract CVEs
python scripts/extract_cves.py --year 2024 --input ./CVEListV5 --output ./data --validate

# 3. Generate embeddings (takes 5–15 min first time)
python -m src.embedder --year 2024

# 4. Build FAISS index
python -m src.indexer --year 2024

# 5. Test retrieval
python -m src.retriever --query "SQL injection vulnerability in web frameworks"

# 6. Run web demo
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
# Get it free (no credit card) at: https://console.groq.com
python web/app.py
# Open http://localhost:5000
```

---

## 13. Project Timeline (Student Estimate)

| Week | Task |
|------|------|
| Week 1 | Extract CVEs, generate embeddings, build FAISS index |
| Week 2 | Build retriever, generator, pipeline — test end-to-end |
| Week 3 | Build web demo + notebooks 01–03 |
| Week 4 | Phase 2: autoencoder training + evaluation |
| Week 5 | Polish, README, viva prep |
