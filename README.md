# CyberSecRAG

CyberSecRAG is a student research project that uses Retrieval-Augmented Generation (RAG) to answer cybersecurity questions from real CVE vulnerability records.

## What It Does

CyberSecRAG lets a user ask a plain-English question such as "critical remote code execution vulnerabilities in 2024" and then finds the most relevant CVE records from a local knowledge base. It does not rely on guesswork alone because it first retrieves matching vulnerabilities and then generates an answer using only that retrieved context. The project currently works on 66,038 CVE records from 2023 and 2024. In Phase 2, it also explores whether an autoencoder can compress embeddings from 384 dimensions down to 64 dimensions while still preserving useful retrieval quality.

## System Architecture

```text
User Query
    │
    ▼
[Embedding]  ← sentence-transformers (all-MiniLM-L6-v2)
    │
    ▼
[FAISS Search]  ← cosine similarity over 66,038 CVE vectors
    │
    ▼
[Context Assembly]
    │
    ▼
[LLM Generation]  ← Groq API (llama-3.1-8b-instant) — FREE
    │
    ▼
Grounded Answer + Source CVEs
```

### Phase 2 Extension — Autoencoder Compression

```text
Baseline Embeddings (384-d)
    │
    ▼
[Autoencoder Encoder]
    │
    ▼
Compressed Embeddings (64-d)
    │
    ▼
[Compressed FAISS Index]
    │
    ▼
Compare Size, Speed, Recall, and Similarity Scores
```

## Tech Stack

| Component | Technology | Purpose | Cost |
| --- | --- | --- | --- |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Convert CVE records and user queries into semantic vectors | Free |
| Vector DB / Search | FAISS (`IndexFlatIP`) | Fast cosine-similarity search over CVE embeddings | Free |
| LLM | Groq API + `llama-3.1-8b-instant` | Generate grounded natural-language answers from retrieved CVEs | Free tier |
| Web Framework | Flask + Flask-CORS | Serve the demo UI and query API | Free |
| Autoencoder | PyTorch | Compress 384-d embeddings to 64-d for research evaluation | Free |
| Data Source | NVD CVE JSON feeds / CVEListV5 | Provide the raw cybersecurity vulnerability records | Free |

## Project Structure

```text
CyberSecRAG/
│
├── data/
│   ├── raw/
│   │   └── CVEListV5/cves/<year>/     # Raw CVE JSON files from the source dataset
│   └── processed/
│       ├── 2023.jsonl                 # Extracted CVE records for 2023
│       ├── 2024.jsonl                 # Extracted CVE records for 2024
│       └── metadata.json              # Tracks which years were processed
│
├── embeddings/
│   ├── embeddings_2023.npy            # 2023 semantic embeddings
│   ├── embeddings_2024.npy            # 2024 semantic embeddings
│   ├── metadata_2023.json             # Lean metadata aligned to 2023 vectors
│   └── metadata_2024.json             # Lean metadata aligned to 2024 vectors
│
├── index/
│   ├── faiss_combined.index           # Baseline 384-d FAISS index
│   └── metadata_combined.json         # Metadata aligned with the combined index
│
├── models/
│   └── autoencoder.pth                # Trained Phase 2 autoencoder weights
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Extraction and dataset exploration notebook
│   ├── 02_embedding_demo.ipynb        # Embedding explanation notebook
│   ├── 03_retrieval_demo.ipynb        # Main live retrieval demo notebook
│   └── 04_autoencoder_eval.ipynb      # Phase 2 evaluation notebook
│
├── src/
│   ├── __init__.py                    # Makes src a Python package
│   ├── embedder.py                    # Builds embeddings from processed CVE records
│   ├── indexer.py                     # Builds baseline and compressed FAISS indexes
│   ├── retriever.py                   # Searches the FAISS index and returns CVE matches
│   ├── generator.py                   # Generates grounded answers from retrieved CVEs
│   ├── pipeline.py                    # Thin orchestrator for retrieval + generation
│   └── autoencoder.py                 # Phase 2 autoencoder training and encoding logic
│
├── web/
│   ├── app.py                         # Flask backend for the demo
│   ├── templates/index.html           # Single-page professor showcase UI
│   └── static/
│       ├── style.css                  # Static style file slot
│       └── script.js                  # Frontend search logic
│
├── tests/
│   ├── test_embedder.py               # Unit tests for embedding logic
│   └── test_retriever.py              # Unit tests for retrieval logic
│
├── requirements.txt                   # Python dependencies
├── .env.example                       # Example environment variable file
├── .gitignore                         # Git ignore rules for generated data and secrets
├── README.md                          # Project overview and usage guide
└── PRD.md                             # Project requirements document
```

## Setup & Installation

1. Clone the repository.

```bash
git clone https://github.com/MrBlack-2207/CyberSecRAG
cd CyberSecRAG
```

2. Create a virtual environment.

```bash
python -m venv venv
```

3. Activate the virtual environment.

```bash
venv\Scripts\activate
```

4. Install the project dependencies.

```bash
pip install -r requirements.txt
```

5. Create your `.env` file and add your Groq API key.

```bash
copy .env.example .env
```

Then open `.env` and paste:

```env
GROQ_API_KEY=your_actual_key_here
```

Get a free key here: [console.groq.com](https://console.groq.com)

## Running the Project

1. Run notebook `01_data_exploration.ipynb` to extract and process the CVE records for 2023 and 2024.

2. Generate embeddings for both years.

```bash
python -m src.embedder --years 2023 2024
```

3. Build the baseline FAISS index.

```bash
python -m src.indexer --years 2023 2024
```

4. Launch the web demo.

```bash
python web/app.py
```

5. Open the app in your browser.

```text
http://localhost:5000
```

## Phase 2 — Autoencoder Research

Phase 2 studies whether the original 384-dimensional embeddings can be compressed into 64-dimensional vectors without losing too much retrieval quality. The autoencoder is trained to reconstruct the original embedding from a smaller hidden representation, which makes it a useful tool for dimensionality reduction. This is interesting because smaller vectors can reduce index size and may improve efficiency, but only if the important semantic information is preserved. The evaluation notebook compares the baseline and compressed systems using index size, search latency, Recall@5, and similarity score distributions.

Train the autoencoder:

```bash
python -m src.autoencoder --train
```

Build the compressed FAISS index:

```bash
python -m src.indexer --compressed
```

Run the evaluation notebook:

```bash
jupyter notebook notebooks/04_autoencoder_eval.ipynb
```

## Results

| Metric | Baseline (384-d) | Compressed (64-d) |
| --- | --- | --- |
| Index Size | ~XX MB | ~XX MB |
| Avg Query Latency | ~X.XX ms | ~X.XX ms |
| Avg Recall@5 | 100% (reference) | ~XX.X% |
| Similarity Score Trend | Higher-dimensional baseline distribution | Comparable compressed distribution |

## Academic Context

This project was built as a student research project exploring how Retrieval-Augmented Generation can be applied to cybersecurity threat intelligence. It combines semantic search over 66,038 CVE records with grounded answer generation so the system responds using retrieved evidence rather than unsupported guesses. The Phase 2 research component extends this by studying autoencoder-based compression as a way to reduce vector index size while maintaining acceptable retrieval quality.
