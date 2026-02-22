# Citation-Aware RAG Pipeline with HyDE

A Retrieval-Augmented Generation (RAG) pipeline optimised for large handbooks.  
It uses the **Hypothetical Document Embeddings (HyDE)** methodology to improve retrieval accuracy for complex queries and enforces strict inline source citations in every LLM response.

## Architecture

Four Docker containers, fully decoupled from OpenWebUI's native ingestion:

| Container | Image | Purpose |
|-----------|-------|---------|
| `ingestion-ui` | Streamlit (custom) | Drag-and-drop upload portal with LLM-assisted metadata pre-fill and human verification |
| `orchestration` | FastAPI (custom) | HyDE pipeline, chunking, ChromaDB writes, OpenAI-compatible query endpoint |
| `ollama` | `ollama/ollama` | Local LLM inference — metadata extraction, hypothetical doc generation, final answer synthesis, embeddings |
| `chromadb` | `chromadb/chroma` | Vector store for chunk embeddings and citation metadata |

### Data flows

```
Ingestion:
  User → Ingestion UI → Ollama (metadata inference)
       → User (verification) → Orchestration API → ChromaDB

Retrieval:
  OpenWebUI → Orchestration API → Ollama (HyDE generation)
            → ChromaDB (vector search) → Orchestration API (prompt assembly)
            → Ollama (final generation) → OpenWebUI
```

## Prerequisites

* Docker ≥ 24 and Docker Compose v2
* An NVIDIA GPU (optional — remove the `deploy` block in `docker-compose.yml` for CPU-only)
* An existing OpenWebUI instance reachable on the same host or Docker network

## Project Structure

```
HyDE-RAG-Pipeline/
├── docker-compose.yml
├── .env.example
├── ingestion-ui/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app.py                  # Streamlit UI
├── orchestration/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                 # FastAPI app — /v1/chat/completions, /ingest
│   ├── hyde_pipeline.py        # HyDE retrieval logic
│   └── chunker.py              # File parsing & text splitting
├── documents/                  # Drop source files here (optional)
│   └── README.md
└── scripts/
    └── init_ollama_models.sh   # Pull required models after first start
```

## Deployment

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env to set GENERATION_MODEL, EMBEDDING_MODEL, ports, etc.
```

### 2. (Optional) Connect to an existing OpenWebUI network

If your OpenWebUI container runs on a named Docker network, add it to the `networks` section of `docker-compose.yml`:

```yaml
networks:
  hyde-net:
    external: true          # if the network already exists
    name: your-openwebui-network
```

### 3. Build and start all services

```bash
docker compose up -d --build
```

### 4. Pull Ollama models

```bash
bash scripts/init_ollama_models.sh
```

Or manually:

```bash
docker exec -it ollama ollama pull llama3
docker exec -it ollama ollama pull nomic-embed-text
```

### 5. Ingest documents

Open the **Ingestion UI** at `http://localhost:8501`, upload a handbook, review the LLM-inferred metadata, and click **Confirm & Ingest**.

## OpenWebUI Integration

1. Open OpenWebUI → **Settings → Connections**.
2. Add a new OpenAI API connection:
   - **URL:** `http://orchestration:8000/v1` (or `http://localhost:8000/v1` if OpenWebUI runs on the host)
   - **API Key:** `none` (any non-empty string is accepted)
3. Select the model `llama3` (or whichever `GENERATION_MODEL` you configured).

## Service Endpoints

| Service | URL | Notes |
|---------|-----|-------|
| Ingestion UI | `http://localhost:8501` | Upload + verify documents |
| Orchestration API docs | `http://localhost:8000/docs` | FastAPI Swagger UI |
| Orchestration health | `http://localhost:8000/health` | Liveness probe |
| ChromaDB | `http://localhost:8080` | REST API |
| Ollama | `http://localhost:11434` | Model API |

## Metadata Schema

Every stored chunk carries the following metadata:

| Field | Description |
|-------|-------------|
| `source_name` | Official title of the document |
| `authors` | Comma-separated author list |
| `publish_date` | Publication date (`YYYY-MM` or `YYYY-MM-DD`) |
| `chapter` | Chapter number or title |
| `paragraph` | Starting paragraph number |
| `chunk_index` | 0-based chunk position within the document |
| `total_chunks` | Total number of chunks for this document |
| `filename` | Original uploaded filename |
