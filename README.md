# Citation-Aware RAG Pipeline with HyDE

This project implements a Retrieval-Augmented Generation (RAG) pipeline optimized for large handbooks. It utilizes the Hypothetical Document Embeddings (HyDE) methodology to improve retrieval accuracy for complex queries and enforces strict source citations in the final LLM output.

## Architecture

The project runs in Docker and consists of three core services that integrate with an existing OpenWebUI instance:

1. **Orchestration API (FastAPI):** Manages the HyDE logic, database queries, and prompt formatting.
2. **Ollama:** Provides local LLM inference for text generation and embeddings.
3. **ChromaDB:** A vector database storing document chunks and citation metadata.

## Prerequisites

* Docker and Docker Compose
* An existing OpenWebUI container running on the same Docker network
* Handbooks/documents prepared for ingestion

## System Components

### 1. Ingestion Script (`ingest.py`)
Processes your large handbooks. Text is chunked, and crucial metadata is attached to each chunk before being embedded into ChromaDB. 
* **Metadata Schema:** `source_name`, `authors`, `publish_date`, `chapter`, `paragraph`.

### 2. Orchestration Service (`main.py`)
Exposes an API endpoint that OpenWebUI connects to (mimicking an OpenAI endpoint or an OpenWebUI Pipeline). 
**Query Flow:**
1. Receives the user query.
2. Prompts Ollama to generate a hypothetical, ideal response to the query.
3. Embeds the hypothetical response.
4. Queries ChromaDB using the hypothetical embedding to retrieve the most semantically relevant real document chunks.
5. Constructs a final prompt injecting the retrieved chunks and their metadata, instructing the LLM to answer the user's query and append citations (e.g., *[Source Name, Chapter X, Paragraph Y]*).
6. Streams the final response back to OpenWebUI.

## Deployment

1. **Configure Docker Compose:**
   Ensure the network configurations in `docker-compose.yml` allow the Orchestration container to communicate with your existing OpenWebUI instance.

2. **Pull and Start Services:**
   ```bash
   docker-compose up -d
   ```

3. **Initialize Models in Ollama:**
   Exec into the Ollama container and pull your preferred models for generation and embedding.
   ```bash
   docker exec -it ollama ollama run llama3
   docker exec -it ollama ollama pull nomic-embed-text
   ```

4. **Ingest Documents:**
   Run the ingestion script to populate ChromaDB.
   ```bash
   python ingest.py --path /documents

   ```



## OpenWebUI Integration

1. Navigate to your OpenWebUI Settings.
2. Go to **Connections** (or Pipelines, depending on your version).
3. Add a new OpenAI API connection pointing to the Orchestration container's address (e.g., `http://orchestration:8000/v1`).
4. Select the proxy model provided by the Orchestration API to chat with your handbooks.
