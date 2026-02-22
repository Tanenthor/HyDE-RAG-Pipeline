1. Architecture & Container Strategy
The system will run across four Docker containers, entirely decoupled from OpenWebUI's native ingestion:

Ingestion UI Container (Streamlit): A lightweight web portal. It provides a drag-and-drop interface for irregular files, uses an LLM to pre-fill metadata fields (Title, Author, Chapter), and allows human review before submission.

Orchestration Container (FastAPI + LangChain/LlamaIndex): The backend brain. It exposes the OpenAI-compatible endpoint for OpenWebUI. It handles chunking incoming files from the Ingestion UI, executing the multi-step HyDE retrieval logic, and formatting the final prompt with strict citation instructions.

Ollama Container: Hosts the local LLMs. It handles three tasks: agentic metadata extraction during ingestion, generating the hypothetical documents for HyDE, and generating the final synthesized response.

ChromaDB Container: The vector database. It stores the dense embeddings of your handbook chunks alongside the manually verified rich metadata.

2. Data Ingestion & Metadata Management
Before querying, the large handbooks must be processed into ChromaDB via the Ingestion UI:

Upload: User drags and drops files (PDF, DOCX, TXT, Markdown) into the Ingestion UI portal.

Metadata Inference: The Ingestion UI sends extracted text to Ollama, which agenically pre-fills metadata fields (source_name, authors, publish_date, chapter, paragraph).

Human Verification: The user reviews and corrects the pre-filled metadata in a form before submission.

Chunking: The Orchestration API receives the verified file and metadata, splitting the text at the paragraph or section level to ensure each chunk represents a cohesive thought. Metadata schema per chunk: {"source_name": "...", "authors": "...", "publish_date": "...", "chapter": "...", "paragraph": "..."}.

Embedding & Storage: The Orchestration API embeds each chunk using the embedding model hosted in Ollama and stores the vectors alongside their metadata in ChromaDB.

3. Data Flow

Ingestion: User -> Ingestion UI -> Ollama (Metadata Inference) -> User (Verification) -> Orchestration API -> ChromaDB.

Retrieval: OpenWebUI -> Orchestration API -> Ollama (HyDE Generation) -> ChromaDB (Vector Search) -> Orchestration API (Prompt Assembly) -> Ollama (Final Generation) -> OpenWebUI.

4. The HyDE Retrieval Pipeline
When a user submits a query via OpenWebUI, the Orchestration container executes the following sequence based on the HyDE methodology:

Hypothetical Generation: Given a query, HyDE first zero-shot prompts an instruction-following language model to generate a hypothetical document. The document captures relevance patterns but is "fake" and may contain hallucinations.

Dense Embedding: Then, an unsupervised contrastively learned encoder encodes the hypothetical document into an embedding vector.

Vector Search: This vector identifies a neighbourhood in the corpus embedding space, from which similar real documents are retrieved based on vector similarity. This second step grounds the generated document to the actual corpus, with the encoder's dense bottleneck filtering out the hallucinations.

Final Generation: The Orchestration service formats a new prompt containing the user's original query, the retrieved chunks, and strict instructions to cite the provided metadata inline. Ollama generates the final answer.
