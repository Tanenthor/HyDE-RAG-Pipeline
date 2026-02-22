1. Architecture & Container Strategy
The system will be distributed across three new Docker containers that collaborate with your existing OpenWebUI instance:

Orchestration Container (FastAPI + LangChain/LlamaIndex): Acts as the brain of the pipeline. It exposes an OpenAI-compatible API endpoint that OpenWebUI can query. It handles the multi-step HyDE logic, formats the prompts, and parses the metadata for citations.

Ollama Container: Hosts the local Large Language Models. It will perform two distinct tasks: generating the hypothetical documents for retrieval, and generating the final synthesized response. It can also host the embedding model (e.g., nomic-embed-text or bge-m3).

ChromaDB Container: The vector database. It will store the dense embeddings of your handbook chunks alongside their rich metadata (Source name, authors, publish date, chapter, paragraph).

2. Data Ingestion & Metadata Management
Before querying, the large handbooks must be processed into ChromaDB:

Parsing: Extract text while maintaining hierarchical awareness (Chapter -> Section -> Paragraph).

Chunking: Split the text at the paragraph or section level to ensure each chunk represents a cohesive thought.

Metadata Tagging: Attach a JSON payload to every chunk containing: {"source": "Handbook A", "author": "John Doe", "date": "2023-05", "chapter": "4", "paragraph": "12"}.

Embedding: Pass the chunks through the embedding model and store them in ChromaDB.

3. The HyDE Retrieval Pipeline
When a user submits a query via OpenWebUI, the Orchestration container executes the following sequence based on the HyDE methodology:

Hypothetical Generation: Given a query, HyDE first zero-shot prompts an instruction-following language model to generate a hypothetical document. The document captures relevance patterns but is "fake" and may contain hallucinations.

Dense Embedding: Then, an unsupervised contrastively learned encoder encodes the document into an embedding vector.

Vector Search: This vector identifies a neighborhood in the corpus embedding space, from which similar real documents are retrieved based on vector similarity. This second step grounds the generated document to the actual corpus, with the encoder's dense bottleneck filtering out the hallucinations.

Final Generation: The Orchestration service formats a new prompt containing the user's original query, the retrieved chunks, and strict instructions to cite the provided metadata inline. Ollama generates the final answer.
