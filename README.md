📄 Scientific PDF Research Assistant

A fully local, privacy-preserving RAG (Retrieval-Augmented Generation) system that lets you ask natural language questions over scientific PDFs - powered by LangChain, ChromaDB, and Mistral running locally via Ollama.

No API keys. No cloud. No cost.

Features
Upload one or multiple PDFs and query across all of them simultaneously
Hybrid retrieval (keyword + semantic search) with cross-encoder reranking for more relevant results
Answers grounded strictly in your documents — refuses out-of-scope questions
Source citations with page numbers for every answer
Runs 100% locally — your documents never leave your machine
Retrieval and answer quality measured with RAGAS against a hand-written evaluation set, not just spot-checked by hand
Architecture

PDF Input ↓ PyPDFLoader → RecursiveCharacterTextSplitter (chunks) ↓ OllamaEmbeddings (nomic-embed-text) → ChromaDB (vector store) ↓ User Question ↓ Hybrid Retriever: BM25 (keyword) + Vector search (semantic), run in parallel ↓ Reciprocal Rank Fusion (merge results) ↓ Cross-Encoder Reranking (top 20 → top 4 most relevant chunks) ↓ PromptTemplate + Retrieved Context → ChatOllama (Mistral) ↓ Grounded Answer with Source Pages

Tech Stack
Component	Technology
Framework	LangChain
LLM	Mistral 7B via Ollama (local)
Embeddings	nomic-embed-text via Ollama (local)
Vector Store	ChromaDB
Keyword Retrieval	BM25 (rank-bm25)
Reranking	cross-encoder/ms-marco-MiniLM-L-6-v2 (local)
Evaluation	RAGAS, judged by a local Ollama model
UI	Streamlit
PDF Parsing	PyPDF
Project Structure

PDF_Assistant/ ├── app.py # Streamlit UI ├── document_loader.py # PDF loading and chunking ├── vector_store.py # ChromaDB embeddings and storage ├── rag_pipeline.py # RAG chain, LLM config, retriever selection ├── hybrid_retriever.py # BM25 + vector fusion + cross-encoder reranking ├── eval_ragas.py # RAGAS evaluation against a golden test set ├── compare_retrievers.py # Baseline vs. hybrid retriever comparison ├── check_vector_store.py # Debug: inspect what's indexed in ChromaDB ├── debug_context_metrics.py # Debug: isolate a single question's retrieval + scores ├── requirements.txt └── README.md

Key Design Decisions

Chunking strategy: 1000-character chunks with 200-character overlap using RecursiveCharacterTextSplitter. The recursive separator order (\n\n → \n → space) ensures splits happen at natural language boundaries rather than mid-sentence.

Hybrid retrieval: Pure vector similarity search can miss exact keyword, acronym, or equation-name matches that don't embed distinctly from surrounding text. Combining it with BM25 (keyword search) via reciprocal rank fusion, then reranking the merged candidates with a cross-encoder (which scores the query and each chunk together, rather than comparing independent embeddings), meaningfully improves what actually reaches the LLM — see Evaluation below for measured results.

Grounded prompting: The prompt explicitly instructs the model to use only retrieved context and to respond with "I cannot find this information" when the answer is absent - preventing hallucination.

Local-first: Using Ollama for both the LLM and embeddings means zero data leaves the user's machine. Suitable for confidential research documents. Evaluation also uses a local Ollama model as judge, keeping the entire pipeline — including testing — free of external API calls.

Evaluation

Retrieval and answer quality are measured with RAGAS against a hand-written golden test set (8 questions with reference answers written from the source PDF, see eval_ragas.py), scored by a local Ollama model acting as judge.

Four metrics are tracked:

Faithfulness — is the answer grounded in the retrieved context, or hallucinated?
Answer relevancy — does the answer address the question actually asked?
Context precision — of the retrieved chunks, how many were relevant?
Context recall — did retrieval surface everything the reference answer needed?
Baseline (plain vector search) vs. hybrid retrieval
Metric	Baseline	Hybrid	Delta
Faithfulness	0.673	0.821	+0.149
Answer relevancy	0.216	0.739	+0.523
Context precision	0.229	0.396	+0.167
Context recall	0.344	0.375	+0.031

Adding hybrid search + reranking improved every metric, most notably answer relevancy (+0.52) and context precision (+0.17). Context recall improved only marginally — investigation traced part of this to equation- and formula-heavy passages that PyPDFLoader extracts poorly into plain text, which embed and retrieve worse than prose. That's an open improvement area (structure-aware chunking, or extracting tables/equations separately).
