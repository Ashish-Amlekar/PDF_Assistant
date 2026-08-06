# 📄 Scientific PDF Research Assistant

A fully local, privacy-preserving RAG (Retrieval-Augmented Generation) system
that lets you ask natural language questions over scientific PDFs - powered by
LangChain, ChromaDB, and Mistral running locally via Ollama.

No API keys. No cloud. No cost.

---


## Features

- Upload one or multiple PDFs and query across all of them simultaneously
- Answers grounded strictly in your documents — refuses out-of-scope questions
- Source citations with page numbers for every answer
- Runs 100% locally — your documents never leave your machine
- Semantic search via vector embeddings (not keyword matching)

---

## Architecture
PDF Input
↓
PyPDFLoader → RecursiveCharacterTextSplitter (chunks)
↓
OllamaEmbeddings (nomic-embed-text) → ChromaDB (vector store)
↓
User Question → Retriever (top-k semantic search)
↓
PromptTemplate + Retrieved Context → ChatOllama (Mistral)
↓
Grounded Answer with Source Pages

---

## Tech Stack

| Component | Technology |
|---|---|
| Framework | LangChain |
| LLM | Mistral 7B via Ollama (local) |
| Embeddings | nomic-embed-text via Ollama (local) |
| Vector Store | ChromaDB |
| UI | Streamlit |
| PDF Parsing | PyPDF |

---



## Project Structure
PDF_Assistant/
├── app.py                # Streamlit UI
├── document_loader.py    # PDF loading and chunking
├── vector_store.py       # ChromaDB embeddings and storage
├── rag_pipeline.py       # RAG chain and LLM configuration
├── requirements.txt
└── README.md

---

## Key Design Decisions

**Chunking strategy:** 1000-character chunks with 200-character overlap using
`RecursiveCharacterTextSplitter`. The recursive separator order
(`\n\n → \n → space`) ensures splits happen at natural language boundaries
rather than mid-sentence.

**Grounded prompting:** The prompt explicitly instructs the model to use only
retrieved context and to respond with "I cannot find this information" when
the answer is absent - preventing hallucination.

**Local-first:** Using Ollama for both the LLM and embeddings means zero
data leaves the user's machine. Suitable for confidential research documents.

