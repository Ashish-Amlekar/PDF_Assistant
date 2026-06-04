from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# This is where ChromaDB stores its data on your disk
CHROMA_DB_PATH = "./chroma_db"

# We use Ollama to generate embeddings locally (no API key needed)
# nomic-embed-text is a lightweight, fast embedding model
EMBEDDING_MODEL = "nomic-embed-text"


def get_embeddings():
    """Return the embedding function we'll use throughout the project."""
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def create_vector_store(chunks: list) -> Chroma:
    """
    Convert text chunks into numerical vectors (embeddings) and store them
    in ChromaDB so we can search them by semantic similarity later.

    What is an embedding? A list of ~768 numbers that captures the
    MEANING of a piece of text. Similar meanings → similar numbers.
    """
    print("\nCreating vector store...")
    print(f"  Embedding {len(chunks)} chunks with Ollama ({EMBEDDING_MODEL})...")
    print("  (This may take a minute — it's running locally on your machine)")

    embeddings = get_embeddings()

    # Chroma.from_documents does three things:
    # 1. Calls the embedding model on each chunk
    # 2. Stores the resulting vectors in ChromaDB
    # 3. Saves to disk at CHROMA_DB_PATH so you don't re-embed on restart
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )

    print(f"  Vector store saved to: {CHROMA_DB_PATH}")
    return vector_store


def load_vector_store() -> Chroma:
    """Load a previously created vector store from disk."""
    embeddings = get_embeddings()
    vector_store = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings
    )
    return vector_store


def vector_store_exists() -> bool:
    """Check whether a vector store has already been created."""
    import os
    return os.path.exists(CHROMA_DB_PATH) and len(os.listdir(CHROMA_DB_PATH)) > 0