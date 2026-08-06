"""
Hybrid retrieval: combines dense (vector) search with sparse (BM25 keyword)
search, then reranks the merged candidates with a cross-encoder.
"""

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Small, CPU-friendly cross-encoder. Good default for reranking on a laptop
# with no GPU. Swap for "cross-encoder/ms-marco-MiniLM-L-12-v2" or a bigger
# model if you have GPU headroom and want higher-quality reranking.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def build_hybrid_retriever(vector_store, chunks: list, k: int = 4, fetch_k: int = 20):
    """
    Build a retriever that:
      1. Runs BM25 (keyword) and vector (semantic) search in parallel
      2. Merges their results via reciprocal rank fusion (EnsembleRetriever)
      3. Reranks the merged candidate pool with a cross-encoder
      4. Returns only the top-k most relevant chunks to feed the LLM

    Args:
        vector_store: an existing Chroma vector store (see vector_store.py)
        chunks: list[Document] — the same chunks used to build the vector
            store. BM25 needs the raw text to build its keyword index; it
            can't be reconstructed from embeddings alone.
        k: final number of chunks handed to the LLM as context
        fetch_k: how many candidates each retriever pulls before the
            fusion + rerank step narrows it down to k. Wider net here
            costs a bit of latency but rarely hurts quality.
    """

    # --- 1. Sparse retriever (keyword-based) ---
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = fetch_k

    # --- 2. Dense retriever (embedding-based, same as the original setup) ---
    vector_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": fetch_k}
    )

    # --- 3. Combine both with reciprocal rank fusion ---
    # weights=[0.5, 0.5] gives equal say to keyword and semantic matches.
    # Tune this per corpus:
    #   - docs dense with jargon/equations/proper nouns -> raise BM25 weight
    #   - questions phrased very differently from the source text -> raise
    #     vector weight
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )

    # --- 4. Rerank the fused candidates with a cross-encoder ---
    # A cross-encoder reads (query, chunk) TOGETHER and outputs one
    # relevance score, instead of comparing two independently-computed
    # embeddings. This is more accurate but slower — which is exactly why
    # we only run it on the ~fetch_k merged candidates, not the whole
    # corpus, and only keep the top-k afterward.
    cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=k)

    hybrid_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=ensemble_retriever
    )

    return hybrid_retriever
