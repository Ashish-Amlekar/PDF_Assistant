from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

from vector_store import load_vector_store
from hybrid_retriever import build_hybrid_retriever

# The LLM we'll use for generating answers
LLM_MODEL = "mistral"


def _reconstruct_documents(vector_store) -> list:
    """
    Pull all stored chunks back out of Chroma as Document objects.

    Why: BM25 (used inside the hybrid retriever) needs the raw chunk text
    to build its keyword index — it can't be derived from embeddings.
    Rather than persisting the chunk list separately, we just read
    everything back out of the vector store, since Chroma already stores
    the original text + metadata alongside each embedding.
    """
    raw = vector_store.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]


def build_rag_chain(use_hybrid: bool = True):
    """
    Build the full RAG (Retrieval-Augmented Generation) pipeline.

    Args:
        use_hybrid: if True (default), use the hybrid retriever (BM25 +
            vector search + cross-encoder reranking). If False, use the
            original plain vector-similarity retriever — useful for
            before/after comparison via compare_retrievers.py.

    How RAG works (hybrid mode):
    1. User asks a question
    2. The question is run through BOTH keyword search (BM25) and vector
       search in parallel, and the two result sets are fused
    3. The fused candidates are reranked by a cross-encoder for true
       relevance (not just embedding similarity)
    4. The top-k reranked chunks are injected into the prompt as "Context"
    5. The LLM reads the context and generates an answer grounded in it

    In baseline mode, step 2-3 are replaced with a single plain vector
    similarity search returning the top-k chunks directly.
    """

    # Step 1: Load the vector store from disk
    vector_store = load_vector_store()

    # Step 2: Build the retriever — hybrid (BM25 + vector + rerank) or
    # plain vector search, depending on use_hybrid
    if use_hybrid:
        chunks = _reconstruct_documents(vector_store)
        retriever = build_hybrid_retriever(vector_store, chunks, k=4, fetch_k=20)
    else:
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

    # Step 3: Define a prompt template
    # {context} will be filled with the retrieved chunks
    # {question} will be filled with the user's question
    # The instruction to cite sources is key — it forces grounded answers
    prompt_template = """You are a helpful scientific research assistant.
Your job is to answer questions based ONLY on the provided document context.

Rules:
- If the answer is clearly in the context, answer it precisely and cite which page/section it came from.
- If the answer is NOT in the context, say exactly: "I cannot find this information in the uploaded documents."
- Do NOT make up information or use knowledge from outside the documents.
- Be concise but thorough. Use bullet points for multi-part answers.

Context from the documents:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Step 4: Load the local Mistral LLM via Ollama
    # temperature=0 -> deterministic answers (good for factual Q&A)
    # num_ctx=4096 -> context window size
    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0,
        num_ctx=4096
    )

    # Step 5: Assemble the chain
    # chain_type="stuff" means: stuff all retrieved chunks into a single prompt
    # return_source_documents=True -> we'll show the user which chunks were used
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return rag_chain


def ask_question(chain, question: str) -> dict:
    """
    Send a question through the RAG chain and return a structured response.
    """
    result = chain.invoke({"query": question})

    answer = result["result"]
    source_docs = result["source_documents"]

    # Extract useful metadata from source chunks
    sources = []
    for doc in source_docs:
        page = doc.metadata.get("page", "unknown")
        source_file = doc.metadata.get("source", "unknown")
        # Get just the filename, not the full temp path
        import os
        filename = os.path.basename(source_file) if source_file != "unknown" else "unknown"
        sources.append({
            "file": filename,
            "page": page + 1 if isinstance(page, int) else page,  # 0-indexed -> 1-indexed
            "preview": doc.page_content[:150].strip() + "..."
        })

    return {
        "answer": answer,
        "sources": sources
    }