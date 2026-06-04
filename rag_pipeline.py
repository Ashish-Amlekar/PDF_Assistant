from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from vector_store import load_vector_store

# The LLM we'll use for generating answers
LLM_MODEL = "mistral"


def build_rag_chain():
    """
    Build the full RAG (Retrieval-Augmented Generation) pipeline.

    How RAG works:
    1. User asks a question
    2. The question is converted to a vector (embedding)
    3. We search the vector store for the top-k most similar chunks
    4. Those chunks are injected into the prompt as "Context"
    5. The LLM reads the context and generates an answer grounded in it
    """

    # Step 1: Load the vector store from disk
    vector_store = load_vector_store()

    # Step 2: Create a retriever
    # search_type="similarity" — find chunks by cosine similarity to the query
    # k=4 — retrieve the top 4 most relevant chunks
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
    # temperature=0 → deterministic answers (good for factual Q&A)
    # num_ctx=4096 → context window size
    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0,
        num_ctx=4096
    )

    # Step 5: Assemble the chain
    # chain_type="stuff" means: stuff all retrieved chunks into a single prompt
    # return_source_documents=True → we'll show the user which chunks were used
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
            "page": page + 1,  # pages are 0-indexed, humans expect 1-indexed
            "preview": doc.page_content[:150].strip() + "..."
        })

    return {
        "answer": answer,
        "sources": sources
    }