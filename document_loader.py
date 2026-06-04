from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os


def load_and_chunk_pdf(file_path: str) -> list:
    """
    Load a PDF file and split it into smaller chunks for embedding.

    Why chunks? LLMs have a limited context window. By splitting the PDF
    into small pieces, we can search only the RELEVANT pieces instead of
    sending the whole document every time.
    """

    # Load the PDF — PyPDFLoader reads each page as a separate document
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    print(f"  Loaded {len(documents)} pages from: {os.path.basename(file_path)}")

    # Split into chunks
    # chunk_size=1000: each chunk is ~1000 characters (roughly half a page)
    # chunk_overlap=200: consecutive chunks share 200 characters at their edges
    #   → this prevents important sentences from being cut off at boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]  # tries to split at paragraph breaks first
    )

    chunks = splitter.split_documents(documents)
    print(f"  Split into {len(chunks)} chunks")

    return chunks