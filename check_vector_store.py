"""
Lists which source PDF(s) are actually stored in ./chroma_db, and how
many chunks came from each. Run this whenever retrieval seems to be
pulling from the wrong document.

Usage: python check_vector_store.py
"""

from collections import Counter
import os

from vector_store import load_vector_store


def main():
    vector_store = load_vector_store()
    raw = vector_store.get(include=["metadatas"])
    metadatas = raw["metadatas"]

    print(f"Total chunks in vector store: {len(metadatas)}\n")

    sources = Counter(
        os.path.basename(m.get("source", "unknown")) if m else "unknown"
        for m in metadatas
    )

    print("Chunks per source file:")
    for filename, count in sources.most_common():
        print(f"  {count:5d}  {filename}")


if __name__ == "__main__":
    main()