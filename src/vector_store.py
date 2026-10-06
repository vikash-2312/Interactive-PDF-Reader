"""
Vector Store Module
Manages FAISS index creation, vector storage, and similarity retrieval for RAG.
"""

from typing import List, Tuple, Optional, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


class VectorStoreError(Exception):
    """Custom exception for vector store operations."""
    pass


def create_vector_store(
    chunks: List[Document],
    embedding_model: Any
) -> FAISS:
    """
    Creates a FAISS vector store from text chunks using the given embedding model.

    Args:
        chunks: List of chunked Document objects.
        embedding_model: Embeddings model instance.

    Returns:
        FAISS vector store instance.
    """
    if not chunks:
        raise VectorStoreError("Cannot build vector store from empty text chunks.")

    try:
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=embedding_model
        )
        return vector_store
    except Exception as e:
        raise VectorStoreError(f"Failed to create FAISS vector store: {str(e)}")


def retrieve_relevant_chunks(
    vector_store: FAISS,
    query: str,
    top_k: int = 4
) -> List[Tuple[Document, float]]:
    """
    Retrieves Top-K most relevant text chunks for a query along with distance scores.

    Args:
        vector_store: Active FAISS vector store.
        query: User search query string.
        top_k: Number of top documents to retrieve.

    Returns:
        List of tuples (Document, float_distance_score).
    """
    if not vector_store or not query.strip():
        return []

    try:
        # L2 or Cosine distance search in FAISS
        results_with_scores = vector_store.similarity_search_with_score(
            query=query,
            k=top_k
        )
        return results_with_scores
    except Exception as e:
        raise VectorStoreError(f"Error during vector similarity search: {str(e)}")
