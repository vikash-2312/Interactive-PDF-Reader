"""
Embeddings Module
Provides embedding models for semantic vector representations of text chunks.
Supports local HuggingFace models (sentence-transformers) and Google Gemini Embeddings.
"""

import os
from typing import Optional, Any


def get_embedding_model(
    provider: str = "HuggingFace (all-MiniLM-L6-v2)",
    google_api_key: Optional[str] = None
) -> Any:
    """
    Instantiates and returns the configured LangChain Embeddings model.

    Args:
        provider: Choice of embedding model provider.
        google_api_key: Google Gemini API key (required if provider is Google Gemini).

    Returns:
        Embeddings model object implementing Embeddings interface.
    """
    api_key = google_api_key or os.getenv("GOOGLE_API_KEY")

    if provider.startswith("Google"):
        if not api_key:
            raise ValueError(
                "Google API Key is required for Google Generative AI Embeddings. "
                "Please provide it in the sidebar or set GOOGLE_API_KEY in .env file."
            )
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=api_key
            )
        except ImportError:
            raise ImportError(
                "langchain-google-genai is required for Google Embeddings. "
                "Install it using `pip install langchain-google-genai`."
            )
    else:
        # Default: HuggingFace sentence-transformers/all-MiniLM-L6-v2
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
        except ImportError:
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                return HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True}
                )
            except ImportError:
                raise ImportError(
                    "langchain-huggingface or sentence-transformers is required for local embeddings. "
                    "Install using `pip install langchain-huggingface sentence-transformers`."
                )
