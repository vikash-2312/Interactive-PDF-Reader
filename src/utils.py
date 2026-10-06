"""
Utility Functions Module
Provides helper functions for environment configuration, metadata formatting, and UI widgets.
"""

import os
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv


def load_api_key() -> Optional[str]:
    """Loads environment variables from .env file and retrieves GOOGLE_API_KEY."""
    load_dotenv(override=True)
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key and api_key != "your_gemini_api_key_here":
        return api_key
    return None


def mask_api_key(key: str) -> str:
    """Masks API key for secure display in UI (e.g., AIzaSy...x4Y)."""
    if not key or len(key) < 8:
        return "Not Set"
    return f"{key[:6]}...{key[-4:]}"


def format_source_citation(doc: Any, distance_score: float) -> Dict[str, Any]:
    """
    Formats a retrieved document chunk into clean metadata for Streamlit UI display.

    Args:
        doc: LangChain Document object.
        distance_score: Similarity distance score from FAISS.

    Returns:
        Formatted dictionary with source details.
    """
    metadata = getattr(doc, "metadata", {})
    return {
        "source_file": metadata.get("source", "Unknown PDF"),
        "page_number": metadata.get("page", "N/A"),
        "total_pages": metadata.get("total_pages", "N/A"),
        "chunk_id": metadata.get("chunk_id", "N/A"),
        "score": round(float(distance_score), 4),
        "content_snippet": getattr(doc, "page_content", "").strip()
    }
