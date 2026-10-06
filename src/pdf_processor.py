"""
PDF Processor Module
Handles PDF text extraction, validation, page-level tracking, and text chunking for RAG pipelines.
"""

import io
from typing import List, Tuple, Dict, Any
import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFProcessingError(Exception):
    """Custom exception raised during PDF processing failures."""
    pass


def extract_text_from_pdfs(uploaded_files: List[Any]) -> Tuple[List[Document], List[Dict[str, Any]]]:
    """
    Extracts text page-by-page from a list of uploaded PDF files (e.g., Streamlit UploadedFile objects).

    Args:
        uploaded_files: List of uploaded file objects with .name and bytes reader.

    Returns:
        Tuple containing:
            - List of LangChain Document objects (page_content + metadata)
            - List of file summary dictionaries (name, total_pages, extracted_chars, status)
    """
    documents: List[Document] = []
    file_summaries: List[Dict[str, Any]] = []

    if not uploaded_files:
        return documents, file_summaries

    for file in uploaded_files:
        filename = getattr(file, "name", "document.pdf")
        try:
            # Read bytes into BytesIO stream for pypdf
            file_bytes = file.read() if hasattr(file, "read") else file
            if hasattr(file, "seek"):
                file.seek(0)

            if not file_bytes:
                file_summaries.append({
                    "filename": filename,
                    "pages": 0,
                    "chars": 0,
                    "status": "Error: Empty file"
                })
                continue

            pdf_stream = io.BytesIO(file_bytes)
            reader = pypdf.PdfReader(pdf_stream)

            if reader.is_encrypted:
                try:
                    # Attempt decrypting with empty password
                    reader.decrypt("")
                except Exception:
                    file_summaries.append({
                        "filename": filename,
                        "pages": len(reader.pages) if reader.pages else 0,
                        "chars": 0,
                        "status": "Error: Password protected PDF"
                    })
                    continue

            total_pages = len(reader.pages)
            file_chars = 0
            file_pages_extracted = 0

            for page_idx, page in enumerate(reader.pages):
                page_number = page_idx + 1
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""

                clean_text = text.strip()
                if clean_text:
                    file_chars += len(clean_text)
                    file_pages_extracted += 1
                    doc = Document(
                        page_content=clean_text,
                        metadata={
                            "source": filename,
                            "page": page_number,
                            "total_pages": total_pages,
                        }
                    )
                    documents.append(doc)

            if file_chars == 0:
                file_summaries.append({
                    "filename": filename,
                    "pages": total_pages,
                    "chars": 0,
                    "status": "Warning: No extractable text found (scanned image or empty)"
                })
            else:
                file_summaries.append({
                    "filename": filename,
                    "pages": total_pages,
                    "chars": file_chars,
                    "status": "Success"
                })

        except Exception as e:
            file_summaries.append({
                "filename": filename,
                "pages": 0,
                "chars": 0,
                "status": f"Error parsing PDF: {str(e)}"
            })

    return documents, file_summaries


def split_text_into_chunks(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Splits document pages into smaller contiguous text chunks using RecursiveCharacterTextSplitter.

    Why chunking is required for RAG:
    1. LLM Context Window Limits: Prompt length limits prevent sending entire documents directly.
    2. Precision & Signal-to-Noise: Smaller chunks improve embedding resolution, pinpointing exact answers.
    3. Relevance Scoring: Vector similarity works best on coherent, focused paragraphs rather than giant books.
    4. Cost Efficiency: Avoids sending thousands of irrelevant tokens to the LLM API.

    Args:
        documents: List of page-level LangChain Documents.
        chunk_size: Target size of each chunk in characters.
        chunk_overlap: Overlap between consecutive chunks to preserve contextual boundary details.

    Returns:
        List of chunked Document objects with appended chunk metadata.
    """
    if not documents:
        return []

    # Enforce safe parameter ranges
    chunk_size = max(100, min(chunk_size, 4000))
    chunk_overlap = max(0, min(chunk_overlap, chunk_size // 2))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    # Decorate chunks with chunk IDs for precise source tracking
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
        chunk.metadata["chunk_size"] = len(chunk.page_content)

    return chunks
