"""
RAG Chain Module
Constructs the LangChain RAG pipeline integrating vector retrieval with Google Gemini LLM.
Enforces strict ground-truth constraints to eliminate hallucinations.
"""

from typing import Dict, Any, List, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from src.vector_store import retrieve_relevant_chunks

# System prompt forcing factual grounded answers only
SYSTEM_PROMPT = """You are a precise and helpful PDF Assistant. Your sole function is to answer questions using strictly the provided context from uploaded PDF documents.

CRITICAL INSTRUCTIONS & CONSTRAINTS:
1. Base your answer ONLY on the text provided in the "CONTEXT" section below.
2. Do NOT use outside knowledge, assume facts, or invent information not explicitly supported by the context.
3. If the answer cannot be determined or found from the provided CONTEXT, you MUST reply with EXACTLY this sentence and nothing else:
   "I couldn't find this information in the uploaded document."
4. Keep your answer clear, direct, and well-structured.
5. If the context partially answers the question, state what is available based strictly on the context and clarify what details are missing.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

# Fallback response constant requested in requirements
NOT_FOUND_RESPONSE = "I couldn't find this information in the uploaded document."


def format_retrieved_context(docs: List[Document]) -> str:
    """Formats retrieved document chunks into a clean, numbered context block for the prompt."""
    if not docs:
        return "No relevant context available."

    formatted_blocks = []
    for idx, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page", "N/A")
        content = doc.page_content.strip()
        formatted_blocks.append(f"[Chunk {idx} | Source: {source}, Page {page}]\n{content}")

    return "\n\n".join(formatted_blocks)


def execute_rag_pipeline(
    vector_store: Any,
    query: str,
    google_api_key: str,
    model_name: str = "gemini-1.5-flash",
    temperature: float = 0.2,
    top_k: int = 4
) -> Dict[str, Any]:
    """
    Executes the full RAG pipeline for a user query:
    1. Retrieves Top-K relevant chunks from vector store.
    2. Formats chunks as context into system prompt.
    3. Sends context and query to Google Gemini.
    4. Returns answer and source metadata.

    Args:
        vector_store: Active FAISS vector store.
        query: User input question.
        google_api_key: Valid Google Gemini API key.
        model_name: Gemini model ID (e.g., 'gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-2.0-flash').
        temperature: Model sampling temperature (0.0 to 1.0).
        top_k: Number of chunks to retrieve.

    Returns:
        Dict containing:
            - 'answer': Generated answer string
            - 'sources': List of retrieved (Document, distance_score) tuples
            - 'context_used': Formatted context string passed to LLM
    """
    if not google_api_key or not google_api_key.strip():
        return {
            "answer": "Error: Google Gemini API Key is missing. Please enter your key in the sidebar or set GOOGLE_API_KEY in the .env file.",
            "sources": [],
            "context_used": ""
        }

    # Step 1: Vector similarity retrieval
    retrieved_results: List[Tuple[Document, float]] = retrieve_relevant_chunks(
        vector_store=vector_store,
        query=query,
        top_k=top_k
    )

    if not retrieved_results:
        return {
            "answer": NOT_FOUND_RESPONSE,
            "sources": [],
            "context_used": ""
        }

    docs_only = [doc for doc, _score in retrieved_results]
    formatted_context = format_retrieved_context(docs_only)

    # Step 2: Initialize Gemini LLM with retry fallback for model names
    supported_models = [model_name, "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"]
    # De-duplicate preserving order
    unique_models = list(dict.fromkeys(supported_models))

    llm = None
    last_err = None
    for try_model in unique_models:
        try:
            llm = ChatGoogleGenerativeAI(
                model=try_model,
                google_api_key=google_api_key,
                temperature=temperature,
                max_retries=2
            )
            break
        except Exception as e:
            last_err = e
            continue

    if not llm:
        return {
            "answer": f"Error connecting to Gemini API: {str(last_err)}",
            "sources": retrieved_results,
            "context_used": formatted_context
        }

    # Step 3: Construct prompt template & chain using LCEL
    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    # Step 4: Invoke chain
    try:
        response_text = chain.invoke({
            "context": formatted_context,
            "question": query
        })
        clean_answer = response_text.strip() if response_text else NOT_FOUND_RESPONSE
        return {
            "answer": clean_answer,
            "sources": retrieved_results,
            "context_used": formatted_context
        }
    except Exception as e:
        err_msg = str(e)
        if "API_KEY_INVALID" in err_msg or "invalid API key" in err_msg.lower():
            return {
                "answer": "Error: Invalid Google Gemini API Key. Please verify your API key in the sidebar.",
                "sources": retrieved_results,
                "context_used": formatted_context
            }
        return {
            "answer": f"An error occurred while generating the answer: {err_msg}",
            "sources": retrieved_results,
            "context_used": formatted_context
        }
