# utils.py
# - Extract text from PDF
# - Token-based chunking (1000) + overlap (200)
# - Generate OpenAI embeddings
# - Batch-add to ChromaDB

from typing import List, Dict, Optional
import fitz  # PyMuPDF
import tiktoken
from openai import OpenAI
import math

# ---- 1) Extract text from PDF ------------------------------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract the full text of a PDF using PyMuPDF.
    """
    doc = fitz.open(pdf_path)
    parts = []
    for page in doc:
        parts.append(page.get_text())
    return "\n".join(parts)

# ---- 2) Token-based chunk splitting (1000 tokens, 200 overlap) ---------------
def chunk_by_tokens(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    model_encoding: str = "cl100k_base",
) -> List[str]:
    """
    Split precisely on a 'token' basis using tiktoken.
    - chunk_size: 1000
    - overlap: 200
    """
    assert chunk_size > 0 and overlap >= 0 and overlap < chunk_size, \
        "Must satisfy chunk_size>0, 0<=overlap<chunk_size."

    enc = tiktoken.get_encoding(model_encoding)
    tokens = enc.encode(text)

    chunks: List[str] = []
    start = 0
    n = len(tokens)

    while start < n:
        end = min(start + chunk_size, n)
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)

        if end >= n:
            break
        # Next start point: step back 200 tokens to account for the overlap
        start = end - overlap
        if start < 0:
            start = 0

    return chunks

# ---- 3) OpenAI embeddings ----------------------------------------------------
def embed_texts_openai(
    texts: List[str],
    model: str = "text-embedding-3-large",
    client: Optional[OpenAI] = None,
    batch_size: int = 64,
) -> List[List[float]]:
    """
    Generate OpenAI embeddings in batches and return them.
    - model: text-embedding-3-large (3072 dims) / text-embedding-3-small (1536 dims)
    - batch_size: number of inputs per API call
    - Uses the OPENAI_API_KEY environment variable
    """
    if client is None:
        client = OpenAI()

    vectors: List[List[float]] = []
    total = len(texts)
    if total == 0:
        return vectors

    num_batches = math.ceil(total / batch_size)
    for b in range(num_batches):
        s = b * batch_size
        e = min((b + 1) * batch_size, total)
        batch = texts[s:e]
        resp = client.embeddings.create(model=model, input=batch)
        # resp.data is returned in input order.
        vectors.extend([d.embedding for d in resp.data])
    return vectors

# ---- 4) Batch add: documents + embeddings into ChromaDB ----------------------
def add_chunks_with_embeddings(
    collection,
    chunks: List[str],
    metadata: Dict,
    id_prefix: str,
    openai_model: str = "text-embedding-3-large",
    batch_size: int = 64,
    client: Optional[OpenAI] = None,
) -> None:
    """
    - Generate OpenAI embeddings for chunks, and
    - batch-add them to the Chroma collection as (documents, embeddings, metadatas, ids).
    """
    if not chunks:
        return

    # Compute embeddings
    vectors = embed_texts_openai(chunks, model=openai_model, batch_size=batch_size,client=client)

    # Prepare metadatas & ids
    metadatas = [metadata for _ in chunks]
    ids = [f"{id_prefix}_chunk{i}" for i in range(len(chunks))]

    # Add to Chroma (can add all at once, or split into parts if large)
    collection.add(
        documents=chunks,
        embeddings=vectors,
        metadatas=metadatas,
        ids=ids,
    )
