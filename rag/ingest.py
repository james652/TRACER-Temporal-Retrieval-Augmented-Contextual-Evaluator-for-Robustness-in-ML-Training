import chromadb
from openai import OpenAI
from utils import extract_text_from_pdf, chunk_by_tokens, add_chunks_with_embeddings

# (Optional) host/port can be specified
chroma_client = chromadb.HttpClient(host="127.0.0.1", port=8000)

# Create/get the collection
collection = chroma_client.get_or_create_collection(name="papers")

openai_client = OpenAI()

# ── RAG knowledge base: register one entry per source PDF ─────────────────────
# Add one dict per paper you want in the "papers" collection. Copy the paper1
# block below for each additional paper (paper2, paper3, …): set pdf_path to the
# PDF on your machine and fill in its metadata (title / authors / year).
# See the README for the list of papers used in the paper's knowledge base.
papers = [
    {
        "id_prefix": "paper1",
        "pdf_path": "./papers/your_paper.pdf",  # <-- put the path to your PDF here
        "metadata": {
            "title": "Paper title",
            "authors": "Author One; Author Two",
            "year": 2024,
        },
    },
    # Add more papers here, each following the paper1 template above.
]

def delete_existing_prefix(collection, id_prefix: str, batch_size: int = 500):
    """
    Delete existing chunks (paperX_chunk*) matching id_prefix to prevent
    duplication/contamination on re-runs.
    """
    total = collection.count()
    offset = 0
    to_delete = []
    while offset < total:
        page = collection.get(limit=batch_size, offset=offset, include=[])
        ids = page.get("ids", []) or []
        for _id in ids:
            if isinstance(_id, str) and _id.startswith(f"{id_prefix}_"):
                to_delete.append(_id)
        offset += len(ids)
        if not ids:
            break

    if to_delete:
        print(f"[+] Deleting {len(to_delete)} existing chunks for prefix '{id_prefix}'")
        collection.delete(ids=to_delete)
    else:
        print(f"[+] No existing chunks found for prefix '{id_prefix}'")


def process_pdf(pdf_path: str, meta: dict, id_prefix: str):
    print(f"[+] Extracting: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)

    print(f"[+] Chunking into 1000-token chunks with 200-token overlap...")
    chunks = chunk_by_tokens(text, chunk_size=1000, overlap=200)

    delete_existing_prefix(collection, id_prefix=id_prefix)

    print(f"[+] Adding {len(chunks)} chunks with OpenAI embeddings to Chroma ({id_prefix})")
    add_chunks_with_embeddings(
        collection=collection,
        chunks=chunks,
        metadata=meta,
        id_prefix=id_prefix,
        openai_model="text-embedding-3-large", 
        batch_size=64,
        client=openai_client,  
    )

for paper in papers:
    process_pdf(
        paper["pdf_path"],
        paper["metadata"],
        paper["id_prefix"],
    )

print(" All papers were successfully added to the Chroma DB.")
