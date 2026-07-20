import os
import sys
from collections import Counter

import chromadb
from openai import OpenAI

HOST = "127.0.0.1"
PORT = 8000
COLLECTION_NAME = "papers"
EMBED_MODEL = "text-embedding-3-large"


def _list_collection_names(client):
    names = []
    try:
        cols = client.list_collections()
    except Exception:
        return names

    for c in cols:
        if hasattr(c, "name"):
            names.append(c.name)
        else:
            names.append(str(c))
    return names


def _fetch_all_ids(collection, batch_size=500):
    total = collection.count()
    ids = []
    offset = 0
    while offset < total:
        page = collection.get(limit=batch_size, offset=offset, include=["metadatas"])
        page_ids = page.get("ids", []) or []
        ids.extend(page_ids)
        offset += len(page_ids)
        if not page_ids:
            break
    return ids


def _fetch_all_rows(collection, batch_size=500):
    total = collection.count()
    rows = []
    offset = 0
    while offset < total:
        page = collection.get(limit=batch_size, offset=offset, include=["documents", "metadatas"])
        page_ids = page.get("ids", []) or []
        page_docs = page.get("documents", []) or []
        page_metas = page.get("metadatas", []) or []
        for i in range(len(page_ids)):
            rows.append({
                "id": page_ids[i],
                "doc": page_docs[i] if i < len(page_docs) else "",
                "meta": page_metas[i] if i < len(page_metas) else {},
            })
        offset += len(page_ids)
        if not page_ids:
            break
    return rows


def _prefix_count(ids):
    cnt = Counter()
    for x in ids:
        if isinstance(x, str) and "_" in x:
            cnt[x.split("_", 1)[0]] += 1
        else:
            cnt["unknown"] += 1
    return cnt


def _chunk_num(doc_id: str) -> int:
    # e.g., paper1_chunk12 -> 12
    if not isinstance(doc_id, str):
        return 10**9
    if "_chunk" not in doc_id:
        return 10**9
    tail = doc_id.rsplit("_chunk", 1)[-1]
    try:
        return int(tail)
    except Exception:
        return 10**9


def main():
    chroma_client = chromadb.HttpClient(host=HOST, port=PORT)
    all_collections = _list_collection_names(chroma_client)
    print(f"[Chroma] host={HOST} port={PORT}")
    print(f"[Chroma] collections={all_collections}")

    try:
        collection = chroma_client.get_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"\n[ERROR] Failed to look up collection '{COLLECTION_NAME}': {e}")
        print("-> Run main.py first to create/load the papers collection.")
        sys.exit(1)

    total = collection.count()
    print(f"\n[OK] '{COLLECTION_NAME}' total chunk count: {total}")
    if total == 0:
        print("-> The collection exists but is empty. Re-run the main.py loading step.")
        sys.exit(1)

    ids = _fetch_all_ids(collection)
    pref = _prefix_count(ids)
    print("\n[Chunk count per ID prefix]")
    for key in ["paper1", "paper2", "paper3", "paper4", "paper5"]:
        print(f"- {key}: {pref.get(key, 0)}")
    unknown = pref.get("unknown", 0)
    if unknown:
        print(f"- unknown: {unknown}")

    # Directly inspect sample chunks per paper
    rows = _fetch_all_rows(collection)
    print("\n[Sample chunk check per paper]")
    for key in ["paper1", "paper2", "paper3", "paper4", "paper5"]:
        picked = [r for r in rows if isinstance(r.get("id"), str) and r["id"].startswith(f"{key}_")]
        picked.sort(key=lambda r: _chunk_num(r.get("id", "")))
        print(f"\n{key}: {len(picked)} chunks")
        if not picked:
            print("- (none) This paper was not loaded, or its id_prefix differs.")
            continue
        for r in picked[:2]:
            snippet = (r.get("doc") or "").replace("\n", " ")[:220]
            print(f"- {r['id']}: {snippet}...")

    # Sample semantic search (only when an OpenAI key is available)
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")):
        print("\n[SKIP] OPENAI_API_KEY is not set, so semantic query verification is skipped.")
        print("Basic load verification (count/prefix) is complete.")
        return

    try:
        oa = OpenAI()
        query_text = "continual learning poisoning attack label flipping"
        emb = oa.embeddings.create(model=EMBED_MODEL, input=[query_text]).data[0].embedding
        results = collection.query(
            query_embeddings=[emb],
            n_results=5,
            include=["metadatas", "documents", "distances"],
        )
    except Exception as e:
        print(f"\n[ERROR] semantic query failed: {e}")
        print("-> Check the embedding model/dimension or the API key status.")
        sys.exit(1)

    print(f"\n[Semantic Query] '{query_text}' top-5")
    ids = results.get("ids", [[]])[0]
    dists = results.get("distances", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    docs = results.get("documents", [[]])[0]

    if not ids:
        print("No search results.")
        return

    for i, doc_id in enumerate(ids):
        print(f"\n--- Result {i + 1} ---")
        print(f"ID: {doc_id}")
        if i < len(dists):
            print(f"distance: {dists[i]:.4f}")
        if i < len(metas):
            print(f"metadata: {metas[i]}")
        if i < len(docs) and isinstance(docs[i], str):
            print(f"chunk: {docs[i][:200].replace(chr(10), ' ')}...")

    print("\n[Done] Embedding load/lookup verification complete")


if __name__ == "__main__":
    main()
