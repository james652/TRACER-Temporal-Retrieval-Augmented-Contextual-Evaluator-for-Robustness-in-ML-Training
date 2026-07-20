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
        print(f"\n[ERROR] 컬렉션 '{COLLECTION_NAME}' 조회 실패: {e}")
        print("→ main.py를 먼저 실행해 papers 컬렉션을 생성/적재하세요.")
        sys.exit(1)

    total = collection.count()
    print(f"\n[OK] '{COLLECTION_NAME}' 총 chunk 수: {total}")
    if total == 0:
        print("→ 컬렉션은 존재하지만 비어 있습니다. main.py 적재를 다시 실행하세요.")
        sys.exit(1)

    ids = _fetch_all_ids(collection)
    pref = _prefix_count(ids)
    print("\n[ID Prefix별 chunk 개수]")
    for key in ["paper1", "paper2", "paper3", "paper4", "paper5"]:
        print(f"- {key}: {pref.get(key, 0)}")
    unknown = pref.get("unknown", 0)
    if unknown:
        print(f"- unknown: {unknown}")

    # 논문별 샘플 chunk 직접 확인
    rows = _fetch_all_rows(collection)
    print("\n[논문별 샘플 chunk 확인]")
    for key in ["paper1", "paper2", "paper3", "paper4", "paper5"]:
        picked = [r for r in rows if isinstance(r.get("id"), str) and r["id"].startswith(f"{key}_")]
        picked.sort(key=lambda r: _chunk_num(r.get("id", "")))
        print(f"\n{key}: {len(picked)} chunks")
        if not picked:
            print("- (없음) 해당 논문이 적재되지 않았거나 id_prefix가 다릅니다.")
            continue
        for r in picked[:2]:
            snippet = (r.get("doc") or "").replace("\n", " ")[:220]
            print(f"- {r['id']}: {snippet}...")

    # 샘플 semantic 검색 (OpenAI 키가 있을 때만)
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")):
        print("\n[SKIP] OPENAI_API_KEY가 없어 semantic query 검증은 건너뜁니다.")
        print("기본 적재 검증(count/prefix)은 완료되었습니다.")
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
        print(f"\n[ERROR] semantic query 실패: {e}")
        print("→ 임베딩 모델/차원 또는 API 키 상태를 확인하세요.")
        sys.exit(1)

    print(f"\n[Semantic Query] '{query_text}' top-5")
    ids = results.get("ids", [[]])[0]
    dists = results.get("distances", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    docs = results.get("documents", [[]])[0]

    if not ids:
        print("검색 결과가 없습니다.")
        return

    for i, doc_id in enumerate(ids):
        print(f"\n--- 결과 {i + 1} ---")
        print(f"ID: {doc_id}")
        if i < len(dists):
            print(f"distance: {dists[i]:.4f}")
        if i < len(metas):
            print(f"metadata: {metas[i]}")
        if i < len(docs) and isinstance(docs[i], str):
            print(f"chunk: {docs[i][:200].replace(chr(10), ' ')}...")

    print("\n[완료] 임베딩 적재/조회 검증 완료")


if __name__ == "__main__":
    main()
