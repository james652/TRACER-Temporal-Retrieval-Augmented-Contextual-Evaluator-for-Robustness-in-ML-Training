import chromadb
from openai import OpenAI
from utils import extract_text_from_pdf, chunk_by_tokens, add_chunks_with_embeddings

# (선택) 호스트/포트 지정 가능
chroma_client = chromadb.HttpClient(host="127.0.0.1", port=8000)

# 컬렉션 생성/가져오기
collection = chroma_client.get_or_create_collection(name="papers")

openai_client = OpenAI()

papers = [
    {
        "id_prefix": "paper1",
        "pdf_path": "/home/jun/work/soongsil/Agent/ChromaDB/paper/Abbasi_BrainWash_A_Poisoning_Attack_to_Forget_in_Continual_Learning_CVPR_2024_paper.pdf",
        "metadata": {
            "title": "BrainWash: A Poisoning Attack to Forget in Continual Learning",
            "authors": "Ali Abbasi; Parsa Nooralinejad; Hamed Pirsiavash; Soheil Kolouri",
            "year": 2024,
        },
    },
    {
        "id_prefix": "paper2",
        "pdf_path": "/home/jun/work/soongsil/Agent/ChromaDB/paper/NeurIPS-2021-accumulative-poisoning-attacks-on-real-time-data-Paper.pdf",
        "metadata": {
            "title": "Accumulative Poisoning Attacks on Real-time Data",
            "authors": "Tianyu Pang; Xiao Yang; Yinpeng Dong; Hang Su; Jun Zhu",
            "year": 2021,
        },
    },
    {
        "id_prefix": "paper3",
        "pdf_path": "/home/jun/work/soongsil/Agent/ChromaDB/paper/Enhancing Backdoor Attacks with Multi-Level MMD Regularization.pdf",
        "metadata": {
            "title": "Enhancing Backdoor Attacks with Multi-Level MMD Regularization",
            "authors": "Pengfei Xia; Hongjing Niu; Ziqiang Li; Bin Li",
            "year": 2023,
        },
    },
    {
        "id_prefix": "paper4",
        "pdf_path": "/home/jun/work/soongsil/Agent/ChromaDB/paper/Neural Relation Graph A Unified Framework for Identifying Label Noise and Outlier Data.pdf",
        "metadata": {
            "title": "Neural Relation Graph: A Unified Framework for Identifying Label Noise and Outlier Data",
            "authors": "Jang-Hyun Kim; Sangdoo Yun; Hyun Oh Song",
            "year": 2023,
        },
    },
    {
        "id_prefix": "paper5",
        "pdf_path": "/home/jun/work/soongsil/Agent/ChromaDB/paper/Rethinking_Label_Poisoning_for_GNNs_Pitfalls_and_Attacks.pdf",
        "metadata": {
            "title": "Rethinking Label Poisoning for GNNs: Pitfalls and Attacks",
            "authors": "Vijay Lingam; MohammadSadegh Akhondzadeh; Aleksandar Bojchevski",
            "year": 2024,
        },
    },
    {
        "id_prefix": "paper6",
        "pdf_path": "/home/jun/work/soongsil/Agent/ChromaDB/paper/EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES.pdf",
        "metadata": {
            "title": "Explaining and Harnessing Adversarial Examples",
            "authors": "Ian J. Goodfellow; Jonathon Shlens; Christian Szegedy",
            "year": 2014,
        },
    },
    {
        "id_prefix": "paper7",
        "pdf_path": "/home/jun/work/soongsil/Agent/ChromaDB/paper/Towards Deep Learning Models Resistant to Adversarial Attacks.pdf",
        "metadata": {
            "title": "Towards Deep Learning Models Resistant to Adversarial Attacks",
            "authors": "Aleksander Madry; Aleksandar Makelov; Ludwig Schmidt; Dimitris Tsipras; Adrian Vladu",
            "year": 2017,
        },
    },
    {
        "id_prefix": "paper8",
        "pdf_path": "/home/jun/work/soongsil/Agent/ChromaDB/paper/PhysPatch A Physically Realizable and Transferable Adversarial Patch Attack for Multimodal Large Language Models-based Autonomous Driving Systems.pdf",
        "metadata": {
            "title": "PhysPatch: A Physically Realizable and Transferable Adversarial Patch Attack for Multimodal Large Language Models-based Autonomous Driving Systems",
            "authors": "Qi Guo; Xiaojun Jia; Shanmin Pang; Simeng Qin; Lin Wang; Ju Jia; Yang Liu; Qing Guo",
            "year": 2026,
        },
    },
]

def delete_existing_prefix(collection, id_prefix: str, batch_size: int = 500):
    """
    id_prefix에 해당하는 기존 chunk(paperX_chunk*)를 삭제해서 재실행 시 중복/오염을 방지한다.
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

print(" 모든 논문이 Chroma DB에 성공적으로 추가되었습니다.")
