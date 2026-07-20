# TRACER

**Temporal Retrieval-Augmented Contextual Evaluator for Robustness in ML Training**

TRACER is an LLM-agent framework that **orchestrates ML data-poisoning / backdoor / adversarial
attack pipelines, reads their execution logs, and decides *which attack occurred* (if any).**
Each step is summarized with a memory-carrying LLM chain, grounded against a RAG knowledge base
of security papers, scored with RAGAS-style metrics, and the most salient terms are extracted and
**injected into the next step's retrieval query** (the "temporal" feedback loop).

---

## How it works

```
                 ┌───────────────────────────── per step ─────────────────────────────┐
 attack pipeline │  run shell command ──► step log                                     │
 (StepSpec list) │        │                                                            │
                 │        ▼                                                             │
                 │  ┌───────────┐   ┌──────────────┐   ┌───────────────┐   ┌─────────┐ │
                 │  │  RAG      │──►│  Summarizer   │──►│  RAGAS score  │──►│ Top-K   │ │
                 │  │ (Chroma)  │   │ (LLM+memory)  │   │ faith/rel/ctx │   │ extract │ │
                 │  └───────────┘   └──────────────┘   └───────────────┘   └────┬────┘ │
                 │        ▲                                                     │       │
                 │        └──────── inject Top-K into NEXT step's query ◄───────┘       │
                 └────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
                                   attack verdict + JSON report
```

Core components (`src/Agent_v12.py`):
- **`MemoryManager`** — per-session summary history via LangChain `FileChatMessageHistory`.
- **`RAGRetriever`** — queries a ChromaDB `papers` collection, synthesizes evidence with the LLM.
- **`Summarizer`** — builds each step summary from *previous summaries + RAG snippet + Top-K + log tail*.
- **`RAGASEvaluator`** — faithfulness (LLM judge), answer relevance (embedding cosine), context relevance.
- **Top-K injection** — extracts salient terms from `(summary + log)` and merges them into the next step's RAG query.
- **Vision path** — for MMD-backdoor pipelines, analyzes figure PNGs with GPT vision.

---

## Repository layout

```
TRACER/
├── src/
│   ├── Agent_v12.py         # main entry point (interactive CLI)
│   ├── attack_spec.py       # LOG_DIR + StepSpec + all build_*_specs() (attack pipeline definitions)
│   ├── detection_prompt.py  # DETECTION_SYSTEM_PROMPT (LLM system prompt)
│   └── ablation/            # ablation variants (no-memory / no-topk / no-rag / no-instruct / gpt4.1 / sweeps)
├── rag/
│   ├── ingest.py            # build the ChromaDB "papers" collection from PDFs
│   ├── utils.py             # PDF extract + token chunking + OpenAI embedding helpers
│   └── chromacheck.py       # inspect the collection
├── logs/                    # runtime output (git-ignored) — see "Outputs" below
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. API key
```bash
cp .env.example .env      # then edit, or just:
export OPENAI_API_KEY="sk-..."
```

### 3. Start the ChromaDB server (required for RAG)
The agent connects to a Chroma HTTP server at `127.0.0.1:8000`:
```bash
chroma run --host 127.0.0.1 --port 8000 --path ./chroma_db
```

### 4. Build the RAG knowledge base
Source papers are **not** shipped in this repo (copyright). Download the PDFs listed in
[`rag/ingest.py`](rag/ingest.py) — BrainWash (CVPR'24), Accumulative Poisoning (NeurIPS'21),
Multi-Level MMD Regularization, Neural Relation Graph, Rethinking Label Poisoning for GNNs,
"Explaining and Harnessing Adversarial Examples", PGD (Madry et al.), PhysPatch — place them where
`ingest.py` expects, then:
```bash
python rag/ingest.py       # extracts, chunks (1000-tok / 200 overlap), embeds → "papers" collection
python rag/chromacheck.py  # sanity check
```

---

## ⚠️ External attack implementations (must be provided separately)

TRACER is an **orchestrator**. The commands built in `src/attack_spec.py` invoke separate attack
codebases by **absolute path**, e.g.:

```
/home/jun/work/soongsil/Brainwash/         (main_baselines.py, main_inv.py, main_brainwash.py)
/home/jun/work/soongsil/PoisoningAttack/   (AccumulativeAttack/*)
/home/jun/work/soongsil/backdoor/Multi-Level-MMD-Regularization/
/home/jun/work/soongsil/Detect/ , .../2nd/
```

These projects, their datasets, and their model checkpoints (`.pkl`, hundreds of MB) are **not**
part of this repository. Before running the orchestration paths you must:

1. Provide those attack projects locally, **and**
2. Edit the absolute paths / `CUDA_VISIBLE_DEVICES` / `LOG_DIR` at the top of `src/attack_spec.py`
   to match your machine.

The `Analysis` paths (below) only need the step logs, so they can be run without re-executing attacks.

---

## Running

```bash
cd src
python Agent_v12.py
# prompt: which program?
#   Brainwash / brainwash_cifar10 / brainwash_miniimagenet / brainwash_tinyimagenet
#   Accumulative / accumulative_cifar100
#   MMD_backdoor / MMD_backdoor_cifar100
#   Detect / Rethink / Rethink_pub / FGSM / PGD / PhysPatch
#   Analysis / Analysis_Accumulative   ← analyze existing logs only (no attack re-run)
```

### Ablation
Variants live in `src/ablation/` and import `attack_spec` / `detection_prompt`, so run them with
`src` on the path:
```bash
PYTHONPATH=src python src/ablation/Agent_without_memory.py
PYTHONPATH=src python src/ablation/Agent_without_topk.py
# aggregate results:
python src/ablation/ablation_extract.py
python src/ablation/ablation_compare.py
```
> `ablation_extract.py` / `ablation_compare.py` read result dirs (`A_result*`) produced by the runs —
> adjust those paths for your setup.

---

## Outputs

Everything is written under `LOG_DIR` (default `.../logs`, set in `src/attack_spec.py:21`) and is
**git-ignored**:

| Path | Contents |
|------|----------|
| `logs/*.log` | raw stdout/stderr of each attack step (`step1.log`, `mini_step*.log`, …) |
| `logs/memory/<session>.json` | LangChain per-session summary memory |
| `logs/LLM_topk/*.json` | extracted Top-K terms + next-step injection records |
| `logs/vision/*.json` | GPT-vision analysis of MMD figures |
| `logs/monitor_summary_<tag>_step*.json` | per-step reports |
| `logs/monitor_summary_<tag>.json`, `logs/analysis*.json` | final aggregated reports |

---

## Environment variables

| Var | Default | Meaning |
|-----|---------|---------|
| `OPENAI_API_KEY` | — | required (falls back to `OPENAI_APIKEY`) |
| `USE_RAG` | `1` | query ChromaDB for evidence |
| `USE_TOPK` | `1` | extract Top-K and inject into next step |
| `USE_VISION` | `1` | analyze MMD figure PNGs with vision |
| `TOPK_K` | `5` | number of Top-K terms/phrases |
| `LOG_TAIL` | `20000` | chars of log tail sent to the LLM |
| `RAGAS_EMBED_MODEL` | `text-embedding-3-small` | embedding model for RAGAS answer-relevance |

---

## Citation

If you use this code, please cite the paper (add your BibTeX here).
