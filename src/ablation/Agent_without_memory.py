#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Agent Runner (ABLATION: NO MEMORY) — RAG + RAGAS + LLM Top-K injection
-------------------------------------------------------------------------------
ABLATION VARIANT: the cross-step MEMORY module is removed from this file.
For the full pipeline (with memory), use Tracer_Agent.py.

What "no memory" means here (single-variable ablation):
- No persistent FileChatMessageHistory: nothing is stored or recalled.
- The LangChain conversational-history placeholder is always EMPTY.
- previous summaries are never injected (prev_summaries == [] -> "(none)").
- Each step is analyzed INDEPENDENTLY using only [current RAG] + [current log]
  + [current Top-K]. RAG / Top-K / RAGAS / Vision are UNCHANGED vs. Tracer_Agent.py.

- Summarization uses: [current RAG] + [current log]  (no previous summaries)
- Optional Chroma RAG and RAGAS-style metrics
- NEW: After each step, extract Top-K terms/phrases from (summary+log) and
       inject them into the NEXT step's rag_request automatically
- Backward-compatible CLI (Brainwash/Accumulative/Test/Analysis/Analysis_Accumulative)

Env:
  OPENAI_API_KEY (or OPENAI_APIKEY)
  USE_RAG=0/1      (default 1)
  USE_TOPK=0/1     (default 1)
  TOPK_K=int       (default 10)
  LOG_TAIL=int     (default 20000)
  RAGAS_EMBED_MODEL (default "text-embedding-3-small")
"""

from __future__ import annotations

import os
import re
import json
import time
import glob
import math
import subprocess
import base64
from typing import Tuple
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ------------------------------------------------------------------------------------
# Attack specs (external module)
# ------------------------------------------------------------------------------------
from attack_spec import (
    LOG_DIR,
    StepSpec,  # dataclass(title, command, log_path, analysis_prompt?, rag_request?, expected_artifacts?, timeout_sec?)
    build_brainwash_specs,
    build_brainwash_miniimagenet_specs,
    build_brainwash_tinyimagenet_specs, 
    build_brainwash_cifar10_specs,
    build_accumulative_specs,
    build_accumulative_cifar100_specs, 
    build_analyze_brainwash_specs,
    build_analyze_accumulative_specs,
    build_mmd_backdoor_specs, 
    build_mmd_backdoor_cifar100_specs,
    build_detect_specs, 
    build_rethink_specs,
    build_rethink_pubmed_specs,
)

# ------------------------------------------------------------------------------------
# Optional OpenAI (guarded import)
# ------------------------------------------------------------------------------------
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    OpenAI = None  # type: ignore
    _OPENAI_AVAILABLE = False

# ------------------------------------------------------------------------------------
# LangChain (memory)
# ------------------------------------------------------------------------------------
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.pop("LANGSMITH_API_KEY", None)  # avoid tracer surprises

from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import FileChatMessageHistory
# [ABLATION: NO MEMORY] non-persistent, in-memory history used as an always-empty
# stand-in so the chain structure stays identical to Tracer_Agent.py while no state
# carries across steps. Import path differs across langchain versions -> fallback.
try:
    from langchain_core.chat_history import InMemoryChatMessageHistory as _EmptyChatHistory
except Exception:  # older langchain
    from langchain_community.chat_message_histories import ChatMessageHistory as _EmptyChatHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda


# ====================================================================================
# Utilities
# ====================================================================================
def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _extract_summary_text(obj: Any) -> str:
    """Prefer explicit summary, then output, then compact JSON string."""
    if not isinstance(obj, dict):
        return str(obj or "")

    summary = obj.get("summary")
    if isinstance(summary, str) and summary.strip():
        return summary

    output = obj.get("output")
    if isinstance(output, str) and output.strip():
        return output

    return json.dumps(obj, ensure_ascii=False)


def _format_topk_text(topk_obj: Dict[str, Any]) -> str:
    terms = topk_obj.get("terms") or []
    phrases = topk_obj.get("phrases") or []
    hint = (topk_obj.get("query_hint") or "").strip()
    t = ", ".join([str(x) for x in terms if str(x).strip()])
    p = ", ".join([str(x) for x in phrases if str(x).strip()])
    return f"terms: {t}; phrases: {p}; query_hint: {hint}"


def _build_rag_support_text(rag_obj: Dict[str, Any], rag_answer: str, max_chars: int = 800) -> str:
    ans = (rag_answer or "").strip()
    sources = (rag_obj or {}).get("sources") or []
    if ans:
        src_parts = []
        for s in sources[:2]:
            sid = s.get("id", "?")
            meta = s.get("meta") or {}
            title = str(meta.get("title") or "unknown")
            year = str(meta.get("year") or "-")
            src_parts.append(f"[{sid}] {title} ({year})")
        src_txt = "; ".join(src_parts) if src_parts else "(source metadata unavailable)"
        return f"{ans[:max_chars]}\nSources: {src_txt}"

    err = str((rag_obj or {}).get("error") or "").strip()
    if err:
        return f"No RAG evidence retrieved (reason: {err})."
    return "No RAG evidence retrieved."


def _ensure_rag_grounding(summary_obj: Dict[str, Any], rag_obj: Dict[str, Any], rag_answer: str) -> Dict[str, Any]:
    if not isinstance(summary_obj, dict):
        return summary_obj

    cur = str(summary_obj.get("supporting_evidence_from_rag") or "").strip()
    cur_l = cur.lower()
    is_empty_or_placeholder = (
        (not cur)
        or ("no rag evidence" in cur_l)
        or ("not provided" in cur_l)
        or ("empty if none" in cur_l)
    )

    if is_empty_or_placeholder:
        summary_obj["supporting_evidence_from_rag"] = _build_rag_support_text(rag_obj, rag_answer)
    return summary_obj


# ====================================================================================
# Memory Manager
# ====================================================================================
class MemoryManager:
    """[ABLATION: MEMORY REMOVED]

    Drop-in, interface-compatible replacement for the persistent
    FileChatMessageHistory-backed memory used in Tracer_Agent.py. Every method is
    neutralized so that NO state is stored or recalled across steps:

    - history():               returns a FRESH, EMPTY in-memory history each call,
                               so the LangChain conversational-history placeholder
                               is always empty and nothing persists across steps.
    - save_summary():          no-op (summaries are never written anywhere).
    - get_previous_summaries(): always returns [] (no cross-step recall).

    Net effect: each step is analyzed independently from only the CURRENT RAG
    snippet + log + Top-K. All other components (RAG / Top-K / RAGAS / Vision)
    and the prompt/chain are byte-identical to Tracer_Agent.py.
    """

    def __init__(self, log_dir: str):
        # Kept for interface compatibility; no memory directory is used.
        self.mem_dir = log_dir

    def store_path(self, session_id: str) -> str:
        # No persistent store in the no-memory ablation.
        return ""

    def history(self, session_id: str):
        # Fresh empty history every call -> RunnableWithMessageHistory sees no
        # prior messages, and anything it appends is discarded (never reused).
        return _EmptyChatHistory()

    def save_summary(self, session_id: str, step_title: str, summary_text: str) -> None:
        # [ABLATION] do not persist any summary.
        return

    def get_previous_summaries(self, session_id: str, max_items: int = 5) -> List[str]:
        # [ABLATION] no recall -> callers receive an empty list ("(none)").
        return []


# ====================================================================================
# RAG Retriever (Chroma, optional)
# ====================================================================================
class RAGRetriever:
    """Queries Chroma and asks LLM to synthesize a short, evidence-grounded snippet."""

    def __init__(self, openai_client: Optional[OpenAI], model: str = "gpt-5"):
        self.client = openai_client
        self.model = model

    def analyze(
        self,
        rag_query: str,
        log_text: str,
        chroma_collection_name: str = "papers",
        chroma_host: str = "127.0.0.1",
        chroma_port: int = 8000,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        if self.client is None:
            return {"ok": False, "answer": "", "sources": [], "error": "OpenAI client unavailable or API key not set"}

        try:
            import chromadb
            chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
            collection = chroma_client.get_collection(name=chroma_collection_name)
        except Exception as e:
            return {"ok": False, "answer": "", "sources": [], "error": f"ChromaDB connection failed: {e}"}

        query = (rag_query or "").strip()
        if not query:
            tail = (log_text or "")[-2000:]
            query = tail[:200] if tail else "poisoning attack analysis"

        # Embed & query
        try:
            from chromadb.utils import embedding_functions
            ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY"),
                model_name="text-embedding-3-large",
            )
            q_emb = ef([query])
            qres = collection.query(query_embeddings=q_emb, n_results=top_k)
            docs = qres.get("documents", [[]])[0]
            ids = qres.get("ids", [[]])[0]
            metas = qres.get("metadatas", [[]])[0]
            sources = [{"id": i, "doc": d, "meta": m} for i, d, m in zip(ids, docs, metas)]
        except Exception as e:
            return {"ok": False, "answer": "", "sources": [], "error": f"ChromaDB query error: {e}"}

        kb_context = "\n---\n".join([
            f"[{s['id']}] {s['meta'].get('title','(no title)')} ({s['meta'].get('year','-' )})\n{s['doc']}"
            for s in sources
        ]) if sources else "(no kb match)"

        tail = (log_text or "")[-2000:]
        user_msg = (
            "The following contains excerpts from the knowledge base and execution logs.\n"
            "Evaluate attack/collapse indicators with concrete evidence and numbers.\n"
            f"[RAG Query]\n{query}\n\n[KB Snippets]\n{kb_context}\n\n[Log Tail]\n{tail}"
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a strict ML security analyst. Be concise, cite numbers."},
                    {"role": "user", "content": user_msg},
                ],
            )
            ans = (resp.choices[0].message.content or "").strip()
            return {"ok": True, "answer": ans, "sources": sources, "error": None}
        except Exception as e:
            return {"ok": False, "answer": "", "sources": sources, "error": f"gpt_request_error: {e}"}


# ====================================================================================
# Summarizer (LangChain chain with memory)
# ====================================================================================
class Summarizer:
    """Builds step summary from [previous summaries] + [RAG snippet] + [log tail]."""

    def __init__(self, memory_mgr: MemoryManager, model: str = "gpt-5", temperature: float = 1.0):
        self.memory_mgr = memory_mgr
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.parser = JsonOutputParser()

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                (
                    "[ROLE]\n"
                    "You are a continual-learning poisoning detection analyst specializing in machine learning security.\n"
                    "Your job is to analyze experimental results and identify which attack occurred.\n\n"

                    "[TASK]\n"
                    "Given the following evidence:\n"
                    "- task-wise accuracies over time (A_{{t,i}}),\n"
                    "- backward transfer (BWT) or forgetting trends,\n"
                    "- model logs, RAG evidence, prior summaries,\n"
                    "- and LLM-extracted Top-K terms/phrases,\n"
                    "infer exactly ONE scenario from the list below:\n"
                    "- Brainwash\n"
                    "- Accumulative Attack\n"
                    "- Label Flipping\n"
                    "- MMDRegularization\n"
                    "- No Attack (clean scenario)\n\n"

                    "[BRAINWASH DETECTION RULES]\n"
                    "Brainwash is a poisoning attack on the MOST RECENT task designed to induce catastrophic forgetting\n"
                    "of PREVIOUS tasks while keeping the last-task accuracy relatively high.\n"
                    "Decide Brainwash if ALL of the following hold:\n"
                    "1) Accuracies of earlier tasks drop after learning the last task,\n"
                    "   while the accuracy of the last task remains unusually high.\n"
                    "2) BWT becomes much more negative (or forgetting sharply increases)\n"
                    "immediately after training on the last task.\n"

                    "[ACCUMULATIVE ATTACK DETECTION RULES]\n"
                    "Accumulative attacks exhibit a delayed threshold effect.\n"
                    "Decide Accumulative Attack if:\n"
                    "1) Accuracy remains relatively stable for a period, then suddenly drops sharply.\n"
                    "2) The drop is global and affects many or all tasks simultaneously.\n\n"

                    "[LABEL FLIPPING DETECTION RULES]\n"
                    "Decide Label Flipping if label-noise metrics provide strong evidence:\n"
                    "- ROC-AUC ≥ 0.80,\n"
                    "- Average Precision (AP) is significantly higher than the base noise rate (8%),\n"
                    "- TNR@95 is stable and around 0.70 or higher.\n\n"

                    "[MMDREGULARIZATION DETECTION RULES]\n"
                    "Decide MMDRegularization if the observed behavior is best explained by\n"
                    "MMD-based regularization or distribution-alignment effects rather than poisoning.\n\n"

                    "[NO ATTACK RULES]\n"
                    "Decide No Attack if behavior matches normal continual learning trends\n"
                    "and no strong poisoning signatures are present.\n\n"

                    "[CONFLICT RESOLUTION POLICY]\n"
                    "Resolve conflicts using the following priority:\n"
                    "1) Task-wise accuracy pattern and timing (Brainwash vs. Accumulative).\n"
                    "2) Magnitude and timing of BWT / forgetting.\n"
                    "3) ROC-AUC, AP, and TNR@95.\n"
                    "4) Top-K terms/phrases are supportive only (they must not override numeric evidence).\n\n"

                    "[DECISION CONSTRAINTS]\n"
                    "- Do not hedge or use uncertain language.\n"
                    "- Be specific and conclusive.\n"
                    "- Choose exactly ONE attack type.\n"
                    "- If any required metric is missing, explicitly state what is missing and how it limits the conclusion.\n\n"

                    "[TOP-K USAGE RULE]\n"
                    "You are given LLM Top-K terms/phrases intended for retrieval/query hints.\n"
                    "They are NOT ground truth. Use them only as supporting context.\n"
                    "However, you MUST include them verbatim in the final output under a [TOPK] section inside evidence_for_attack.\n\n"

                    "[RAG SUPPORT REQUIREMENT]\n"
                    "If RAG snippet is provided, supporting_evidence_from_rag MUST quote concrete claims from it and include source ids/titles when available.\n"
                    "If RAG snippet is empty, write a short reason why evidence is unavailable (e.g., retrieval disabled, kb match missing, or retrieval error).\n\n"

                    "[OUTPUT FORMAT]\n"
                    "Respond ONLY in VALID JSON (no markdown, no extra text). Use exactly these keys:\n"
                    "- attack_identified: one of [Brainwash, Accumulative Attack, Label Flipping, MMDRegularization, No Attack]\n"
                    "- evidence_for_attack: string\n"
                    "- risk_level: integer from 1 (Safe) to 5 (High Risk)\n"
                    "- rationale_for_decision: string\n"
                    "- supporting_evidence_from_rag: string\n\n"

                    "[EVIDENCE_FOR_ATTACK REQUIREMENT]\n"
                    "The evidence_for_attack string MUST contain TWO sections in this exact order:\n"
                    "1) [NUMERIC_EVIDENCE] ... (cite concrete numbers from logs/accuracies/BWT)\n"
                    "2) [TOPK] ... (copy/paste the provided Top-K terms/phrases/query_hint)\n\n"

                    "Example JSON schema (do not copy values, only follow the structure):\n"
                    "{{\n"
                    "  \"attack_identified\": \"No Attack\",\n"
                    "  \"evidence_for_attack\": \"[NUMERIC_EVIDENCE] ...\\n[TOPK] terms: ...; phrases: ...; query_hint: ...\",\n"
                    "  \"risk_level\": 1,\n"
                    "  \"rationale_for_decision\": \"...\",\n"
                    "  \"supporting_evidence_from_rag\": \"...\"\n"
                    "}}\n"
                ),
            ),
            MessagesPlaceholder(variable_name="history"),
            (
                "human",
                (
                    #"Step: {step_title}\n\n"
                    "Previous step summaries (max 5):\n{prev_summaries}\n\n"
                    "RAG snippet (empty if none):\n{rag_text}\n\n"
                    "LLM Top-K (empty if none):\n{topk_text}\n\n"
                    "Current log (last 200k chars):\n{log_text}\n\n"
                    "Return the JSON now."
                ),
            ),
        ])


        def _ensure_output_key(x: Any) -> Dict[str, Any]:
            if isinstance(x, dict):
                if "output" not in x:
                    out = x.get("summary") or json.dumps(x, ensure_ascii=False)
                    return {"output": out, **x}
                return x
            return {"output": str(x)}

        self.chain_core = (self.prompt | self.llm | self.parser) | RunnableLambda(_ensure_output_key)

    def summarize(
        self,
        session_id: str,
        step_title: str,
        rag_text: str,
        prev_summaries: List[str],
        log_text: str,
        topk_text: str = ""
    ) -> Dict[str, Any]:
        with_history = RunnableWithMessageHistory(
            self.chain_core,
            lambda session_id=session_id: self.memory_mgr.history(session_id),
            input_messages_key="step_title",
            history_messages_key="history",
        )

        res = with_history.invoke(
            {
                "step_title": step_title,
                "prev_summaries": ("\n- " + "\n- ".join(prev_summaries)) if prev_summaries else "(none)",
                "rag_text": rag_text or "",
                "topk_text": topk_text or "",
                "log_text": (log_text or "")[-200000:],
            },
            config={"configurable": {"session_id": session_id}},
        )

        for k, default in [
            ("step_executed", None), ("errors", []), ("warnings", []),
            ("evidence", []), ("summary", ""), ("output", "")
        ]:
            res.setdefault(k, default)
        return res


# ====================================================================================
# RAGAS Evaluator
# ====================================================================================
class RAGASEvaluator:
    """Computes faithfulness (LLM judge), answer relevance (embed cosine), context relevance (LLM judge)."""

    def __init__(self, openai_client: Optional[OpenAI], model: str = "gpt-5"):
        self.client = openai_client
        self.model = model

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        dot = sum(x*y for x, y in zip(a, b))
        na = math.sqrt(sum(x*x for x in a)) or 1e-9
        nb = math.sqrt(sum(y*y for y in b)) or 1e-9
        return max(min(dot / (na * nb), 1.0), -1.0)

    @staticmethod
    def _safe_num(s: str, default: float = 0.0) -> float:
        try:
            return float((s or "").strip())
        except Exception:
            return default

    def evaluate(self, question: str, context: str, answer: str) -> Dict[str, Any]:
        if self.client is None:
            return {"error": "openai client not available"}

        out: Dict[str, Any] = {}

        # Faithfulness
        try:
            f_prompt = (
                "Split the answer into minimal factual statements and check each against the context.\n"
                "Return ONLY one numeric value in [0,1] = fraction supported by context.\n\n"
                f"Context:\n{context}\n\nAnswer:\n{answer}"
            )
            f_resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Return a single number [0,1]."},
                    {"role": "user", "content": f_prompt},
                ],
            )
            val = self._safe_num(f_resp.choices[0].message.content, 0.0)
            out["faithfulness"] = max(0.0, min(1.0, val))
        except Exception as e:
            out["faithfulness_error"] = f"{e}"

        # Answer Relevance
        try:
            emb_model = os.getenv("RAGAS_EMBED_MODEL", "text-embedding-3-small")
            q_emb = self.client.embeddings.create(input=question or "", model=emb_model).data[0].embedding
            a_emb = self.client.embeddings.create(input=answer or "", model=emb_model).data[0].embedding
            cos = self._cosine(q_emb, a_emb)
            out["answer_relevance"] = (cos + 1.0) / 2.0
        except Exception as e:
            out["answer_relevance_error"] = f"{e}"

        # Context Relevance
        try:
            cr_prompt = (
                "Estimate how focused and necessary the context is to answer the question.\n"
                "Return ONLY one numeric value in [0,1].\n\n"
                f"Question:\n{question}\n\nContext:\n{context}"
            )
            cr_resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Return a single number [0,1]."},
                    {"role": "user", "content": cr_prompt},
                ],
            )
            val = self._safe_num(cr_resp.choices[0].message.content, 0.0)
            out["context_relevance"] = max(0.0, min(1.0, val))
        except Exception as e:
            out["context_relevance_error"] = f"{e}"

        return out


@dataclass
class StepRunResult:
    exit_code: int
    duration_sec: float
    log_path: str
    found_artifacts: Dict[str, List[str]]
    title: Optional[str] = None
    analysis_prompt: Optional[str] = None
    rag_request: Optional[str] = None
    gpt_flags: Optional[Dict[str, Any]] = None
    gpt_summary: Optional[str] = None
    ragas_scores: Optional[Dict[str, Any]] = None
    agent_time_sec: Optional[float] = None


class Runner:
    """Runs shell commands of StepSpec and collects local artifacts.

    - Instead of reading stdout/stderr line by line,
      subprocess.run(..., stdout=logf, stderr=STDOUT) writes
      directly to the log file only, reducing I/O overhead.
    - If timeout_sec is None, 0, or negative, no timeout is applied.
    """

    @staticmethod
    def run_and_stream(spec: StepSpec) -> StepRunResult:
        t0 = time.time()
        os.makedirs(os.path.dirname(spec.log_path) or ".", exist_ok=True)

        with open(spec.log_path, "w", encoding="utf-8") as logf:
            # Write header
            logf.write(f"--- RUN START: {spec.title} ---\n")
            logf.write(f"CMD: {spec.command}\n\n")
            logf.flush()

            # Handle timeout_sec: 0 or negative means "no timeout"
            raw_timeout = getattr(spec, "timeout_sec", None)
            timeout_val: Optional[float]
            if isinstance(raw_timeout, (int, float)) and raw_timeout > 0:
                timeout_val = float(raw_timeout)
            else:
                timeout_val = None  # timeout disabled

            try:
                proc = subprocess.run(
                    spec.command,
                    shell=True,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout_val,
                )
                exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                # On timeout, log a message and mark exit_code as negative
                msg = f"[MONITOR] Timeout reached ({raw_timeout}s), process killed.\n"
                print(msg, end="")
                logf.write(msg)
                exit_code = -9

            # Write footer
            logf.write(f"\n--- RUN END ({spec.title}) exit_code={exit_code} ---\n")

        # Search for artifacts
        found_artifacts: Dict[str, List[str]] = {}
        for patt in getattr(spec, "expected_artifacts", []):
            found_artifacts[patt] = sorted(glob.glob(patt, recursive=True))

        return StepRunResult(
            exit_code=exit_code or 0,
            duration_sec=time.time() - t0,
            log_path=spec.log_path,
            found_artifacts=found_artifacts,
            title=spec.title,
            analysis_prompt=getattr(spec, "analysis_prompt", None),
            rag_request=getattr(spec, "rag_request", None),
        )



# ====================================================================================
# LLM Top-K helpers (NEW)
# ====================================================================================
def _sanitize_filename(s: str) -> str:
    s = re.sub(r"[^\w\s\-\.\(\)\[\]가-힣]+", "_", s)
    return re.sub(r"\s+", "_", s).strip("_")[:120] or "noname"

def _unique_tokens(seq):
    seen = set()
    out = []
    for x in seq:
        t = (x or "").strip()
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out

def _merge_rag_request(existing: str, topk_obj: Dict[str, Any], max_len: int = 512) -> str:
    """
    Merge existing + (terms + phrases + query_hint), then dedup/trim.
    """
    existing = (existing or "").strip()
    terms = topk_obj.get("terms") or []
    phrases = topk_obj.get("phrases") or []
    hint = (topk_obj.get("query_hint") or "").strip()

    bag = []
    bag.extend(terms if isinstance(terms, list) else [])
    bag.extend(phrases if isinstance(phrases, list) else [])
    if hint:
        bag.append(hint)

    bag = _unique_tokens(bag)
    merged = (existing + " " + " ".join(bag)).strip() if existing else " ".join(bag)
    return merged[:max_len]

def llm_extract_topk(openai_client: Optional[OpenAI], model: str, text: str, k: int = 10) -> Dict[str, Any]:
    """
    Use the LLM to extract the top-K terms/phrases for RAG retrieval from log/summary text.
    On failure, perform a simple statistical fallback extraction.
    """
    out = {"terms": [], "phrases": [], "query_hint": ""}

    if openai_client is None:
        # fallback: very simple keyword extraction
        toks = re.findall(r"[A-Za-z가-힣0-9_\-\.%]+", text.lower())
        counts: Dict[str, int] = {}
        stop = {"the","and","for","with","acc","loss","epoch","step","task","logs","avg","mean","last","after","before"}
        for t in toks:
            if len(t) < 3 or t in stop:
                continue
            counts[t] = counts.get(t, 0) + 1
        top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:k]
        terms = [t for (t, _) in top]
        out["terms"] = terms
        out["query_hint"] = " ".join(terms[:max(3, min(6, k))])
        return out

    prompt = f"""
You are helping build a retrieval query for detecting data-poisoning patterns in continual learning logs.
From the TEXT below, extract the {k} most salient search terms and short multi-word phrases (2-4 words) that would be most helpful for literature/code retrieval.
Rules:
- Keep each term/phrase <= 4 words.
- Return STRICT JSON with keys: terms (list[str]), phrases (list[str]), query_hint (string).

TEXT:
\"\"\"{text[:15000]}\"\"\"
""".strip()

    try:
        resp = openai_client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Return only valid JSON with keys: terms, phrases, query_hint."},
                {"role": "user", "content": prompt},
            ],
        )
        data = json.loads(resp.choices[0].message.content)
        data["terms"] = _unique_tokens([str(x) for x in (data.get("terms") or [])])[:k]
        data["phrases"] = _unique_tokens([str(x) for x in (data.get("phrases") or [])])[:k]
        data["query_hint"] = (data.get("query_hint") or "").strip()
        return data
    except Exception:
        # fallback: simple extraction
        toks = re.findall(r"[A-Za-z가-힣0-9_\-\.%]+", text.lower())
        counts: Dict[str, int] = {}
        for t in toks:
            if len(t) < 3:
                continue
            counts[t] = counts.get(t, 0) + 1
        top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:k]
        terms = [t for (t, _) in top]
        return {"terms": terms, "phrases": [], "query_hint": " ".join(terms[:max(3, min(6, k))])}

def _save_llm_topk_json(log_dir: str, session_id: str, step_idx: int, step_title: str, obj: Dict[str, Any]) -> str:
    topk_dir = os.path.join(log_dir, "LLM_topk")
    os.makedirs(topk_dir, exist_ok=True)
    fname = f"LLM_topk_{session_id}_step{step_idx}_{_sanitize_filename(step_title)}.json"
    fpath = os.path.join(topk_dir, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return fpath


# ====================================================================================
# Analyzer (analysis-only path using memory + RAG + RAGAS)
# ====================================================================================
class Analyzer:
    def __init__(self, log_dir: str, model: str = "gpt-5"):
        self.log_dir = log_dir
        self.model = model

        # OpenAI client (optional)
        self.client = None
        if _OPENAI_AVAILABLE:
            _key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")
            if _key:
                try:
                    self.client = OpenAI(api_key=_key)
                except Exception:
                    self.client = None

        self.memory = MemoryManager(log_dir)
        self.summarizer = Summarizer(self.memory, model=self.model, temperature=1.0)
        self.ragas = RAGASEvaluator(self.client, model=self.model)
        self.rag = RAGRetriever(self.client, model=self.model)

    def _read_log_tail(self, path: str, tail_chars: int) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()[-tail_chars:]

    def analyze(self, specs: List[StepSpec], session_id: Optional[str] = None) -> Dict[str, Any]:
        session_id = session_id or f"analysis_{time.strftime('%Y%m%d_%H%M%S')}"
        _ = _ensure_dir(os.path.join(self.log_dir, "memory"))

        USE_RAG = os.getenv("USE_RAG", "1") == "1"
        USE_TOPK = os.getenv("USE_TOPK", "1") == "1"
        TOPK_K = int(os.getenv("TOPK_K", "10"))
        TRUNC = int(os.getenv("LOG_TAIL", "20000"))

        items: List[Dict[str, Any]] = []
        for spec in specs:
            t0 = time.time()

            # 1) Read log
            try:
                log_text = self._read_log_tail(spec.log_path, TRUNC)
            except Exception as e:
                items.append({
                    "title": spec.title,
                    "log_path": spec.log_path,
                    "error": f"log read error: {e}",
                    "rag": None,
                    "summary": None,
                })
                continue

            # 2) Previous summaries
            prev_sums = self.memory.get_previous_summaries(session_id, max_items=5)

            # 3) RAG (optional)
            rag_answer = ""
            rag_obj = {"ok": False, "answer": "", "sources": [], "error": "disabled"}
            if USE_RAG and self.client is not None:
                rag_obj = self.rag.analyze(
                    rag_query=spec.rag_request or "",
                    log_text=log_text,
                    chroma_collection_name="papers",
                    chroma_host="127.0.0.1",
                    chroma_port=8000,
                    top_k=1,
                )
                rag_answer = (rag_obj or {}).get("answer") or ""

            # 3.5) Top-K text for current summarize input
            topk_text = ""
            if USE_TOPK:
                topk_seed = f"{rag_answer}\n\n{log_text}"
                topk_obj = llm_extract_topk(self.client, self.model, topk_seed, k=TOPK_K)
                topk_text = _format_topk_text(topk_obj)

            # 4) Summarize with memory + rag + log
            g = self.summarizer.summarize(
                session_id=session_id,
                step_title=spec.title,
                rag_text=rag_answer,
                prev_summaries=prev_sums,
                log_text=log_text,
                topk_text=topk_text,
            )
            if isinstance(g, dict):
                g = _ensure_rag_grounding(g, rag_obj, rag_answer)
            summary_text = _extract_summary_text(g)

            # 5) Save current summary into memory
            self.memory.save_summary(session_id, spec.title, summary_text)

            # 6) RAGAS scoring
            try:
                question = (spec.rag_request or spec.analysis_prompt or f"What happened in step: {spec.title}?")[:1000]
                context = f"rag_answer: {rag_answer}\n\nlog_text: {log_text}"
                answer = summary_text[:4000]
                ragas_scores = self.ragas.evaluate(question, context, answer)
            except Exception as e:
                ragas_scores = {"error": f"ragas_scoring_failed: {e}"}

            # 7) Collect
            items.append({
                "title": spec.title,
                "log_path": spec.log_path,
                "rag": rag_obj,
                "summary": {"text": summary_text, "error": None},
                "ragas_scores": ragas_scores,
                "elapsed_sec": round(time.time() - t0, 2),
            })

        # Aggregate RAGAS avg
        def _avg(xs: List[float]) -> Optional[float]:
            return round(sum(xs) / len(xs), 3) if xs else None

        f_vals, ar_vals, cr_vals = [], [], []
        for it in items:
            rs = it.get("ragas_scores") or {}
            f = rs.get("faithfulness"); ar = rs.get("answer_relevance"); cr = rs.get("context_relevance")
            if isinstance(f, (int, float)): f_vals.append(float(f))
            if isinstance(ar, (int, float)): ar_vals.append(float(ar))
            if isinstance(cr, (int, float)): cr_vals.append(float(cr))

        return {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": self.model,
            "session_id": session_id,
            "ragas_avg": {
                "faithfulness": _avg(f_vals),
                "answer_relevance": _avg(ar_vals),
                "context_relevance": _avg(cr_vals),
            } if any([f_vals, ar_vals, cr_vals]) else None,
            "items": items,
            "hints": {"USE_RAG": USE_RAG, "LOG_TAIL": TRUNC},
        }


# ====================================================================================
# Legacy GPT verify (optional; used only in monitor mode for raw log summary)
# ====================================================================================
def gpt_verify_step(step_log_path: str, step_title: str, model: str = "gpt-5") -> Dict[str, Any]:
    if not _OPENAI_AVAILABLE:
        return {"error": "openai library not available"}
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")
    if not api_key:
        return {"error": "OPENAI_API_KEY not set"}
    client = OpenAI(api_key=api_key)

    try:
        with open(step_log_path, "r", encoding="utf-8", errors="ignore") as f:
            log_text = f.read()[-200000:]
    except Exception as e:
        return {
            "step_executed": False,
            "errors": [f"read error: {e}"],
            "warnings": [],
            "evidence": [],
            "summary": f"Failed to read log file ({step_log_path}): {e}",
        }

    prompt = f"""
Below is the log of the '{step_title}' step. Please respond only in the following JSON format (with exactly 5 keys).
{{
  "step_executed": true/false,
  "errors": ["..."],
  "warnings": ["..."],
  "evidence": ["3-5 representative lines from the log"],
  "summary": "A short, human-readable summary"
}}
log:
-----
{log_text}
-----
""".strip()

    try:
        resp = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Return valid JSON with exactly the keys: step_executed, errors, warnings, evidence, summary."},
                {"role": "user", "content": prompt},
            ],
        )
        data = json.loads(resp.choices[0].message.content)
    except Exception as e:
        last_lines = [ln for ln in log_text.strip().splitlines()[-10:]]
        summary_guess = ("Log tail:\n" + "\n".join(last_lines[:5])) if last_lines else "Cannot summarize (failed to parse model response)"
        data = {
            "step_executed": None,
            "errors": [f"gpt_parse_error: {e}"],
            "warnings": [],
            "evidence": last_lines[:3] if last_lines else [],
            "summary": summary_guess,
        }

    for k, default in [("step_executed", None), ("errors", []), ("warnings", []), ("evidence", []), ("summary", "No summary")]:
        data.setdefault(k, default)
    return data


# ====================================================================================
# Monitor (exec + analysis + LLM Top-K injection)
# ====================================================================================
def _step_local_verdict(s: StepRunResult) -> bool:
    art_ok = True if not s.found_artifacts else any(len(v) > 0 for v in s.found_artifacts.values())
    return (s.exit_code == 0) and art_ok


def build_report(steps: List[StepRunResult], gpt_model: str = "gpt-5") -> Dict[str, Any]:
    report_steps: List[Dict[str, Any]] = []
    all_pass = True
    f_vals, ar_vals, cr_vals = [], [], []

    for s in steps:
        local_pass = _step_local_verdict(s)
        all_pass = all_pass and local_pass
        entry: Dict[str, Any] = {
            "title": s.title,
            "local_pass": local_pass,
            "exit_code": s.exit_code,
            "duration_sec": round(s.duration_sec, 1),
            "log_path": s.log_path,
            "found_artifacts": {k: len(v) for k, v in s.found_artifacts.items()},
            "gpt": s.gpt_flags or {},
        }
        if s.agent_time_sec is not None:
            entry["agent_time_sec"] = round(s.agent_time_sec, 1)
        
        if s.ragas_scores:
            entry["ragas_scores"] = s.ragas_scores
            f = s.ragas_scores.get("faithfulness"); ar = s.ragas_scores.get("answer_relevance"); cr = s.ragas_scores.get("context_relevance")
            if isinstance(f, (int, float)): f_vals.append(float(f))
            if isinstance(ar, (int, float)): ar_vals.append(float(ar))
            if isinstance(cr, (int, float)): cr_vals.append(float(cr))
        report_steps.append(entry)

    def _avg(xs: List[float]) -> Optional[float]:
        return round(sum(xs) / len(xs), 3) if xs else None

    ragas_avg = {
        "faithfulness": _avg(f_vals),
        "answer_relevance": _avg(ar_vals),
        "context_relevance": _avg(cr_vals),
    } if any([f_vals, ar_vals, cr_vals]) else None

    return {"overall_verdict": "PASS" if all_pass else "CHECK", "ragas_avg": ragas_avg, "steps": report_steps}


def monitor_pipeline(
    specs: List[StepSpec],
    use_gpt: bool = True,
    gpt_model: str = "gpt-5",
    pipeline_tag: str = "brainwash",  # file name tagging
    use_langchain_analysis: bool = True,  # use the same path as Analysis
    fallback_gpt_verify: bool = False,    # legacy verification if needed
) -> List[StepRunResult]:
    """
    Run each step ➜ summarize/evaluate (Analysis style) ➜ (NEW: MMD PNG vision analysis) ➜
    extract/save LLM Top-K ➜ auto-inject Top-K into the next step's rag_request ➜ save per-step report
    """
    out: List[StepRunResult] = []

    # Shared context
    session_id = f"monitor_{pipeline_tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    analyzer = Analyzer(LOG_DIR, model=gpt_model)
    TRUNC = int(os.getenv("LOG_TAIL", "20000"))
    USE_RAG = os.getenv("USE_RAG", "1") == "1"
    USE_TOPK = os.getenv("USE_TOPK", "1") == "1"
    TOPK_K = int(os.getenv("TOPK_K", "10"))
    USE_VISION = os.getenv("USE_VISION", "1") == "1"  

    print(f"[CFG] USE_RAG={USE_RAG}  USE_TOPK={USE_TOPK}  TOPK_K={TOPK_K}  USE_VISION={USE_VISION}")
    print(f"[CFG] session_id={session_id}")

    for idx, spec in enumerate(specs, start=1):
        print("\n" + "=" * 80)
        print(f"[RUN] {spec.title}")
        res = Runner.run_and_stream(spec)
        print("-" * 80)

        # ---- removed summarize_local call → replaced with inline output ----
        art_ok = True if not res.found_artifacts else any(len(v) > 0 for v in res.found_artifacts.values())
        verdict = "PASS" if (res.exit_code == 0 and art_ok) else "CHECK"
        artifacts_count = {k: len(v) for k, v in res.found_artifacts.items()}
        print(f"[{verdict}] exit={res.exit_code} time={res.duration_sec:.1f}s log={res.log_path}")
        print(f"  artifacts: {artifacts_count}")

        agent_t0 = time.time()

        # ----- Analysis-style summary (default path)
        if use_langchain_analysis:
            # 1) Read log tail
            try:
                with open(spec.log_path, "r", encoding="utf-8", errors="ignore") as f:
                    log_text = f.read()[-TRUNC:]
            except Exception as e:
                log_text = f"[log read error: {e}]"

            # 2) Previous mem summaries
            prev_sums = analyzer.memory.get_previous_summaries(session_id, max_items=5)

            # 3) RAG
            rag_answer = ""
            rag_obj = {"ok": False, "answer": "", "sources": [], "error": "disabled"}
            if USE_RAG and analyzer.client is not None:
                rag_obj = analyzer.rag.analyze(
                    rag_query=spec.rag_request or "",
                    log_text=log_text,
                    chroma_collection_name="papers",
                    chroma_host="127.0.0.1",
                    chroma_port=8000,
                    top_k=1,
                )
                rag_answer = (rag_obj or {}).get("answer") or ""

            # 3.5) Top-K text for current summarize input
            topk_text = ""
            if USE_TOPK:
                topk_seed = f"{rag_answer}\n\n{log_text}"
                topk_seed_obj = llm_extract_topk(analyzer.client, gpt_model, topk_seed, k=TOPK_K)
                topk_text = _format_topk_text(topk_seed_obj)

            # 4) Summarize (LangChain + memory)
            g = analyzer.summarizer.summarize(
                session_id=session_id,
                step_title=spec.title,
                rag_text=rag_answer,
                prev_summaries=prev_sums,
                log_text=log_text,
                topk_text=topk_text
            )
            if isinstance(g, dict):
                g = _ensure_rag_grounding(g, rag_obj, rag_answer)
            
            if isinstance(g, dict):
                print(pretty_print_attack_result(g))
            
            summary_text = _extract_summary_text(g)
            analyzer.memory.save_summary(session_id, spec.title, summary_text)

            # ------------------------------------------------------------------
            #  (NEW) If this is an MMD Visualization step, analyze the PNG once more with GPT Vision
            # ------------------------------------------------------------------
            try:
                is_mmd_pipeline = pipeline_tag.lower() in ("mmd_backdoor", "mmd", "mmd_backdoor_cifar100")
                is_visual_step = ("visual" in (spec.title or "").lower()) or ("viz" in (spec.title or "").lower())

                if USE_VISION and is_mmd_pipeline and is_visual_step and analyzer.client is not None:
                    # Prefer the fixed PNG path
                    fixed_png = (
                        "/path/to/backdoor/Multi-Level-MMD-Regularization/"
                        "figures/cifar10_vgg11_blended_mlmmdr_0.1_all.png"
                    )
                    if os.path.exists(fixed_png):
                        png_path = fixed_png
                    else:
                        png_path = _find_latest_png(
                            "/path/to/backdoor/Multi-Level-MMD-Regularization/figures/*.png"
                        )

                    if png_path:
                        vision_obj = gpt_analyze_png(analyzer.client, gpt_model, png_path)

                        # Save
                        vis_dir = os.path.join(LOG_DIR, "vision")
                        os.makedirs(vis_dir, exist_ok=True)
                        vis_path = os.path.join(vis_dir, f"vision_{session_id}_step{idx}.json")
                        with open(vis_path, "w", encoding="utf-8") as f:
                            json.dump(vision_obj, f, ensure_ascii=False, indent=2)
                        print(f"[VISION] saved: {vis_path}")

                        # Merge into summary to use as decision evidence
                        summary_text = (
                            summary_text
                            + "\n\n[VISION_IMAGE_ANALYSIS]\n"
                            + json.dumps(vision_obj, ensure_ascii=False, indent=2)
                        )

                        # Also save to memory (optional)
                        analyzer.memory.save_summary(session_id, spec.title + " (vision)", json.dumps(vision_obj, ensure_ascii=False))
                    else:
                        print("[VISION] png not found, skipped")
            except Exception as e:
                print(f"[VISION] error: {e}")

            # 5) RAGAS  (score using summary_text merged with vision)
            try:
                question = (spec.rag_request or spec.analysis_prompt or f"What happened in step: {spec.title}?")[:1000]
                context = f"rag_answer: {rag_answer}\n\nlog_text: {log_text}"
                answer = summary_text[:4000]
                ragas_scores = analyzer.ragas.evaluate(question, context, answer)
            except Exception as e:
                ragas_scores = {"error": f"ragas_scoring_failed: {e}"}

            # 6) Attach to result  (store summary_text with vision reflected)
            res.gpt_flags = g if isinstance(g, dict) else {"raw": str(g)}
            res.gpt_summary = summary_text
            res.ragas_scores = ragas_scores

            # 7) 🔹 Extract LLM Top-K + save + inject into next step's rag_request
            if USE_TOPK:
                combined_text = f"{summary_text}\n\n{log_text}"
                topk_obj = llm_extract_topk(analyzer.client, gpt_model, combined_text, k=TOPK_K)
                saved_path = _save_llm_topk_json(LOG_DIR, session_id, idx, spec.title, topk_obj)
                print(f"[LLM TOP-K] saved: {saved_path}")

                # If there is a next step, update its rag_request
                if idx < len(specs):
                    next_spec = specs[idx]  # next item at 0-based index
                    base = next_spec.rag_request or ""
                    next_spec.rag_request = _merge_rag_request(base, topk_obj, max_len=512)

                    injected_note = {
                        "session_id": session_id,
                        "applied_to_next_step": next_spec.title,
                        "base_query_before": base,
                        "merged_query_after": next_spec.rag_request,
                        "from_step": spec.title,
                        "topk_used": topk_obj,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    note_path = os.path.join(
                        LOG_DIR, "LLM_topk",
                        f"LLM_topk_injected_{session_id}_step{idx}_to_step{idx+1}.json"
                    )
                    with open(note_path, "w", encoding="utf-8") as f:
                        json.dump(injected_note, f, ensure_ascii=False, indent=2)
                    print(f"[LLM TOP-K] injected into next rag_request: {note_path}")

        # ----- (Optional) Fallback legacy GPT verify
        elif fallback_gpt_verify:
            g = gpt_verify_step(spec.log_path, spec.title, model=gpt_model)
            res.gpt_flags = g if isinstance(g, dict) else {"raw": str(g)}
            res.gpt_summary = (g or {}).get("summary")
            if res.gpt_summary:
                try:
                    with open(spec.log_path, "r", encoding="utf-8", errors="ignore") as f:
                        _log = f.read()[-2000:]
                    question = (spec.rag_request or spec.analysis_prompt or f"Summarize step: {spec.title}")[:1000]
                    context = _log
                    answer = res.gpt_summary[:4000]
                    res.ragas_scores = analyzer.ragas.evaluate(question, context, answer)
                except Exception as e:
                    res.ragas_scores = {"error": f"ragas_scoring_failed: {e}"}

        #  End of agent analysis timing
        agent_elapsed = time.time() - agent_t0
        res.agent_time_sec = agent_elapsed
        print(f"[AGENT] time={agent_elapsed:.1f}s")

        #  Step-level Agent/ChromaDB/paper report save
        step_report = build_report([res], gpt_model=gpt_model)

        if isinstance(res.gpt_flags, dict):
            step_report["pretty"] = pretty_print_attack_result(res.gpt_flags)

        step_path = os.path.join(LOG_DIR, f"monitor_summary_{pipeline_tag}_step{idx}.json")
        with open(step_path, "w", encoding="utf-8") as f:
            json.dump(step_report, f, ensure_ascii=False, indent=2)
        print(f"[STEP REPORT] saved: {step_path}")

        out.append(res)

    return out



def _b64_data_url_from_png(png_path: str) -> str:
    with open(png_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def _find_latest_png(pattern: str) -> Optional[str]:
    paths = sorted(glob.glob(pattern), key=lambda p: os.path.getmtime(p))
    return paths[-1] if paths else None

def gpt_analyze_png(openai_client: Optional[OpenAI], model: str, png_path: str) -> Dict[str, Any]:
    """
    Feed the PNG as vision input and receive a backdoor/clean/uncertain verdict as JSON.
    """
    if openai_client is None:
        return {"ok": False, "error": "OpenAI client unavailable", "png_path": png_path}

    img_url = _b64_data_url_from_png(png_path)
    prompt = (
        "You are an ML security analyst.\n"
        "Explain what the plot indicates and conclude whether it suggests backdoor behavior.\n"
        "Please write it in the form of a report. "
        "Your task is to infer which exact attack occurred (if any) from model logs, RAG evidence, and prior summaries. "
        "You must choose decisively among known attack types such as Brainwash, Accumulative Attack, Label Flipping, MMDRegularization or No Attack (clean scenario). "
        "Avoid vague language like 'similar to' or 'resembling' — be specific and conclusive. "
        "Please look at the figure and add an explanation describing what it shows."
        "Return STRICT JSON with keys:\n"
        "  verdict: one of ['backdoor','clean','uncertain']\n"
        "  evidence: list[str]\n"
        "  explanation: str\n"
    )

    resp = openai_client.responses.create(
        model=model,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": img_url},
            ],
        }],
        # Optionally add options like temperature=0 if needed
    )

    # If the SDK provides output_text use it, otherwise handle raw as a string
    text = getattr(resp, "output_text", None) or ""
    # Assume text is a JSON string and parse it
    try:
        obj = json.loads(text)
        return {"ok": True, "png_path": png_path, "vision": obj}
    except Exception:
        return {"ok": True, "png_path": png_path, "vision_raw": text}

def pretty_print_attack_result(obj: Dict[str, Any]) -> str:
    attack = obj.get("attack_identified", "UNKNOWN")
    risk = obj.get("risk_level", "?")
    evidence = obj.get("evidence_for_attack", "")
    rationale = obj.get("rationale_for_decision", "")
    rag = obj.get("supporting_evidence_from_rag", "")

    # Split evidence
    num_ev, topk_ev = "", ""
    if "[TOPK]" in evidence:
        num_ev, topk_ev = evidence.split("[TOPK]", 1)
    else:
        num_ev = evidence

    return (
        "================ ATTACK ANALYSIS ================\n"
        f"Attack Identified : {attack}\n"
        f"Risk Level        : {risk}\n\n"
        "---- NUMERIC EVIDENCE ----\n"
        f"{num_ev.strip()}\n\n"
        "---- TOP-K CONTEXT ----\n"
        f"{topk_ev.strip() if topk_ev else '(none)'}\n\n"
        "---- RATIONALE ----\n"
        f"{rationale}\n\n"
        "---- RAG SUPPORT ----\n"
        f"{rag}\n"
        "=================================================\n"
    )

# ====================================================================================
# CLI
# ====================================================================================
if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    choice = input("Which program to run? (Brainwash / brainwash_miniimagenet/ brainwash_cifar10 / accumulative_cifar100/ brainwash_tinyimagenet / Accumulative / Test / Analysis / Analysis_Accumulative  / MMD_backdoor / MMD_backdoor_cifar100 / Detect / Rethink/ Rethink_pub) [Brainwash]: ").strip() or "Brainwash"

    if choice.lower() in ("analysis", "analyze", "a"):
        specs = build_analyze_brainwash_specs()
        session_id = f"analysis_{time.strftime('%Y%m%d_%H%M%S')}"
        analyzer = Analyzer(LOG_DIR, model="gpt-5")
        report = analyzer.analyze(specs, session_id=session_id)

        out_path = os.path.join(LOG_DIR, "analysis.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 40)
        print("=== ANALYSIS REPORT (JSON) ===")
        print("=" * 40)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"\nAnalysis saved to: {out_path}")

    elif choice.lower() in ("analysis_accumulative", "analyze_accumulative", "aa"):
        specs = build_analyze_accumulative_specs()
        session_id = f"analysis_accu_{time.strftime('%Y%m%d_%H%M%S')}"
        analyzer = Analyzer(LOG_DIR, model="gpt-5")
        report = analyzer.analyze(specs, session_id=session_id)

        out_path = os.path.join(LOG_DIR, "analysis_accumulative.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 40)
        print("=== ACCUMULATIVE ANALYSIS REPORT (JSON) ===")
        print("=" * 40)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"\nAnalysis saved to: {out_path}")

    else:
        # Legacy execution path + Top-K injection
        build = {
            "brainwash": build_brainwash_specs,
            "bw": build_brainwash_specs,
            "brainwash_miniimagenet": build_brainwash_miniimagenet_specs,
            "brainwash_tinyimagenet": build_brainwash_tinyimagenet_specs,
            "brainwash_cifar10": build_brainwash_cifar10_specs, 
            "accumulative": build_accumulative_specs,
            "accumulative_cifar100": build_accumulative_cifar100_specs, 
            "accu": build_accumulative_specs,
            "acc": build_accumulative_specs,
            "mmd_backdoor": build_mmd_backdoor_specs,
            "mmd_backdoor_cifar100": build_mmd_backdoor_cifar100_specs,
            "detect": build_detect_specs,  
            "rethink": build_rethink_specs,
            "rethink_pub": build_rethink_pubmed_specs,
        }.get(choice.lower(), build_brainwash_specs)

        specs = build()
        results = monitor_pipeline(
            specs,
            use_gpt=True,
            gpt_model="gpt-5",
            pipeline_tag=choice.lower(),
            use_langchain_analysis=True,
            fallback_gpt_verify=False,
        )
        report = build_report(results, gpt_model="gpt-5")

        print("\n" + "=" * 40)
        print("=== EXECUTION & MONITOR REPORT (JSON) ===")
        print("=" * 40)
        print(json.dumps(report, ensure_ascii=False, indent=2))

        out_path = os.path.join(LOG_DIR, f"monitor_summary_{choice.lower()}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nReport saved to: {out_path}")
