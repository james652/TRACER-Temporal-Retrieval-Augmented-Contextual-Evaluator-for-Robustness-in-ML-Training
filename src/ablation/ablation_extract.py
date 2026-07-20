#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract per-step diagnoses across the 4 ablation result trees for comparison."""
from __future__ import annotations
import os, re, glob, json

VARIANTS = {
    "full":      "/home/jun/work/soongsil/Agent/A_result",
    "no_memory": "/home/jun/work/soongsil/Agent/A_result_no_memory",
    "no_topk":   "/home/jun/work/soongsil/Agent/A_result_no_topk",
    "no_rag":    "/home/jun/work/soongsil/Agent/A_result_norag",
}

# ground truth attack per scenario key
GT = {
    "Brainwash/cifar10": "Brainwash", "Brainwash/cifar100": "Brainwash",
    "Brainwash/mini": "Brainwash", "Brainwash/tiny": "Brainwash",
    "Accumulative/cifar10": "Accumulative Attack", "Accumulative/cifar100": "Accumulative Attack",
    "Detect": "Label Flipping",     # Label Error env -> agent labels Label Flipping (per paper)
    "Rethink": "Label Flipping",    # meta-label poisoning on GNN
    "MMD/cifar10": "MMDRegularization", "MMD/cifar100": "MMDRegularization",
}

def scenario_of(path):
    p = path.lower()
    def ds():
        if "cifar_100" in p or "cifar100" in p: return "cifar100"   # check 100 BEFORE 10 (substring)
        if "cifar_10" in p or "cifar10" in p: return "cifar10"
        if "mini" in p: return "mini"
        if "tiny" in p: return "tiny"
        return "?"
    if "brainwash" in p and "/brainwash" in p:
        return f"Brainwash/{ds()}"
    if "accumulative" in p:
        return f"Accumulative/{ds()}"
    if "/detect" in p: return "Detect"
    if "/rethink" in p: return "Rethink"
    if "mmd" in p: return f"MMD/{ds()}"
    return "UNKNOWN:" + path

def run_id(path):
    m = re.search(r"/(logs[^/]*)/", path)
    return m.group(1) if m else "?"

def load_steps(fp):
    try:
        d = json.load(open(fp))
    except Exception as e:
        return None
    out = []
    for s in d.get("steps", []):
        g = s.get("gpt", {}) or {}
        rs = s.get("ragas_scores", {}) or {}
        out.append({
            "title": (s.get("title") or "")[:34],
            "attack": g.get("attack_identified"),
            "risk": g.get("risk_level"),
            "F": rs.get("faithfulness"),
            "AR": rs.get("answer_relevance"),
            "CR": rs.get("context_relevance"),
        })
    return out

# collect: scenario -> variant -> list of (run_id, fp, steps)
data = {}
for vname, root in VARIANTS.items():
    for fp in glob.glob(os.path.join(root, "**", "monitor_summary_*.json"), recursive=True):
        if re.search(r"_step\d+\.json$", fp):
            continue
        scen = scenario_of(fp)
        steps = load_steps(fp)
        if not steps:
            continue
        data.setdefault(scen, {}).setdefault(vname, []).append((run_id(fp), fp, steps))

def fmt(x):
    if x is None: return " - "
    if isinstance(x, float): return f"{x:.2f}"
    return str(x)

order_scen = ["Brainwash/cifar10","Brainwash/cifar100","Brainwash/mini","Brainwash/tiny",
              "Accumulative/cifar10","Accumulative/cifar100","Detect","Rethink",
              "MMD/cifar10","MMD/cifar100"]
order_var = ["full","no_rag","no_topk","no_memory"]

for scen in order_scen + [s for s in data if s not in order_scen]:
    if scen not in data: continue
    gt = GT.get(scen, "?")
    print("\n" + "="*100)
    print(f"SCENARIO: {scen}    (ground truth = {gt})")
    print("="*100)
    for v in order_var + [x for x in data[scen] if x not in order_var]:
        if v not in data[scen]: continue
        for rid, fp, steps in sorted(data[scen][v]):
            final = steps[-1]["attack"] if steps else None
            detected = [st["attack"] for st in steps]
            hit_any = gt in detected
            final_ok = (final == gt)
            traj = " -> ".join(f"{fmt(st['attack'])}(r{fmt(st['risk'])})" for st in steps)
            favg = [st["F"] for st in steps if isinstance(st["F"],(int,float))]
            print(f"\n [{v:9}|{rid:14}] final={fmt(final):20} final_ok={final_ok}  detected_any={hit_any}")
            print(f"   traj: {traj}")
            print(f"   F per step: {[fmt(st['F']) for st in steps]}  (mean {fmt(sum(favg)/len(favg) if favg else None)})")
