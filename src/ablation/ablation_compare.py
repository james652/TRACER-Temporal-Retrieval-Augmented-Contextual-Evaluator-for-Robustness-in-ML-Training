#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical ablation comparison: full vs no_rag / no_topk / no_memory."""
from __future__ import annotations
import os, re, glob, json
from statistics import mean

VARIANTS = {  # <-- point these to your ablation result directories
    "full":      "./results/full",
    "no_rag":    "./results/no_rag",
    "no_topk":   "./results/no_topk",
    "no_memory": "./results/no_memory",
}
GT = {
    "Brainwash/cifar10":"Brainwash","Brainwash/cifar100":"Brainwash",
    "Brainwash/mini":"Brainwash","Brainwash/tiny":"Brainwash",
    "Accumulative/cifar10":"Accumulative Attack","Accumulative/cifar100":"Accumulative Attack",
    "Detect":"Label Flipping","Rethink":"Label Flipping",
}
SCEN_ORDER = ["Brainwash/cifar10","Brainwash/cifar100","Brainwash/mini","Brainwash/tiny",
              "Accumulative/cifar10","Accumulative/cifar100","Detect","Rethink"]

def scenario_of(path):
    p = path.lower()
    def ds():
        if "cifar_100" in p or "cifar100" in p: return "cifar100"
        if "cifar_10" in p or "cifar10" in p: return "cifar10"
        if "mini" in p: return "mini"
        if "tiny" in p: return "tiny"
        return "?"
    if "brainwash" in p and "/brainwash" in p: return f"Brainwash/{ds()}"
    if "accumulative" in p: return f"Accumulative/{ds()}"
    if "/detect" in p: return "Detect"
    if "/rethink" in p: return "Rethink"
    if "mmd" in p: return f"MMD/{ds()}"
    return "UNKNOWN"

def run_id(path):
    m = re.search(r"/(logs[^/]*)/", path); return m.group(1) if m else "?"

def load_steps(fp):
    try: d = json.load(open(fp))
    except Exception: return None
    out=[]
    for s in d.get("steps",[]):
        g=s.get("gpt",{}) or {}; rs=s.get("ragas_scores",{}) or {}
        out.append({"attack":g.get("attack_identified"),"risk":g.get("risk_level"),
                    "F":rs.get("faithfulness"),"AR":rs.get("answer_relevance"),"CR":rs.get("context_relevance")})
    return out

# preference for canonical full run (exclude *newagent* = older generation)
def pick_canonical(runs):
    # runs: list of (run_id, fp, steps)
    primary = [r for r in runs if "newagent" not in r[0].lower()]
    pool = primary or runs
    pref = {"logs":0,"logs_1":1,"logs_2":2,"logs_newAgent":3}
    pool = sorted(pool, key=lambda r: pref.get(r[0], 9))
    return pool[0]

# gather
data={}
for v,root in VARIANTS.items():
    for fp in glob.glob(os.path.join(root,"**","monitor_summary_*.json"),recursive=True):
        if re.search(r"_step\d+\.json$",fp): continue
        scen=scenario_of(fp); steps=load_steps(fp)
        if not steps or scen not in GT: continue
        data.setdefault(scen,{}).setdefault(v,[]).append((run_id(fp),fp,steps))

def f(x): return f"{x:.2f}" if isinstance(x,(int,float)) else " - "

# build table
print("="*118)
print(f"{'SCENARIO':<22}{'GT':<14}| " + "".join(f"{v:<24}" for v in VARIANTS))
print(f"{'':<22}{'':<14}| " + "".join(f"{'final  ok meanF':<24}" for v in VARIANTS))
print("="*118)

counts={v:{"ok":0,"tot":0,"fn":[],"fp":[],"midwrong":[]} for v in VARIANTS}
ragas_all={v:{"F":[],"AR":[],"CR":[]} for v in VARIANTS}

for scen in SCEN_ORDER:
    if scen not in data: continue
    gt=GT[scen]; row=f"{scen:<22}{gt[:13]:<14}| "
    for v in VARIANTS:
        if v not in data[scen]:
            row+=f"{'(missing)':<24}"; continue
        rid,fp,steps=pick_canonical(data[scen][v])
        final=steps[-1]["attack"]; ok=(final==gt)
        favg=[s["F"] for s in steps if isinstance(s["F"],(int,float))]
        mF=mean(favg) if favg else None
        counts[v]["tot"]+=1; counts[v]["ok"]+=ok
        for s in steps:
            for k in ("F","AR","CR"):
                if isinstance(s[k],(int,float)): ragas_all[v][k].append(s[k])
        # false negative: attack scenario but final says No Attack or wrong
        if not ok:
            if final=="No Attack": counts[v]["fn"].append((scen,rid,fp))
            else: counts[v]["fp"].append((scen,rid,final,fp))
        # intermediate misclassification (any non-final step gives a wrong attack label, not No Attack)
        mids=[s["attack"] for s in steps[:-1]]
        if any(m not in (gt,"No Attack",None) for m in mids):
            counts[v]["midwrong"].append((scen,rid,[s["attack"] for s in steps]))
        tag = "OK " if ok else ("FN " if final=="No Attack" else "FP ")
        row+=f"{final[:12]:<13}{tag:<4}{f(mF):<7}"
    print(row)

print("\n"+"="*70)
print("SUMMARY per variant")
print("="*70)
for v in VARIANTS:
    c=counts[v]; r=ragas_all[v]
    print(f"\n[{v}]  final-correct {c['ok']}/{c['tot']}"
          f"   meanF={mean(r['F']):.2f}  meanAR={mean(r['AR']):.2f}  meanCR={mean(r['CR']):.2f}"
          f"   (zeros in F: {sum(1 for x in r['F'] if x==0)})")
    if c['fn']:
        print("   FALSE NEGATIVES (final=No Attack):")
        for scen,rid,fp in c['fn']: print(f"     - {scen} [{rid}]")
    if c['fp']:
        print("   WRONG-TYPE (final=other attack):")
        for scen,rid,final,fp in c['fp']: print(f"     - {scen} [{rid}] -> {final}")
    if c['midwrong']:
        print("   INTERMEDIATE misclassification (final still ok):")
        for scen,rid,traj in c['midwrong']: print(f"     - {scen} [{rid}]: {' -> '.join(str(x) for x in traj)}")
