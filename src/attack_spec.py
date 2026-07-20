# attack_specs.py
# ----------------
# Standalone module that defines:
# - Constants: LOG_DIR, BASE_DIR
# - Data model: StepSpec
# - Pipeline builders: build_brainwash_specs, build_accumulative_specs
#
# This file is meant to be imported by your agent/runner (e.g., attack_agent.py).

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

import os
import glob
import json

# ------------------------------------------------------------------------------------
# Centralized paths
# ------------------------------------------------------------------------------------
LOG_DIR = "./logs"  # <-- set this to your desired log/output directory
BASE_DIR = "."      # <-- set this to the base directory that holds your scripts/checkpoints
_TOPK_DIR = os.path.join(LOG_DIR, "LLM_topk")

# ------------------------------------------------------------------------------------
# HOW TO USE THIS FILE
# ------------------------------------------------------------------------------------
# The build_*_specs() functions below each return a list of StepSpec objects. Every
# StepSpec describes one step of an attack/analysis pipeline, and its `command` field is
# the exact shell command that is executed for that step.
#
# All file-system paths shown inside these commands (and in expected_artifacts / vision_globs)
# are PLACEHOLDERS written as "/path/to/...". They are NOT real paths. Before running,
# replace every "/path/to/..." with the real path to your own attack/training scripts,
# model checkpoints, datasets, and output directories. You may also replace an entire
# command with your own equivalent invocation.
#
# Flags, dataset names, epoch counts, hyper-parameters, and CUDA_VISIBLE_DEVICES values
# are kept intact so each command still documents how the corresponding step is meant to
# be run. Adjust them as needed for your environment.
# ------------------------------------------------------------------------------------


@dataclass
class StepSpec:
    title: str
    command: str
    log_path: str
    #forbidden_regex: List[str] = field(default_factory=list)
    expected_artifacts: List[str] = field(default_factory=list)  # glob patterns
    timeout_sec: int = 0  # 0 = no timeout
    analysis_log_path: Optional[str] = None
    analysis_prompt: Optional[str] = None
    rag_request: Optional[str] = None
    vision_globs: List[str] = field(default_factory=list)  # glob patterns of generated images to feed to GPT Vision


# ------------------------------------------------------------------------------------
# Helpers to load Top-k terms saved by Analyzer/Runner
# ------------------------------------------------------------------------------------
def _read_json_safe(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _pick_latest(paths: List[str]) -> Optional[str]:
    if not paths:
        return None
    try:
        return max(paths, key=lambda p: os.path.getmtime(p))
    except Exception:
        # if mtime fails, just pick lexicographically last
        return sorted(paths)[-1]


def _collect_terms_from_file(obj: dict, limit: int = 20) -> str:
    # Accepted schema:
    # {
    #   "attack_terms": ["...", ...],
    #   "non_attack_terms": ["...", ...],
    #   "evidence_phrases": ["...", ...]  # optional
    # }
    atk = obj.get("attack_terms") or []
    non = obj.get("non_attack_terms") or []
    evd = obj.get("evidence_phrases") or []

    # flatten to short, comma-joined strings
    def _take(xs): return ", ".join([str(x) for x in xs[:limit] if isinstance(x, (str, int, float))])

    parts = []
    if atk:
        parts.append(f"topk_attack_terms: {_take(atk)}")
    if non:
        parts.append(f"topk_non_attack_terms: {_take(non)}")
    if evd:
        parts.append(f"topk_evidence: {_take(evd)}")
    return " | ".join(parts)


def _load_topk_for_step(step_key: str, limit: int = 20) -> str:
    """
    step_key examples: 'step1', 'step2', 'step3', 'step4'
    filename example: analysis_20251107_153012__step1.json
    """
    if not os.path.isdir(_TOPK_DIR):
        return ""

    # First, look only for files matching this step
    PAT = os.path.join(_TOPK_DIR, f"*__{step_key}.json")
    cand = glob.glob(PAT)
    chosen = _pick_latest(cand)
    if chosen:
        obj = _read_json_safe(chosen)
        if isinstance(obj, dict):
            return _collect_terms_from_file(obj, limit=limit)

    # Otherwise, fall back to the most recent file overall (graceful fallback)
    all_json = glob.glob(os.path.join(_TOPK_DIR, "*.json"))
    chosen = _pick_latest(all_json)
    if chosen:
        obj = _read_json_safe(chosen)
        if isinstance(obj, dict):
            return _collect_terms_from_file(obj, limit=limit)

    return ""


def _concat_rag(base: str, addon: str) -> str:
    base = (base or "").strip()
    addon = (addon or "").strip()
    if not addon:
        return base
    if base:
        return f"{base} || {addon}"
    return addon


# ------------------------------------------------------------------------------------
# Spec builders
# ------------------------------------------------------------------------------------
def build_brainwash_specs() -> List[StepSpec]:
    base = BASE_DIR

    # Common base query
    base_query = (
        "brainwash continual learning poisoning ewc split_cifar100 cifar100 "
        "fisher diagonal catastrophic forgetting average accuracy BWT"
    )

    # For each stage, load the previous step's LLM_topk and inject it into rag_request
    # step1: initial training, so there is no prior topk
    step1_topk = ""  # no prior step
    step2_topk = _load_topk_for_step("step1")
    step3_topk = _load_topk_for_step("step2")
    step4_topk = _load_topk_for_step("step3")

    return [
        StepSpec(
            title="Stage 1/4: Initial Model Training (EWC)",
            command=(
                "CUDA_VISIBLE_DEVICES=1 python /path/to/Brainwash/main_baselines.py "
                "--experiment split_cifar100 --approach ewc --lasttask 9 --tasknum 10 --nepochs 20 "
                "--batch-size 16 --lamb 500000 --lamb_emp 100 --clip 100. --lr 0.01"
            ),
            log_path=f"{LOG_DIR}/step1.log",
            expected_artifacts=[f"{base}/**/*.pkl"],
            analysis_prompt=(
                "Analyze the initial EWC training stage. "
                "Summarize training metrics, "
                "detect any irregularities (NaNs, divergence, or instability), "
                "and describe what these suggest about the model's baseline behavior. "
                f"Please analyze the following log file.: {LOG_DIR}/step1.log"
            ),
            rag_request=_concat_rag(base_query, step1_topk),
        ),
        StepSpec(
            title="Stage 2/4: Execution",
            command=(
                "python /path/to/Brainwash/main_inv.py "
                "--pretrained_model_add=/path/to/Brainwash/checkpoint/data_cifar100/ewc_lamb_500000.0__model_type_resnet_dataset_split_cifar100_class_num_10_bs_16_lr_0.01_n_epochs_20__model_name_ResNet_task_num_9__seed_0_emb_fact_1_im_sz_32_.pkl --save_dir=/path/to/Brainwash/inverse_file/cifar_100 "
                "--task_lst=0,1,2,3,4,5,6,7,8 --save_every=1000 --batch_reg --init_acc --n_iters=10000"
            ),
            log_path=f"{LOG_DIR}/step2.log",
            expected_artifacts=[f"{base}/continual-learning-baselines/test/Brainwash/model_inv/**/*.npz"],
            analysis_prompt=(
                "Summarize training metrics, "
                "detect any irregularities (NaNs, divergence, or instability), "
                "and describe what these suggest about the model's baseline behavior. "
                f"Please analyze the following log file.: {LOG_DIR}/step2.log"
            ),
            rag_request=_concat_rag(base_query, step2_topk),
        ),
        StepSpec(
            title="Stage 3/4: Application",
            command=(
                "CUDA_VISIBLE_DEVICES=0 python /path/to/Brainwash/main_brainwash.py "
                "--extra_desc=reckless_test "
                "--pretrained_model_add=/path/to/Brainwash/checkpoint/data_cifar100/ewc_lamb_500000.0__model_type_resnet_dataset_split_cifar100_class_num_10_bs_16_lr_0.01_n_epochs_20__model_name_ResNet_task_num_9__seed_0_emb_fact_1_im_sz_32_.pkl "
                "--mode='reckless' --target_task_for_eval=0 --delta=0.3 --seed=0 "
                "--eval_every=10 "
                "--distill_folder=/path/to/Brainwash/inverse_file/cifar_100 "
                "--init_acc --noise_norm=inf --cont_learner_lr=0.001 --n_epochs=101 --save_every=100 "
            ),
            log_path=f"{LOG_DIR}/step3.log",
            expected_artifacts=[f"{base}/**/*.pkl"],
            analysis_prompt=(
                "Explain performance changes, instability, or potential evidence of model collapse. "
                "Summarize training metrics, "
                "detect any irregularities (NaNs, divergence, or instability), "
                "and describe what these suggest about the model's baseline behavior. "
                f"Please analyze the following log file.: {LOG_DIR}/step3.log"
            ),
            rag_request=_concat_rag(base_query, step3_topk),
        ),
        StepSpec(
            title="Stage 4/4: Final Evaluation",
            command=(
                "CUDA_VISIBLE_DEVICES=0 python /path/to/Brainwash/main_baselines.py "
                "--experiment split_cifar100 --approach ewc --lasttask 9 --tasknum 10 --nepochs 20 "
                "--batch-size 16 --lr 0.01 --clip 100. --lamb 500000 --lamb_emp 100 "
                "--checkpoint /path/to/noise_ewc_reckless_test__delta_0.3_dataset_split_cifar100_target_task_0_attacked_task_9_noise_optim_lr_0.005__n_iters_1_n_epochs_101_seed_0_mode_reckless____min_acc_target_36.pkl "
                "--init_acc --addnoise"
            ),
            log_path=f"{LOG_DIR}/step4.log",
            expected_artifacts=[f"{base}/**/acc_mat_*.npy"],
            analysis_prompt=(
                "Based on the analyses from steps 1–3, identify the most probable attack type. "
                "Choose exactly one from the following: Brainwash, Accumulative Attack, or Label Flipping. "
                "Do not hedge with expressions like 'similar to' or 'Brainwash-like' — be decisive. "
                "Provide supporting numerical evidence (accuracy, BWT, or collapse pattern) from the logs. "
                f"Please analyze the following log file. {LOG_DIR}/step4.log"
            ),
            rag_request=_concat_rag(base_query, step4_topk),
        ),
    ]

# attack_spec.py

def build_accumulative_specs() -> List[StepSpec]:
    base_query = "accumulative attack online training PGD Linf epsilon trigger collapse"

    # Stage 1: base training
    step1_cmd = (
        "python /path/to/PoisoningAttack/AccumulativeAttack/train_cifar.py "
        "--outf /path/to/PoisoningAttack/AccumulativeAttack/checkpoints_base_bn"
    )

    # Stage 2: online accumulative attack training
    step2_cmd = (
        "python /path/to/PoisoningAttack/AccumulativeAttack/online_accu_train.py "
        "--batch_size 100 --epoch 100 --test_batch_size 500 --log_name log_test_online.txt "
        "--resume /path/to/PoisoningAttack/AccumulativeAttack/checkpoints_base_bn "
        "--use_bn --model_name cifar10_epoch40.pth "
        "--mode 'eval' --onlinemode 'train' --lr 1e-1 --momentum 0.9 "
        "--beta 1. --only_reg --threshold 0.18 --use_advtrigger"
    )

    # Top-K to inject into the second step (extracted from the first step's results)
    step2_topk = _load_topk_for_step("step1")

    return [
        # 1) Run base training
        StepSpec(
            title="Accumulative 1/2: Base Train",
            command=step1_cmd,
            log_path=f"{LOG_DIR}/accumulative_base_train.log",
            expected_artifacts=[f"{BASE_DIR}/**/*.pth", f"{BASE_DIR}/**/*.pt", f"{BASE_DIR}/**/*.pkl"],
            analysis_prompt=(
                "Summarize online training: report accuracy/loss trends per epoch, "
                "note when threshold is breached, and describe any progressive degradation."
            ),
            rag_request=_concat_rag(base_query, ""),  # step1 has no external injection
        ),

        # 2) Run online accumulative attack training
        StepSpec(
            title="Accumulative 2/2: Online Poisoning Train",
            command=step2_cmd,
            log_path=f"{LOG_DIR}/accumulative_train.log",
            analysis_prompt=(
                "Based on the analyses from step 1, determine what type of attack (if any) occurred. "
                "If it matches a known method, specify which paper's technique it resembles. "
                "If it is not an attack, explain why. "
                "Finally, concisely summarize the observed behavior and key metrics. "
                "Choose exactly one: Brainwash, Accumulative Attack, Label Flipping, or No Attack."
            ),
            # <- step1 Top-K is auto-injected here
            rag_request=_concat_rag(base_query + " final evaluation analysis", step2_topk),
        ),
    ]

def build_accumulative_cifar100_specs() -> List[StepSpec]:
    base_query = "accumulative attack online training PGD Linf epsilon trigger collapse cifar100"

    # Stage 1: CIFAR-100 base training (clean)
    step1_cmd = (
        "python /path/to/PoisoningAttack/AccumulativeAttack/train_cifar.py "
        "--dataset cifar100 --dataroot ./data --epochs 120"
    )

    # Stage 2: CIFAR-100 accumulative online training + trigger
    step2_cmd = (
        "python /path/to/PoisoningAttack/AccumulativeAttack/online_accu_train.py "
        "--batch_size 100 --epoch 100 --test_batch_size 500 --log_name log_test_online.txt "
        "--resume /path/to/PoisoningAttack/AccumulativeAttack/checkpoints_base_bn_cifar100 "
        "--use_bn --model_name cifar100_epoch120.pth "
        "--mode 'eval' --onlinemode 'train' --lr 1e-1 --momentum 0.9 --poison_scale 0.3 "
        "--beta 1. --only_reg --threshold 0.46 --use_online_advtrigger "
        "--num_classes 100 --dataset cifar100 --dataroot ./data"
    )

    step1_topk = ""  # no Top-K at the start
    step2_topk = _load_topk_for_step("cifar100_step1")

    return [
        # ---------------------- Stage 1: Base Train ----------------------
        StepSpec(
            title="Accumulative CIFAR100 1/2: Base Train (clean)",
            command=step1_cmd,
            log_path=f"{LOG_DIR}/accu_cifar100_base_train.log",
            expected_artifacts=[
                f"{BASE_DIR}/**/*.pth",
                f"{BASE_DIR}/**/*.pt",
                f"{BASE_DIR}/**/*.pkl",
            ],
            analysis_prompt=(
                "Summarize online training: report accuracy/loss trends per epoch, "
                "note when threshold is breached, and describe any progressive degradation."
            ),
            rag_request=_concat_rag(base_query, step1_topk),
        ),

        # ---------------------- Stage 2: Online Accumulative ----------------------
        StepSpec(
            title="Accumulative CIFAR100 2/2: Online Poisoning Train",
            command=step2_cmd,
            log_path=f"{LOG_DIR}/accumulative_cifar100_train.log",
            expected_artifacts=[
                f"{BASE_DIR}/**/*.pth",
                f"{BASE_DIR}/**/*.pt",
                f"{BASE_DIR}/**/*.pkl",
            ],
            analysis_prompt=(
                "Based on the analyses from step 1, determine what type of attack (if any) occurred. "
                "If it matches a known method, specify which paper's technique it resembles. "
                "If it is not an attack, explain why. "
                "Finally, concisely summarize the observed behavior and key metrics. "
                "Choose exactly one: Brainwash, Accumulative Attack, Label Flipping, or No Attack."
            ),
            rag_request=_concat_rag(base_query + " cifar100 final evaluation analysis", step2_topk),
        ),
    ]



def build_analyze_brainwash_specs() -> List[StepSpec]:
    base_query = "average accuracy"
    # Analysis-only: attaching Top-k is not strictly required here, but it can be injected the same way if desired
    return [
        StepSpec(
            title="Analysis: Initial Model Training (EWC) Log",
            command="",
            log_path=f"{LOG_DIR}/step1.log",
            analysis_prompt=(
                "Analyze the initial EWC training stage. "
                "Summarize training metrics (loss, accuracy, BWT, etc.), "
                "detect any irregularities (NaNs, divergence, or instability), "
                "and describe what these suggest about the model's baseline behavior."
            ),
            rag_request=_concat_rag(base_query + " explain EWC training stability and baseline performance",
                                    _load_topk_for_step("step1")),
        ),
        StepSpec(
            title="Analysis: Inversion Attack Log",
            command="",
            log_path=f"{LOG_DIR}/step2.log",
            analysis_prompt=(
                "Summarize progress, identify any unusual patterns or warnings, "
                "and explain how the attack might influence the model's internal representations."
            ),
            rag_request=_concat_rag(base_query + " inversion attack effect on representations",
                                    _load_topk_for_step("step2")),
        ),
        StepSpec(
            title="Analysis: Brainwash Reckless Mode Execution Log",
            command="",
            log_path=f"{LOG_DIR}/step3.log",
            analysis_prompt=(
                "Summarize performance changes, instability, or potential evidence of model collapse. "
            ),
            rag_request=_concat_rag(base_query + " brainwash reckless mode effect analysis",
                                    _load_topk_for_step("step3")),
        ),
        StepSpec(
            title="Analysis: Final Evaluation Log",
            command="",
            log_path=f"{LOG_DIR}/step4.log",
            analysis_prompt=(
                "Based on the analyses from steps 1–3, identify the most probable attack type. "
                "Choose exactly one from the following: Brainwash, Accumulative Attack, or Label Flipping. "
                "Do not hedge with expressions like 'similar to' or 'Brainwash-like' — be decisive. "
                "Provide supporting numerical evidence (accuracy, BWT, or collapse pattern) from the logs."
            ),
            rag_request=_concat_rag(base_query + " catastrophic forgetting", _load_topk_for_step("step3")),
        ),
    ]


# brainwash pipeline for mini-Imagenet
def build_brainwash_miniimagenet_specs() -> List[StepSpec]:
    base = BASE_DIR

    base_query = "average accuracy"

    step1_topk = ""
    step2_topk = _load_topk_for_step("mini_step1")
    step3_topk = _load_topk_for_step("mini_step2")
    step4_topk = _load_topk_for_step("mini_step3")

    return [
        # ---------------------- Stage 1: Train ----------------------
        StepSpec(
            title="MiniImageNet steps 1/4",
            command=(
                "CUDA_VISIBLE_DEVICES=1 python /path/to/Brainwash/main_baselines.py "
                "--experiment split_mini_imagenet --approach ewc --lasttask 9 --tasknum 10 --nepochs 20 "
                "--batch-size 16 --lamb 500000 --lamb_emp 100 --clip 100. --lr 0.01"
            ),
            log_path=f"{LOG_DIR}/mini_step1.log",
            expected_artifacts=[f"{base}/**/*.pkl"],
            analysis_prompt=(
                "Analyze the initial EWC training stage on split_mini_imagenet. "
                "Summarize training metrics, detect any irregularities (NaNs, divergence, or instability), "
                "and describe what these suggest about the model's baseline behavior. "
                f"Please analyze the following log file.: {LOG_DIR}/mini_step1.log"
            ),
            rag_request=_concat_rag(base_query, step1_topk),
        ),

        # ---------------------- Stage 2: inversion ----------------------
        StepSpec(
            title="MiniImageNet steps 2/4",
            command=(
                "CUDA_VISIBLE_DEVICES=1 python /path/to/Brainwash/main_inv.py "
                "--pretrained_model_add=/path/to/Brainwash/checkpoint/data_miniImagenet/"
                "ewc_lamb_500000.0__model_type_resnet_dataset_split_mini_imagenet_class_num_10_bs_16_lr_0.01_n_epochs_20__"
                "model_name_ResNet_task_num_9__seed_0_emb_fact_1_im_sz_84_.pkl "
                "--num_samples=128 "
                "--save_dir=/path/to/Brainwash/output_mini_Imagenet_inversion "
                "--task_lst=0,1,2,3,4,5,6,7,8 "
                "--save_every=1000 --batch_reg --init_acc --n_iters=10000"
            ),
            log_path=f"{LOG_DIR}/mini_step2.log",
            expected_artifacts=[f"{base}/Brainwash/output_mini_Imagenet_inversion/**/*.npz"],
            analysis_prompt=(
                "Summarize progress during inversion on split_mini_imagenet, identify any unusual patterns or warnings, "
                "and explain how the attack might influence the model's internal representations. "
                f"Please analyze the following log file.: {LOG_DIR}/mini_step2.log"
            ),
            rag_request=_concat_rag(base_query, step2_topk),
        ),

        # ---------------------- Stage 3: Brainwash reckless mode ----------------------
        StepSpec(
            title="MiniImageNet steps 3/4",
            command=(
                "CUDA_VISIBLE_DEVICES=1 python /path/to/Brainwash/main_brainwash.py "
                "--extra_desc=reckless_test "
                "--pretrained_model_add=/path/to/Brainwash/checkpoint/data_miniImagenet/"
                "ewc_lamb_500000.0__model_type_resnet_dataset_split_mini_imagenet_class_num_10_bs_16_lr_0.01_n_epochs_20__"
                "model_name_ResNet_task_num_9__seed_0_emb_fact_1_im_sz_84_.pkl "
                "--mode='reckless' --target_task_for_eval=0 --delta=0.3 --seed=0 --eval_every=10 "
                "--distill_folder=/path/to/Brainwash/output_mini_Imagenet_inversion "
                "--init_acc --noise_norm=inf --cont_learner_lr=0.001 --n_epochs=51 --save_every=50"
            ),
            log_path=f"{LOG_DIR}/mini_step3.log",
            expected_artifacts=[f"{base}/**/*.pkl"],
            analysis_prompt=(
                "Explain performance changes, instability, or potential evidence of model collapse "
                "when running Brainwash reckless mode on split_mini_imagenet. "
                "Summarize training metrics, detect any irregularities (NaNs, divergence, or instability), "
                "and describe what these suggest about the model's behavior. "
                f"Please analyze the following log file.: {LOG_DIR}/mini_step3.log"
            ),
            rag_request=_concat_rag(base_query, step3_topk),
        ),

        # ---------------------- Stage 4: Final Evaluation ----------------------
        StepSpec(
            title="MiniImageNet steps 4/4",
            command=(
                "CUDA_VISIBLE_DEVICES=1 python /path/to/Brainwash/main_baselines.py "
                "--experiment split_mini_imagenet --approach ewc --lasttask 9 --tasknum 10 --nepochs 20 "
                "--batch-size 16 --lr 0.01 --clip 100. --lamb 500000 --lamb_emp 100 "
                "--checkpoint /path/to/Brainwash/checkpoint_noise/mini_imagenet/"
                "noise_ewc_reckless_test__delta_0.3_dataset_split_mini_imagenet_target_task_0_attacked_task_9_"
                "noise_optim_lr_0.005__n_iters_1_n_epochs_51_seed_0_mode_reckless____min_acc_target_46.pkl "
                "--init_acc --addnoise"
            ),
            log_path=f"{LOG_DIR}/mini_step4.log",
            expected_artifacts=[f"{base}/**/acc_mat_*.npy"],
            analysis_prompt=(
                "Based on the analyses from steps 1–3, identify the most probable attack type. "
                "Choose exactly one from the following: Brainwash, Accumulative Attack, or Label Flipping. "
                "Do not hedge with expressions like 'similar to' or 'Brainwash-like' — be decisive. "
                "Provide supporting numerical evidence (accuracy, BWT, or collapse pattern) from the logs."
                f"Please analyze the following log file.: {LOG_DIR}/mini_step4.log"
            ),
            rag_request=_concat_rag(base_query, step4_topk),
        ),
    ]

# brainwash pipeline for tiny-Imagenet
def build_brainwash_tinyimagenet_specs() -> List[StepSpec]:
    base = BASE_DIR

    # Common base query (add more later if needed)
    base_query = "average accuracy tiny_imagenet"

    # Top-K works the same way as mini; only the step key is separated as tiny_*
    step1_topk = ""  # initial training stage has no prior step
    step2_topk = _load_topk_for_step("tiny_step1")
    step3_topk = _load_topk_for_step("tiny_step2")
    step4_topk = _load_topk_for_step("tiny_step3")

    return [
        # ---------------------- Stage 1: Train ----------------------
        StepSpec(
            title="TinyImageNet steps 1/4",
            command=(
                "CUDA_VISIBLE_DEVICES=2 python /path/to/Brainwash/main_baselines.py "
                "--experiment split_tiny_imagenet --approach ewc --lasttask 9 --tasknum 10 --nepochs 20 "
                "--batch-size 16 --lamb 500000 --lamb_emp 100 --clip 100. --lr 0.01"
            ),
            log_path=f"{LOG_DIR}/tiny_step1.log",
            expected_artifacts=[f"{base}/**/*.pkl"],
            analysis_prompt=(
                "Analyze the initial EWC training stage on split_tiny_imagenet. "
                "Summarize training metrics, detect any irregularities (NaNs, divergence, or instability), "
                "and describe what these suggest about the model's baseline behavior. "
                f"Please analyze the following log file.: {LOG_DIR}/tiny_step1.log"
            ),
            rag_request=_concat_rag(base_query, step1_topk),
        ),

        # ---------------------- Stage 2: inversion ----------------------
        StepSpec(
            title="TinyImageNet steps 2/4",
            command=(
                "CUDA_VISIBLE_DEVICES=2 python /path/to/Brainwash/main_inv.py "
                "--pretrained_model_add=/path/to/Brainwash/checkpoint/data_tinyImagenet/"
                "ewc_lamb_500000.0__model_type_resnet_dataset_split_tiny_imagenet_class_num_20_bs_16_lr_0.01_n_epochs_20__"
                "model_name_ResNet_task_num_9__seed_0_emb_fact_9_im_sz_64_.pkl "
                "--num_samples=128 "
                "--save_dir=/path/to/Brainwash/output_tiny_Imagenet_inversion "
                "--task_lst=0,1,2,3,4,5,6,7,8 --save_every=1000 --batch_reg --init_acc --n_iters=10000"
            ),
            log_path=f"{LOG_DIR}/tiny_step2.log",
            expected_artifacts=[f"{base}/Brainwash/output_tiny_Imagenet_inversion/**/*.npz"],
            analysis_prompt=(
                "Summarize progress during inversion on split_tiny_imagenet, "
                "identify any unusual patterns or warnings, "
                "and explain how the attack might influence the model's internal representations. "
                f"Please analyze the following log file.: {LOG_DIR}/tiny_step2.log"
            ),
            rag_request=_concat_rag(base_query, step2_topk),
        ),

        # ---------------------- Stage 3: Brainwash reckless mode ----------------------
        StepSpec(
            title="TinyImageNet steps 3/4",
            command=(
                "CUDA_VISIBLE_DEVICES=2 python /path/to/Brainwash/main_brainwash.py "
                "--extra_desc=reckless_test "
                "--pretrained_model_add=/path/to/Brainwash/checkpoint/data_tinyImagenet/"
                "ewc_lamb_500000.0__model_type_resnet_dataset_split_tiny_imagenet_class_num_20_bs_16_lr_0.01_n_epochs_20__"
                "model_name_ResNet_task_num_9__seed_0_emb_fact_9_im_sz_64_.pkl "
                "--mode='reckless' --target_task_for_eval=0 --delta=0.3 --seed=0 --eval_every=10 "
                "--distill_folder=/path/to/Brainwash/output_tiny_Imagenet_inversion "
                "--init_acc --noise_norm=inf --cont_learner_lr=0.001 --n_epochs=51 --save_every=50"
            ),
            log_path=f"{LOG_DIR}/tiny_step3.log",
            expected_artifacts=[f"{base}/**/*.pkl"],
            analysis_prompt=(
                "Explain performance changes, instability, or potential evidence of model collapse "
                "when running Brainwash reckless mode on split_tiny_imagenet. "
                "Summarize training metrics, detect any irregularities (NaNs, divergence, or instability), "
                "and describe what these suggest about the model's behavior. "
                f"Please analyze the following log file.: {LOG_DIR}/tiny_step3.log"
            ),
            rag_request=_concat_rag(base_query, step3_topk),
        ),

        # ---------------------- Stage 4: Final Evaluation ----------------------
        StepSpec(
            title="TinyImageNet steps 4/4",
            command=(
                "CUDA_VISIBLE_DEVICES=2 python /path/to/Brainwash/main_baselines.py "
                "--experiment split_tiny_imagenet --approach ewc --lasttask 9 --tasknum 10 --nepochs 20 --batch-size 16 "
                "--lr 0.01 --clip 100. --lamb 500000 --lamb_emp 100 "
                "--checkpoint=/path/to/Brainwash/checkpoint_noise/tiny_imagenet/"
                "noise_ewc_reckless_test__delta_0.3_dataset_split_tiny_imagenet_target_task_0_attacked_task_9_"
                "noise_optim_lr_0.005__n_iters_1_n_epochs_51_seed_0_mode_reckless____min_acc_target_5.pkl "
                "--init_acc --addnoise"
            ),
            log_path=f"{LOG_DIR}/tiny_step4.log",
            expected_artifacts=[f"{base}/**/acc_mat_*.npy"],
            analysis_prompt=(
                "Based on the analyses from steps 1–3, identify the most probable attack type. "
                "Choose exactly one from the following: Brainwash, Accumulative Attack, or Label Flipping. "
                "Do not hedge with expressions like 'similar to' or 'Brainwash-like' — be decisive. "
                "Provide supporting numerical evidence (accuracy, BWT, or collapse pattern) from the logs."
                f"Please analyze the following log file.: {LOG_DIR}/tiny_step4.log"
            ),
            rag_request=_concat_rag(base_query, step4_topk),
        ),
    ]


def build_brainwash_cifar10_specs() -> List[StepSpec]:
    base = BASE_DIR

    # Common base query (for cifar10)
    base_query = (
        "brainwash continual learning poisoning ewc split_cifar10 cifar10 "
        "fisher diagonal catastrophic forgetting average accuracy BWT"
    )

    # Keep only the Top-K mechanism identical (may be empty at the start)
    step1_topk = ""  # initial training stage has no prior step
    step2_topk = _load_topk_for_step("c10_step1")
    step3_topk = _load_topk_for_step("c10_step2")
    step4_topk = _load_topk_for_step("c10_step3")

    return [
        # ---------------------- Stage 1: Train ----------------------
        StepSpec(
            title="CIFAR10 Stage 1/4: Initial Model Training (EWC)",
            command=(
                "CUDA_VISIBLE_DEVICES=1 python /path/to/Brainwash/main_baselines.py "
                "--experiment split_cifar10 --approach ewc --lasttask 4 --tasknum 5 --nepochs 20 "
                "--batch-size 16 --lamb 500000 --lamb_emp 100 --clip 100. --lr 0.01"
            ),
            log_path=f"{LOG_DIR}/c10_step1.log",
            expected_artifacts=[f"{base}/**/*.pkl"],
            analysis_prompt=(
                "Analyze the initial EWC training stage on split_cifar10. "
                "Summarize training metrics, detect any irregularities (NaNs, divergence, or instability), "
                "and describe what these suggest about the model's baseline behavior. "
                f"Please analyze the following log file.: {LOG_DIR}/c10_step1.log"
            ),
            rag_request=_concat_rag(base_query, step1_topk),
        ),

        # ---------------------- Stage 2: inversion ----------------------
        StepSpec(
            title="CIFAR10 Stage 2/4: Inversion Attack",
            command=(
                "CUDA_VISIBLE_DEVICES=1 python /path/to/Brainwash/main_inv.py "
                "--pretrained_model_add=/path/to/Brainwash/checkpoint/data_cifar10/"
                "ewc_lamb_500000.0__model_type_resnet_dataset_split_cifar10_class_num_2_bs_16_lr_0.01_n_epochs_20__"
                "model_name_ResNet_task_num_4__seed_0_emb_fact_1_im_sz_32_.pkl "
                "--num_samples=512 "
                "--save_dir=/path/to/Brainwash/output_cifar10_inversion "
                "--task_lst=0,1,2,3 --save_every=1000 --batch_reg --init_acc "
                "--n_iters=40000"
            ),
            log_path=f"{LOG_DIR}/c10_step2.log",
            expected_artifacts=[
                f"{base}/Brainwash/output_cifar10_inversion/**/*.npz"
            ],
            analysis_prompt=(
                "Summarize progress during inversion on split_cifar10, identify any unusual patterns or warnings, "
                "and explain how the attack might influence the model's internal representations. "
                f"Please analyze the following log file.: {LOG_DIR}/c10_step2.log"
            ),
            rag_request=_concat_rag(base_query, step2_topk),
        ),

        # ---------------------- Stage 3: Brainwash reckless mode ----------------------
        StepSpec(
            title="CIFAR10 Stage 3/4: Apply Brainwash Reckless Mode",
            command=(
                "CUDA_VISIBLE_DEVICES=0 python /path/to/Brainwash/main_brainwash.py "
                "--extra_desc=reckless_test "
                "--pretrained_model_add=/path/to/Brainwash/checkpoint/data_cifar10/"
                "ewc_lamb_500000.0__model_type_resnet_dataset_split_cifar10_class_num_2_bs_16_lr_0.01_n_epochs_20__"
                "model_name_ResNet_task_num_4__seed_0_emb_fact_1_im_sz_32_.pkl "
                "--mode='reckless' --target_task_for_eval=0 --delta=0.7 --seed=0 --eval_every=10 "
                "--distill_folder=/path/to/Brainwash/inverse_file/cifar_10 "
                "--init_acc --noise_norm=inf --cont_learner_lr=0.002 "
                "--n_epochs=301 --save_every=300"
            ),
            log_path=f"{LOG_DIR}/c10_step3.log",
            expected_artifacts=[f"{base}/**/*.pkl"],
            analysis_prompt=(
                "Explain performance changes, instability, or potential evidence of model collapse "
                "when running Brainwash reckless mode on split_cifar10. "
                "Summarize training metrics, detect any irregularities (NaNs, divergence, or instability), "
                "and describe what these suggest about the model's behavior. "
                f"Please analyze the following log file.: {LOG_DIR}/c10_step3.log"
            ),
            rag_request=_concat_rag(base_query, step3_topk),
        ),

        # ---------------------- Stage 4: Final Evaluation ----------------------
        StepSpec(
            title="CIFAR10 Stage 4/4: Final Evaluation",
            command=(
                "CUDA_VISIBLE_DEVICES=0 python /path/to/Brainwash/main_baselines.py "
                "--experiment split_cifar10 --approach ewc --lasttask 4 --tasknum 5 --nepochs 80 "
                "--batch-size 16 --lr 0.01 --clip 100. --lamb 50000 --lamb_emp 100 "
                "--checkpoint /path/to/noise_ewc_reckless_test__delta_0.7_dataset_split_cifar10_target_task_0_"
                "attacked_task_4_noise_optim_lr_0.005__n_iters_1_n_epochs_301_seed_0_mode_reckless____min_acc_target_"
                "70.pkl "
                "--init_acc --addnoise"
            ),
            log_path=f"{LOG_DIR}/c10_step4.log",
            expected_artifacts=[f"{base}/**/acc_mat_*.npy"],
            analysis_prompt=(
                "Based on the analyses from steps 1–3 on split_cifar10, identify the most probable attack type. "
                "Choose exactly one from the following: Brainwash, Accumulative Attack, or Label Flipping. "
                "Do not hedge with expressions like 'similar to' or 'Brainwash-like' — be decisive. "
                "Provide supporting numerical evidence (accuracy, BWT, or collapse pattern) from the logs. "
                f"Please analyze the following log file.: {LOG_DIR}/c10_step4.log"
            ),
            rag_request=_concat_rag(base_query, step4_topk),
        ),
    ]

def build_analyze_accumulative_specs() -> List[StepSpec]:
    """
    Spec builder for analyzing Accumulative Attack logs with RAG + GPT.
    Add or modify log file paths as needed.
    """
    base_query = (
        "You are a machine learning results analysis expert."
        "accuracy degradation detection rate"
    )
    return [
        StepSpec(
            title="Analyse log",
            command="",
            log_path=f"{LOG_DIR}/accumulative_train.log",
            analysis_prompt=(
                "Summarize the progress, outputs, and warning signals from the following log concisely."
                "Summarize concisely the effects of the beta and normalization settings, and the signs of collapse or stability, with supporting evidence."
            ),
            rag_request=_concat_rag(base_query + " online training log interpretation",
                                    _load_topk_for_step("train")),
        ),
        StepSpec(
            title="Analyse log",
            command="",
            log_path=f"{LOG_DIR}/accumulative_train.log",
            analysis_prompt=(
                "Based on the analyses from steps 1-2, "
                "summarize the overall behavioral pattern and determine what type of attack (if any) occurred. "
                "If it matches a known method, specify which paper's technique it resembles. "
                "If it is not an attack, explain why. "
                "Finally, describe the results obtained so far for steps 1 through 2 in a concise summary."
                "Choose exactly one from the following: Brainwash, Accumulative Attack, or Label Flipping. "
            ),
            rag_request=_concat_rag(base_query + " final evaluation analysis",
                                    _load_topk_for_step("train")),
        ),
    ]


def build_mmd_backdoor_specs() -> List[StepSpec]:
    """
    Backdoor (MMD) training + visualization pipeline based on Multi-Level MMD Regularization
    """
    base_query = (
        "multi-level mmd regularization backdoor training vgg11 cifar10 "
        "mlmmdr_lamb mlmmdr_layer all trigger detection"
    )

    # 1) MMD Training
    step1_cmd = (
        "python /path/to/backdoor/Multi-Level-MMD-Regularization/train.py "
        "--data_path /path/to/data/cifar-10-batches-py "
        "--data_name cifar10 "
        "--model_name vgg11 "
        "--mlmmdr_lamb 0.1 "
        "--mlmmdr_layer all"
    )

    # 2) MMD Visualization (uses the trained weights)
    step2_cmd = (
        "python visualize.py "
        "--data_path /path/to/data/cifar-10-batches-py "
        "--data_name cifar10 "
        "--model_name vgg11 "
        "--mlmmdr_lamb 0.1 "
        "--mlmmdr_layer all "
        "--weight_path /path/to/backdoor/Multi-Level-MMD-Regularization/weights"
    )

    # Top-K injection (optional): step2 appends the topk produced by step1
    step2_topk = _load_topk_for_step("mmd_step1")

    return [
        StepSpec(
            title="MMD Backdoor 1/2: MMD Training",
            command=step1_cmd,
            log_path=f"{LOG_DIR}/mmd_step1_train.log",
            expected_artifacts=[
                # Broad patterns so any of pt/pth/pkl save formats are matched
                f"{BASE_DIR}/**/*.pth",
                f"{BASE_DIR}/**/*.pt",
                f"{BASE_DIR}/**/*.pkl",
            ],
            analysis_prompt=(
                "Summarize MMD training behavior (loss/acc trends, stability, NaNs, divergence). "
                "If backdoor-related signals appear, cite concrete log evidence."
            ),
            rag_request=_concat_rag(base_query, ""),
        ),
        StepSpec(
            title="MMD Backdoor 2/2: MMD Visualization",
            command=step2_cmd,
            log_path=f"{LOG_DIR}/mmd_step2_visualize.log",
            expected_artifacts=[
                # If visualize produces something like figure_path, it is best to align these patterns to it
                f"{BASE_DIR}/**/*.png",
                f"{BASE_DIR}/**/*.jpg",
                f"{BASE_DIR}/**/*.jpeg",
                f"{BASE_DIR}/**/*.pdf",
            ],
            analysis_prompt=(
                "Summarize what the visualization outputs indicate. "
                "Extract any quantitative or categorical signals supporting backdoor presence/absence."
            ),
            rag_request=_concat_rag(base_query + " visualization interpretation", step2_topk),
        ),
    ]


def build_mmd_backdoor_cifar100_specs() -> List[StepSpec]:
    """
    Backdoor (MMD) training + visualization pipeline based on Multi-Level MMD Regularization (CIFAR-100)
    """
    base_query = (
        "multi-level mmd regularization backdoor training vgg11 cifar100 "
        "mlmmdr_lamb mlmmdr_layer all trigger detection"
    )

    # 1) MMD Training (CIFAR-100)
    step1_cmd = (
        "python /path/to/backdoor/Multi-Level-MMD-Regularization/train.py "
        "--data_path /path/to/data/cifar-100-python "
        "--data_name cifar100 "
        "--model_name vgg11 "
        "--mlmmdr_lamb 0.1 "
        "--mlmmdr_layer all"
    )

    # 2) MMD Visualization (CIFAR-100)
    # Relative paths may break depending on where visualize.py is run from,
    # so it is recommended to hard-code absolute paths when possible.
    step2_cmd = (
        "python /path/to/backdoor/Multi-Level-MMD-Regularization/visualize.py "
        "--data_path /path/to/data/cifar-100-python "
        "--data_name cifar100 "
        "--model_name vgg11 "
        "--mlmmdr_lamb 0.1 "
        "--mlmmdr_layer all "
        "--weight_path /path/to/backdoor/Multi-Level-MMD-Regularization/weights"
    )

    # (optional) Top-K injection: to use a CIFAR100-specific step key, the step name must match accordingly
    step2_topk = _load_topk_for_step("mmd_c100_step1")

    return [
        StepSpec(
            title="MMD Backdoor CIFAR100 1/2: MMD Training",
            command=step1_cmd,
            log_path=f"{LOG_DIR}/mmd_cifar100_step1_train.log",
            expected_artifacts=[
                f"{BASE_DIR}/**/*.pth",
                f"{BASE_DIR}/**/*.pt",
                f"{BASE_DIR}/**/*.pkl",
            ],
            analysis_prompt=(
                "Summarize MMD training behavior (loss/acc trends, stability, NaNs, divergence). "
                "If backdoor-related signals appear, cite concrete log evidence."
            ),
            rag_request=_concat_rag(base_query, ""),
        ),
        StepSpec(
            title="MMD Backdoor CIFAR100 2/2: MMD Visualization",
            command=step2_cmd,
            log_path=f"{LOG_DIR}/mmd_cifar100_step2_visualize.log",
            expected_artifacts=[
                f"{BASE_DIR}/**/*.png",
                f"{BASE_DIR}/**/*.jpg",
                f"{BASE_DIR}/**/*.jpeg",
                f"{BASE_DIR}/**/*.pdf",
            ],
            analysis_prompt=(
                "Summarize what the visualization outputs indicate. "
                "Extract any quantitative or categorical signals supporting backdoor presence/absence."
            ),
            rag_request=_concat_rag(base_query + " visualization interpretation", step2_topk),
        ),
    ]


def build_detect_specs() -> List[StepSpec]:
    """
    Execution spec for Neural-Relation-Graph detect.py
    """
    base = BASE_DIR
    base_query = "Neural Relation Graph label noise outlier detection"

    return [
        StepSpec(
            title="Detect: NRG detect.py",
            command=(
                "CUDA_VISIBLE_DEVICES=0 python /path/to/Detect/Neural-Relation-Graph/detect.py "
                "-n mae_large_49 --pow 4 --cache_dir /path/to/Detect/Neural-Relation-Graph/results"
            ),
            log_path=f"{LOG_DIR}/detect_mae_large",
            expected_artifacts=[
                # If detect.py's output folder is something like results_tiny, adjust these to match
                f"{base}/Detect/Neural-Relation-Graph/**/mae_large*",
                f"{base}/Detect/Neural-Relation-Graph/**/*.npy",
                f"{base}/Detect/Neural-Relation-Graph/**/*.json",
                f"{base}/Detect/Neural-Relation-Graph/**/*.png",
            ],
            analysis_prompt = (
                "You are a machine learning security analyst specializing in label-noise and data-poisoning detection. "
                "Analyze the outputs of detect.py using the following detection signals: "
                "CWE (Confident Wrong Examples), Entropy, Least-Confidence, Loss, Margin, TracIn, and Relation-based methods. "
                "For each signal, you are given evaluation metrics including ROC-AUC, Average Precision (AP), and TNR@95%TPR. "

                "Interpret the metrics using the following quantitative definitions and criteria: "

                "ROC-AUC represents the probability that a noisy (poisoned) sample is assigned a higher noise score than a clean sample. "
                "Values close to 0.5 indicate no discriminative power, while values significantly above 0.8 indicate strong separability "
                "between clean and poisoned samples. "

                "Average Precision (AP) measures the proportion of noisy samples among the top-ranked samples when all samples are sorted "
                "by descending noise score. The overall dataset noise ratio is 8%. Therefore, an AP value substantially higher than 0.08 "
                "indicates strong enrichment of noisy samples in the top-ranked region and is considered strong evidence of label poisoning. "

                "TNR@95 (True Negative Rate at 95% True Positive Rate) measures the fraction of clean samples correctly identified as clean "
                "when 95% of noisy samples are detected. A TNR@95 around 0.70 means that approximately 70% of clean samples are preserved "
                "and not falsely flagged as noisy, indicating a practically useful detection signal. "

                "Based strictly on these metrics, determine whether the dataset is affected by label poisoning. "
                "If label poisoning is present, estimate the contamination ratio as a single percentage value. "
                "Identify which detection signals provide the strongest evidence, prioritizing signals with high ROC-AUC, "
                "AP significantly exceeding the base noise rate (8%), and stable TNR@95. "

                "If different signals disagree, resolve the conflict by weighting them according to their reliability, "
                "consistency across metrics, and robustness to noise. "

                "Do not hedge or use uncertain language. "
                "Choose exactly one final verdict: 'Label Poisoning Detected' or 'No Label Poisoning Detected'. "
                "If any required metrics are missing, explicitly state what is missing and how this limits the conclusion. "
                "Summarize your conclusion concisely with clear, evidence-based reasoning."
            )
            ,
            rag_request=_concat_rag(base_query, ""),
        )
    ]

def build_rethink_specs() -> List[StepSpec]:
    """
    Execution spec for RethinkingLabelPoisoningForGNNs.
    """
    base = BASE_DIR

    base_query = (
        "label poisoning GNN meta label poisoning LAFak cora_ml large_val "
        "GCN attack success rate validation accuracy degradation"
    )

    cmd = (
        "python /path/to/PoisoningAttack/RethinkingLabelPoisoningForGNNs/meta-label-poisoning/meta_label.py --dataset citeseer --setting cv --attack meta --model GCN"
    )

    return [
        StepSpec(
            title="Rethink",
            command=cmd,
            log_path=f"{LOG_DIR}/rethink_meta_label_cora_ml_lafak_gcn.log",
            expected_artifacts=[
                # Result file format is uncertain, so patterns are kept wide (narrow them down if desired)
                f"{base}/**/*.pth",
                f"{base}/**/*.pt",
                f"{base}/**/*.pkl",
                f"{base}/**/*.csv",
                f"{base}/**/*.json",
                f"{base}/**/*.npy",
            ],
            analysis_prompt=(
                "Analyze the meta_label.py execution log experiment.\n"
                "Extract key metrics (train/val/test accuracy, attack success, best hyperparameters, "
                "and any reported poisoning rate or perturbation budget).\n"
                "Conclude decisively whether the run indicates a successful label-poisoning attack, "
                "and summarize evidence (numbers + log lines)."
            ),
            rag_request=_concat_rag(base_query, ""),  # step1 has no external injection
        )
    ]


def build_rethink_pubmed_specs() -> List[StepSpec]:
    """
    Execution spec for RethinkingLabelPoisoningForGNNs on PubMed.
    """
    base = BASE_DIR

    base_query = (
        "label poisoning GNN meta label poisoning pubmed cv "
        "GCN learning rate 0.001 attack success rate validation accuracy degradation"
    )

    cmd = (
        "python /path/to/PoisoningAttack/"
        "RethinkingLabelPoisoningForGNNs/meta-label-poisoning/meta_label.py "
        "--dataset pubmed --setting cv --attack meta --model GCN --lr 0.001"
    )

    return [
        StepSpec(
            title="Rethink_pub",
            command=cmd,
            log_path=f"{LOG_DIR}/rethink_meta_label_pubmed_meta_gcn.log",
            expected_artifacts=[
                f"{base}/**/*.pth",
                f"{base}/**/*.pt",
                f"{base}/**/*.pkl",
                f"{base}/**/*.csv",
                f"{base}/**/*.json",
                f"{base}/**/*.npy",
            ],
            analysis_prompt=(
                "Analyze the meta_label.py execution log experiment.\n"
                "Extract key metrics (train/val/test accuracy, attack success, best hyperparameters, "
                "and any reported poisoning rate or perturbation budget).\n"
                "Conclude decisively whether the run indicates a successful label-poisoning attack, "
                "and summarize evidence (numbers + log lines)."
            ),
            rag_request=_concat_rag(base_query, ""),
        )
    ]


# ------------------------------------------------------------------------------------
# Evasion attacks (test-time adversarial examples): FGSM / PGD / PhysPatch
# ------------------------------------------------------------------------------------
def build_fgsm_specs() -> List[StepSpec]:
    """FGSM / I-FGSM adversarial example generation on a trained MNIST/FMNIST classifier."""
    base_query = "FGSM I-FGSM adversarial example evasion epsilon perturbation Linf classifier accuracy drop"
    fgsm_dir = "/path/to/2nd/evasion/FGSM/FGSM"

    # Load the trained checkpoint (checkpoints/mytest/best_acc.tar) and generate adversarial examples
    gen_cmd = (
        f"cd {fgsm_dir} && python main.py "
        "--mode generate --iteration 1 --epsilon 0.03 "
        "--env_name mytest --load_ckpt best_acc.tar"
    )

    return [
        StepSpec(
            title="FGSM 1/1: Generate Adversarial Examples",
            command=gen_cmd,
            log_path=f"{LOG_DIR}/fgsm_generate.log",
            expected_artifacts=[f"{fgsm_dir}/output/**/*"],
            analysis_prompt=(
                "This is a test-time adversarial-example run. "
                "Report clean vs. adversarial accuracy, the epsilon/iteration used, and how much "
                "accuracy dropped under perturbation. Also use the attached generated images "
                "(legitimate vs. perturbed vs. perturbation/changed) as visual evidence. "
                "Judge from observed BEHAVIOR/metrics only — do NOT rely on the command or script/file "
                "names. If it is evasion, commit to the SPECIFIC method: FGSM (single step, iterations==1) "
                "vs PGD (iterative multi-step). Decide decisively whether an attack occurred or not."
            ),
            rag_request=_concat_rag(base_query, _load_topk_for_step("step1")),
            # Feed the generated adversarial-example images (legitimate/perturbed/changed) to GPT Vision analysis
            vision_globs=[
                f"{fgsm_dir}/output/mytest/legitimate*.jpg",
                f"{fgsm_dir}/output/mytest/perturbed*.jpg",
                f"{fgsm_dir}/output/mytest/changed*.jpg",
            ],
        ),
    ]


def build_pgd_specs() -> List[StepSpec]:
    """PGD (Madry) Linf attack on the MNIST challenge model: build adv set, then evaluate."""
    base_query = "PGD Madry Linf adversarial robustness epsilon k steps MNIST accuracy under attack evasion"
    pgd_dir = "/path/to/2nd/evasion/PGD/mnist_challenge"

    # 1) Generate adversarial examples (uses config.json settings, saves attack.npy)
    step1_cmd = f"cd {pgd_dir} && python pgd_attack.py"
    # 2) Evaluate model accuracy on the saved attack.npy (clean vs adversarial)
    step2_cmd = f"cd {pgd_dir} && python run_attack.py"
    # 3) Render attack.npy (the adversarial-example array) as a grid PNG for Vision analysis
    pgd_grid_png = f"{LOG_DIR}/pgd_adv_grid.png"
    viz_script = "/path/to/Agent/main/pgd_visualize.py"
    step3_cmd = f"cd {pgd_dir} && python {viz_script} {pgd_dir}/attack.npy {pgd_grid_png}"

    step2_topk = _load_topk_for_step("step1")

    return [
        StepSpec(
            title="PGD 1/3: Generate Adversarial Set",
            command=step1_cmd,
            log_path=f"{LOG_DIR}/pgd_attack.log",
            expected_artifacts=[f"{pgd_dir}/*.npy"],
            analysis_prompt=(
                "PGD adversarial example generation step. Report epsilon, number of steps (k), "
                "step size, and the maximum perturbation found. Summarize whether generation succeeded."
            ),
            rag_request=_concat_rag(base_query, ""),
        ),
        StepSpec(
            title="PGD 2/3: Evaluate Model Under Attack",
            command=step2_cmd,
            log_path=f"{LOG_DIR}/pgd_run_attack.log",
            analysis_prompt=(
                "Based on step 1, evaluate the test-time adversarial result. Report natural accuracy vs. "
                "accuracy on the adversarial examples, the perturbation budget (epsilon), step size, and the "
                "number of iteration steps performed. Judge from observed BEHAVIOR/metrics only — do NOT rely "
                "on the command or script/file names. If it is evasion, commit to the SPECIFIC method: "
                "PGD (iterative multi-step Linf) vs FGSM (single step). Decide decisively whether a successful "
                "attack occurred (large accuracy drop under Linf perturbation) or not."
            ),
            rag_request=_concat_rag(base_query + " final evaluation accuracy under attack", step2_topk),
        ),
        StepSpec(
            title="PGD 3/3: Visualize Adversarial Examples",
            command=step3_cmd,
            log_path=f"{LOG_DIR}/pgd_visualize.log",
            expected_artifacts=[pgd_grid_png],
            analysis_prompt=(
                "Use the attached rendered grid of PGD adversarial examples (from attack.npy) as visual "
                "evidence. Describe the visible Linf perturbation noise on the digits and, together with "
                "the accuracy numbers from the previous step, decide decisively whether a PGD evasion "
                "attack occurred or not."
            ),
            rag_request=_concat_rag(base_query + " adversarial example visualization perturbation grid", step2_topk),
            # Feed the rendered adversarial-example grid image to GPT Vision analysis
            vision_globs=[pgd_grid_png],
        ),
    ]


def build_physpatch_specs() -> List[StepSpec]:
    """PhysPatch: physically-realizable adversarial patch attack on a VLM-based autonomous-driving
    perception pipeline, run as the real 5-stage workflow:

      1) SoM segmentation (SAM) -> region masks/labels
      2) GPT picks patch placement coordinates -> coords.txt
      3) Optimize & apply the adversarial patch -> results/pgd/images
      4) Query the target VLM (gpt-4o) on the patched images -> <model>_response.txt
      5) Evaluate how well the VLM was fooled vs. a reference text (attack success / similarity)

    NOTE on API keys: stages 2/4/5 read the key from the OPENAI_API_KEY environment variable
    (the agent already sets it). The key is NEVER hardcoded here. Make sure OPENAI_API_KEY is
    exported before running. Requires the SAM checkpoint at SoM/checkpoints/sam_vit_h_4b8939.pth.
    """
    base_query = (
        "PhysPatch physical adversarial patch VLM multimodal large language model autonomous driving "
        "attack success rate transferability perception manipulation stop sign"
    )
    phys_dir = "/path/to/2nd/evasion/Physpatch/physpatch/code"
    som_dir = f"{phys_dir}/SoM"
    key = '"$OPENAI_API_KEY"'  # shell-expanded at run time; never store the literal key

    # PhysPatch requires a separate conda environment (physpatch, including detectron2).
    # Since `conda activate` does not work under /bin/sh, invoke that env's python directly by absolute path.
    # (Overridable via the PHYSPATCH_PYTHON environment variable.)
    py = os.environ.get("PHYSPATCH_PYTHON", "/path/to/anaconda3/envs/physpatch/bin/python")

    # 1) Set-of-Marks segmentation (SAM) over clean driving images
    step1_cmd = (
        f"cd {som_dir} && {py} batch_som.py "
        f"--input_dir {phys_dir}/data/clean "
        "--output_dir ./som --label_dir ./sam_label "
        "--sam_ckpt ./checkpoints/sam_vit_h_4b8939.pth "
        "--granularity 2.6 --alpha 0.1 --label_mode Number --anno_mode Mask Mark"
    )
    # 2) GPT chooses patch-placement coordinates -> coords.txt
    step2_cmd = (
        f"cd {phys_dir} && {py} som_gpt.py "
        "--original_folder ./data/clean --sam_folder ./SoM/som --label_folder ./SoM/sam_label "
        f"--output_path ./coords.txt --api_key {key} --model gpt-4o"
    )
    # 3) Optimize & apply the physical adversarial patch -> results/pgd/images
    step3_cmd = (
        f"cd {phys_dir} && {py} main.py "
        "--cle_data_path ./data/clean --tgt_data_path ./data/target "
        "--output_dir ./results/pgd --txt_path ./coords.txt "
        "--epsilon 16 --alpha 1.0 --num_iters 300 --num_samples 1000 "
        "--K 10 --tau0 0.6 --beta 0.002 --sigma 0.2 --lr_mask 0.15 --eta 1.0"
    )
    # 4) Query the target VLM (gpt-4o) on the patched images -> results/gpt-4o_response.txt
    step4_cmd = (
        f"cd {phys_dir} && {py} vlm_response.py "
        "--image_dir ./results/pgd/images --output_dir ./results --model gpt-4o "
        f"--api_key {key} "
        "--query \"Describe the main object in the scene that is most likely to influence the "
        "vehicle's next driving decision. You only need to describe the object in JSON format "
        "{'object': ,'describe:' }.\""
    )
    # 5) Evaluate attack success: similarity of VLM answers to the attacker's reference text
    step5_cmd = (
        f"cd {phys_dir} && {py} evaluation.py "
        "--file_path ./results/gpt-4o_response.txt --model_name gpt-4o "
        "--reference_text \"A stop sign is visible\" --start 0 --end 1000 "
        f"--api_key {key}"
    )

    topk2 = _load_topk_for_step("step1")
    topk3 = _load_topk_for_step("step2")
    topk4 = _load_topk_for_step("step3")
    topk5 = _load_topk_for_step("step4")

    return [
        StepSpec(
            title="PhysPatch 1/5: SoM Segmentation (SAM)",
            command=step1_cmd,
            log_path=f"{LOG_DIR}/physpatch_1_som.log",
            expected_artifacts=[f"{som_dir}/som/**/*", f"{som_dir}/sam_label/**/*"],
            analysis_prompt=(
                "Set-of-Marks segmentation step (SAM) over clean driving-scene images. "
                "Report how many images/regions were segmented and whether masks/labels were produced. "
                "This is preprocessing for a physical adversarial-patch attack; summarize health only."
            ),
            rag_request=_concat_rag(base_query + " set of marks SAM segmentation", topk2),
        ),
        StepSpec(
            title="PhysPatch 2/5: GPT Patch-Placement Coordinates",
            command=step2_cmd,
            log_path=f"{LOG_DIR}/physpatch_2_coords.log",
            expected_artifacts=[f"{phys_dir}/coords.txt"],
            analysis_prompt=(
                "A VLM (gpt-4o) selects WHERE to place the adversarial patch in each scene, written to "
                "coords.txt. Report how many coordinates were produced and whether placement targets "
                "decision-relevant objects. Judge from behavior/outputs, not from file/command names."
            ),
            rag_request=_concat_rag(base_query + " patch placement coordinates region selection", topk3),
        ),
        StepSpec(
            title="PhysPatch 3/5: Optimize & Apply Adversarial Patch",
            command=step3_cmd,
            log_path=f"{LOG_DIR}/physpatch_3_attack.log",
            expected_artifacts=[f"{phys_dir}/results/pgd/images/*.png", f"{phys_dir}/results/pgd/images/*.jpg"],
            analysis_prompt=(
                "Core PhysPatch attack: a localized, physically-realizable adversarial PATCH is optimized "
                "(epsilon=16, iterative, dynamic mask) and applied onto the driving images. Report the "
                "loss/optimization trend, epsilon, iterations, number of patched images produced. Using the "
                "attached patched images as visual evidence, confirm this is a localized physical-patch evasion "
                "(PhysPatch) rather than full-image L-inf noise. Judge from behavior, not file/command names."
            ),
            rag_request=_concat_rag(base_query + " adversarial patch optimization dynamic mask epsilon iterations", topk4),
            # Analyze the patch-applied result images with GPT Vision
            vision_globs=[
                f"{phys_dir}/results/pgd/images/*.png",
                f"{phys_dir}/results/pgd/images/*.jpg",
            ],
        ),
        StepSpec(
            title="PhysPatch 4/5: Target VLM Response",
            command=step4_cmd,
            log_path=f"{LOG_DIR}/physpatch_4_vlm_response.log",
            expected_artifacts=[f"{phys_dir}/results/gpt-4o_response.txt"],
            analysis_prompt=(
                "The target VLM (gpt-4o) is queried on the PATCHED images to describe the main "
                "driving-decision object. Report whether the model's answers appear manipulated toward a "
                "wrong/target object (the vulnerability being probed). Summarize representative responses."
            ),
            rag_request=_concat_rag(base_query + " VLM response manipulation misperception driving decision", topk5),
        ),
        StepSpec(
            title="PhysPatch 5/5: Evaluate Attack Success",
            command=step5_cmd,
            log_path=f"{LOG_DIR}/physpatch_5_eval.log",
            analysis_prompt=(
                "Final evaluation: per-image similarity of the VLM's answers to the attacker's reference "
                "text ('A stop sign is visible'), with a pass count (score>0.5) and mean similarity = the "
                "attack-success metric. Report Total passed (acc/N) and Mean Similarity, and decide decisively "
                "whether the PhysPatch physical-patch attack successfully fooled the VLM-based perception "
                "(a safety-critical vulnerability) or not."
            ),
            rag_request=_concat_rag(base_query + " attack success rate mean similarity evaluation safety", topk5),
        ),
    ]
