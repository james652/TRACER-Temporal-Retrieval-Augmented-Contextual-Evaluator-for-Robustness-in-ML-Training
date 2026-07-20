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