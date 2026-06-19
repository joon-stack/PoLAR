<div align="center">

# PoLAR

**Official PyTorch implementation of**
### *PoLAR: Factorizing Extent and Mode in Latent Actions for Robot Policy Learning*


[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-43A047?style=flat-square)](LICENSE)

Youngjoon Jeong · Jihwan Yu · Minsoo Jo · Junha Chun · Taesup Kim

[![Project Page](https://img.shields.io/badge/Project%20Page-coming%20soon-4285F4?style=for-the-badge)](#)
[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](#citation)
[![Models](https://img.shields.io/badge/Hugging%20Face-models-FFD21E?style=for-the-badge)](https://huggingface.co/quiet-storm/polar-bridge-vla)


</div>

---

PoLAR learns discrete latent actions for robot policy learning by factorizing transition **extent** and transition **mode**. The tokenizer uses temporal offset as weak supervision for radial progress, then exposes a VLA-friendly action-token interface: one radial token followed by direction tokens.

This repository contains the PoLAR tokenizer, latent VLA training code, downstream action-decoder fine-tuning, and SimplerEnv evaluation utilities. The VLA stack is based on UniVLA and Prismatic/OpenVLA-style infrastructure, with the public path narrowed to the PoLAR paper workflow.

## Installation

```bash
conda create -n polar python=3.10 -y
conda activate polar
conda install -c conda-forge libgl -y

python -m pip install --upgrade pip

# PoLAR follows the UniVLA/OpenVLA stack; our runs use torch 2.2.0 + CUDA 12.1.
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# Install PoLAR plus the SimplerEnv/ManiSkill3 evaluation stack.
pip install -e ".[simplerenv]"

python -m pip check
```

For large VLA training, FlashAttention is recommended when it matches your CUDA/PyTorch stack:

```bash
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation
```

Pinned base dependencies are listed in [`requirements.txt`](requirements.txt). The core dependency stack follows UniVLA/OpenVLA, including `torch==2.2.0` and `torchvision==0.17.0`; the README installs the CUDA 12.1 PyTorch wheels explicitly before the editable package install. The `simplerenv` extra installs the Python packages needed by the SimplerEnv/ManiSkill3 evaluator, and the first evaluator run may download ManiSkill task and robot assets.

## Data

PoLAR tokenizer and latent VLA training use BridgeData V2 in RLDS format. Set the dataset arguments in the commands below to your local RLDS cache. Downstream action-decoder fine-tuning can use the same Bridge-style RLDS data or a prebuilt offline LeRobot window cache.

## Checkpoints

| Artifact | Training Data | Link |
| :--- | :--- | :--- |
| PoLAR tokenizer | BridgeData V2 | [polar-tokenizer-bridge](https://huggingface.co/quiet-storm/polar-tokenizer-bridge) |
| Bridge latent VLA checkpoint | BridgeData V2 | [polar-bridge-vla](https://huggingface.co/quiet-storm/polar-bridge-vla) |

Expected local layout after downloading weights:

```text
weights/
  polar_tokenizer_bridge.ckpt
  polar_bridge_vla/
  # Optional: downstream fine-tuned policy and action decoder for SimplerEnv evaluation.
```

## Train

### PoLAR Tokenizer

Train the PoLAR tokenizer on BridgeData V2 RLDS:

```bash
cd latent_action_model

torchrun --standalone --nnodes 1 --nproc-per-node 8 main_visual_vq.py fit \
    --config config/polar_tokenizer_bridge.yaml \
    --data.data_root /path/to/rlds_bridge_orig \
    --model.log_path /path/to/output/lam_bridge/logs
```

### Latent VLA Training

```bash
cd vla-scripts

torchrun --nproc_per_node 8 train.py \
    --vla.type prism-dinosiglip-224px+mx-bridge \
    --trackers jsonl \
    --pretrain_vlm /path/to/prism-dinosiglip-224px-7b \
    --lam_path /path/to/polar_tokenizer.ckpt \
    --lam_config_path /path/to/polar_tokenizer_bridge.yaml \
    --data_root_dir /path/to/rlds_data_collection \
    --run_root_dir runs/polar-vla
```

PoLAR uses a five-token discrete interface in the VLA path: `1 radial + 4 direction` tokens. With the default 16-radius/16-direction factorization, radial IDs occupy `<ACT_0>` through `<ACT_15>`, while direction IDs occupy `<ACT_16>` through `<ACT_31>`.

### Downstream Fine-tuning

Bridge/SimplerEnv-style action-decoder fine-tuning:

```bash
cd vla-scripts

WANDB_MODE=disabled torchrun --standalone --nnodes 1 --nproc-per-node 8 finetune_bridge.py \
    --vla_path /path/to/pretrained-polar-vla \
    --lam_path /path/to/polar_tokenizer.ckpt \
    --lam_config_path /path/to/polar_tokenizer_bridge.yaml \
    --data_root_dir /path/to/rlds_data \
    --dataset_name bridge \
    --run_root_dir runs/polar-simpler
```

To fine-tune from a prebuilt offline LeRobot window cache, use the same script with `--dataset_format lerobot_cache`:

```bash
WANDB_MODE=disabled torchrun --standalone --nnodes 1 --nproc-per-node 8 finetune_bridge.py \
    --vla_path /path/to/pretrained-polar-vla \
    --lam_path /path/to/polar_tokenizer.ckpt \
    --lam_config_path /path/to/polar_tokenizer_bridge.yaml \
    --dataset_format lerobot_cache \
    --lerobot_cache_dir /path/to/lerobot_window_cache \
    --dataset_name bridge \
    --run_root_dir runs/polar-simpler
```

## Evaluation

### SimplerEnv

The installation command above includes the SimplerEnv/ManiSkill3 Python stack. Check the simulator setup with a random-action smoke run first:

```bash
cd experiments/robot/simpler-bridge
XLA_PYTHON_CLIENT_PREALLOCATE=false python real2sim_eval_maniskill3.py \
    -e PutCarrotOnPlateInScene-v1 \
    --num-episodes 1 \
    --num-envs 1 \
    --record-dir /tmp/polar-simpler-random-smoke \
    --no-save-video
```

Run PoLAR evaluation one task at a time with a downstream fine-tuned policy and matching action decoder:

```bash
cd experiments/robot/simpler-bridge

XLA_PYTHON_CLIENT_PREALLOCATE=false python real2sim_eval_maniskill3.py \
    --model polar \
    --ckpt-path /path/to/downstream_policy \
    --action-decoder-path /path/to/downstream_policy/action_decoder.pt \
    -e PutCarrotOnPlateInScene-v1 \
    --num-episodes 24 \
    --num-envs 1 \
    --pred-action-horizon 10 \
    --record-dir videos/polar_carrot \
    --no-save-video
```

## Citation

Citation information will be updated when the paper is public. For now, please use:

```bibtex
@misc{jeong2026polar,
  title = {PoLAR: Factorizing Extent and Mode in Latent Actions for Robot Policy Learning},
  author = {Jeong, Youngjoon and Yu, Jihwan and Jo, Minsoo and Chun, Junha and Kim, Taesup},
  year = {2026},
  note = {Manuscript in preparation}
}
```

## Acknowledgements

This implementation is based on UniVLA and adapts Prismatic/OpenVLA-style VLA training infrastructure. We thank the UniVLA, OpenVLA/Prismatic, ManiSkill, SimplerEnv, and BridgeData V2 authors for releasing the code, environments, and datasets that make this work possible.

PoLAR is released under the Apache-2.0 license in this repository. Third-party components remain subject to their original licenses; please keep upstream attribution and license notices intact when preparing derivative releases.
