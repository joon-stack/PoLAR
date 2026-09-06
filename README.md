<div align="center">

# PoLAR (CoRL 2026)

**Official PyTorch implementation of**
### *PoLAR: Factorizing Extent and Mode in Latent Actions for Robot Policy Learning*


[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-43A047?style=flat-square)](LICENSE)

Youngjoon Jeong · Jihwan Yu · Minsoo Jo · Junha Chun · Taesup Kim

[![Project Page](https://img.shields.io/badge/Project%20Page-4285F4?style=for-the-badge)](https://joon-stack.github.io/PoLAR/)
[![arXiv](https://img.shields.io/badge/arXiv-2606.21139-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.21139)
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

PoLAR tokenizer and latent VLA training use BridgeData V2 in RLDS format. Set the dataset arguments in the commands below to your local RLDS cache. Downstream action-decoder fine-tuning uses RLDS-format data.

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

The Bridge latent VLA is trained for 50k steps from a Prismatic/OpenVLA-style base VLM and the PoLAR tokenizer checkpoint:

```bash
PRETRAIN_VLM=/path/to/prism-dinosiglip-224px-7b \
LAM_PATH=weights/polar_tokenizer_bridge.ckpt \
LAM_CONFIG_PATH=latent_action_model/config/polar_tokenizer_bridge.yaml \
BRIDGE_DATA_ROOT=/path/to/rlds_bridge_orig \
RUN_ROOT_DIR=runs/polar-vla \
bash vla-scripts/train_polar_bridge_vla.sh
```

The wrapper defaults to 8 GPUs, `--vla.max_steps 50000`, BridgeData V2, JSONL logging, image augmentation, a 20k shuffle buffer, and the five-token PoLAR interface (`1 radial + 4 direction` tokens). The run directory contains `config.yaml`, `config.json`, `dataset_statistics.json`, and `checkpoints/step-*.pt`.

With the default 16-radius/16-direction factorization, radial IDs occupy `<ACT_0>` through `<ACT_15>`, while direction IDs occupy `<ACT_16>` through `<ACT_31>`.

### Convert VLA Checkpoint

Convert the native VLA training checkpoint to Hugging Face format before downstream fine-tuning or evaluation:

```bash
VLA_RUN_DIR=/path/to/vla_run_dir \
VLA_CKPT_STEP=050000 \
LLM_TOKENIZER_PATH=/path/to/llama2-7b-hf \
OUTPUT_HF_MODEL_LOCAL_PATH=weights/polar_bridge_vla \
bash vla-scripts/convert_polar_bridge_vla_to_hf.sh
```

If `CKPT_NAME` is not set, the wrapper looks for exactly one checkpoint matching `checkpoints/step-050000-*.pt` under `VLA_RUN_DIR`.

### Downstream Action-Decoder Fine-tuning

Fine-tune the downstream action decoder from the converted Bridge latent VLA:

```bash
VLA_PATH=weights/polar_bridge_vla \
LAM_PATH=weights/polar_tokenizer_bridge.ckpt \
LAM_CONFIG_PATH=latent_action_model/config/polar_tokenizer_bridge.yaml \
DATA_ROOT_DIR=/path/to/rlds_data \
DATASET_NAME=bridge \
RUN_ROOT_DIR=runs/polar-action-decoder \
bash vla-scripts/finetune_polar_action_decoder.sh
```

The recommended Bridge/Simpler-style setting uses 4 GPUs, batch size 32, 20k steps, checkpoint saves every 5k steps, learning rate `3.5e-4`, no scheduler, shuffle buffer 512, image augmentation, and prediction window size 10. The wrapper also enables train-loop LAM token materialization and lightweight DataLoader prefetching through `POLAR_*` environment variables. Fine-tuning writes the latest action decoder to `action_decoder.pt` in the run directory.

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

```bibtex
@misc{jeong2026polar,
  title = {PoLAR: Factorizing Extent and Mode in Latent Actions for Robot Policy Learning},
  author = {Jeong, Youngjoon and Yu, Jihwan and Jo, Minsoo and Chun, Junha and Kim, Taesup},
  year = {2026},
  eprint = {2606.21139},
  archivePrefix = {arXiv}
}
```

## Acknowledgements

This implementation is based on UniVLA and adapts Prismatic/OpenVLA-style VLA training infrastructure. We thank the UniVLA, OpenVLA/Prismatic, ManiSkill, SimplerEnv, and BridgeData V2 authors for releasing the code, environments, and datasets that make this work possible.

PoLAR is released under the Apache-2.0 license in this repository. Third-party components remain subject to their original licenses; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for upstream attribution and license notes.
