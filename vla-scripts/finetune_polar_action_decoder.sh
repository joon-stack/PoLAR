#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NNODES="${NNODES:-1}"
MASTER_PORT="${MASTER_PORT:-28761}"

VLA_PATH="${VLA_PATH:-weights/polar_bridge_vla}"
LAM_PATH="${LAM_PATH:-weights/polar_tokenizer_bridge.ckpt}"
LAM_CONFIG_PATH="${LAM_CONFIG_PATH:-latent_action_model/config/polar_tokenizer_bridge.yaml}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/path/to/rlds_data}"
DATASET_NAME="${DATASET_NAME:-bridge}"
DATASET_FORMAT="${DATASET_FORMAT:-rlds}"
LEROBOT_CACHE_DIR="${LEROBOT_CACHE_DIR:-}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-runs/polar-action-decoder}"
ADAPTER_TMP_DIR="${ADAPTER_TMP_DIR:-runs/polar-action-decoder-adapter-tmp}"
RUN_ID_NOTE="${RUN_ID_NOTE:-action_decoder_ft}"

BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_STEPS="${MAX_STEPS:-20000}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-0.00035}"
USE_SCHEDULER="${USE_SCHEDULER:-false}"
SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-512}"
IMAGE_AUG="${IMAGE_AUG:-true}"
WINDOW_SIZE="${WINDOW_SIZE:-10}"
WANDB_PROJECT="${WANDB_PROJECT:-polar_finetune_bridge}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

export WANDB_MODE="${WANDB_MODE:-disabled}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export POLAR_LAM_IN_TRAIN_LOOP="${POLAR_LAM_IN_TRAIN_LOOP:-1}"
export POLAR_BATCH_LAM_IN_COLLATOR="${POLAR_BATCH_LAM_IN_COLLATOR:-1}"
export POLAR_VLA_DATALOADER_WORKERS="${POLAR_VLA_DATALOADER_WORKERS:-2}"
export POLAR_VLA_DATALOADER_PREFETCH_FACTOR="${POLAR_VLA_DATALOADER_PREFETCH_FACTOR:-2}"
export POLAR_TRAJ_THREADS="${POLAR_TRAJ_THREADS:-4}"
export POLAR_TRAJ_READ_THREADS="${POLAR_TRAJ_READ_THREADS:-4}"
export POLAR_FRAME_THREADS="${POLAR_FRAME_THREADS:-4}"
export POLAR_TF_RAM_BUDGET_MB="${POLAR_TF_RAM_BUDGET_MB:-512}"

cmd=(
  torchrun
  --standalone
  --nnodes "${NNODES}"
  --nproc_per_node "${GPUS_PER_NODE}"
  --master_port "${MASTER_PORT}"
  vla-scripts/finetune_bridge.py
  --vla_path "${VLA_PATH}"
  --lam_path "${LAM_PATH}"
  --lam_config_path "${LAM_CONFIG_PATH}"
  --lam_kind visual_vq_factorized
  --lam_token_view indices
  --latent_action_token_len 5
  --data_root_dir "${DATA_ROOT_DIR}"
  --dataset_name "${DATASET_NAME}"
  --dataset_format "${DATASET_FORMAT}"
  --run_root_dir "${RUN_ROOT_DIR}"
  --adapter_tmp_dir "${ADAPTER_TMP_DIR}"
  --wandb_project "${WANDB_PROJECT}"
  --run_id_note "${RUN_ID_NOTE}"
  --batch_size "${BATCH_SIZE}"
  --grad_accumulation_steps "${GRAD_ACCUMULATION_STEPS}"
  --learning_rate "${LEARNING_RATE}"
  --use_scheduler "${USE_SCHEDULER}"
  --max_steps "${MAX_STEPS}"
  --save_steps "${SAVE_STEPS}"
  --shuffle_buffer_size "${SHUFFLE_BUFFER_SIZE}"
  --image_aug "${IMAGE_AUG}"
  --window_size "${WINDOW_SIZE}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  cmd+=(--wandb_entity "${WANDB_ENTITY}")
fi

if [[ "${DATASET_FORMAT}" == "lerobot_cache" ]]; then
  if [[ -z "${LEROBOT_CACHE_DIR}" ]]; then
    echo "Set LEROBOT_CACHE_DIR when DATASET_FORMAT=lerobot_cache." >&2
    exit 1
  fi
  cmd+=(--lerobot_cache_dir "${LEROBOT_CACHE_DIR}")
fi

"${cmd[@]}" "$@"
