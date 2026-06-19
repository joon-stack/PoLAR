#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NNODES="${NNODES:-1}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-28596}"

PRETRAIN_VLM="${PRETRAIN_VLM:-/path/to/prism-dinosiglip-224px-7b}"
LAM_PATH="${LAM_PATH:-weights/polar_tokenizer_bridge.ckpt}"
LAM_CONFIG_PATH="${LAM_CONFIG_PATH:-latent_action_model/config/polar_tokenizer_bridge.yaml}"
BRIDGE_DATA_ROOT="${BRIDGE_DATA_ROOT:-/path/to/rlds_bridge_orig}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-runs/polar-vla}"
RUN_ID_NOTE="${RUN_ID_NOTE:-bridge50k}"
TRACKERS="${TRACKERS:-jsonl}"
VLA_MAX_STEPS="${VLA_MAX_STEPS:-50000}"
VLA_SHUFFLE_BUFFER_SIZE="${VLA_SHUFFLE_BUFFER_SIZE:-20000}"
IMAGE_AUG="${IMAGE_AUG:-true}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

torchrun \
  --nproc_per_node "${GPUS_PER_NODE}" \
  --nnodes "${NNODES}" \
  --node_rank "${RANK}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  vla-scripts/train.py \
  --vla.type prism-dinosiglip-224px+mx-bridge \
  --vla.max_steps "${VLA_MAX_STEPS}" \
  --vla.shuffle_buffer_size "${VLA_SHUFFLE_BUFFER_SIZE}" \
  --trackers "${TRACKERS}" \
  --pretrain_vlm "${PRETRAIN_VLM}" \
  --lam_path "${LAM_PATH}" \
  --lam_config_path "${LAM_CONFIG_PATH}" \
  --lam_kind visual_vq_factorized \
  --lam_token_view indices \
  --latent_action_token_len 5 \
  --action_vocab_size 32 \
  --data_root_dir "${BRIDGE_DATA_ROOT}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id_note "${RUN_ID_NOTE}" \
  --image_aug "${IMAGE_AUG}" \
  "$@"
