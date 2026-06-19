#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

RUN_ROOT_DIR="${RUN_ROOT_DIR:-runs/polar-vla}"
VLA_RUN_ID="${VLA_RUN_ID:-prism-dinosiglip-224px+mx-bridge+n1+b32+x42--bridge50k--image_aug-Latent-Action-Pretraining}"
VLA_RUN_DIR="${VLA_RUN_DIR:-${RUN_ROOT_DIR}/${VLA_RUN_ID}}"
VLA_CKPT_STEP="${VLA_CKPT_STEP:-050000}"
CKPT_NAME="${CKPT_NAME:-}"
OUTPUT_HF_MODEL_LOCAL_PATH="${OUTPUT_HF_MODEL_LOCAL_PATH:-weights/polar_bridge_vla}"
LLM_TOKENIZER_PATH="${LLM_TOKENIZER_PATH:-/path/to/llama2-7b-hf}"
ACTION_VOCAB_SIZE="${ACTION_VOCAB_SIZE:-32}"
LATENT_ACTION_TOKEN_LEN="${LATENT_ACTION_TOKEN_LEN:-5}"
HF_TOKEN="${HF_TOKEN:-}"

if [[ -z "${CKPT_NAME}" ]]; then
  shopt -s nullglob
  candidates=("${VLA_RUN_DIR}"/checkpoints/step-"${VLA_CKPT_STEP}"-*.pt)
  shopt -u nullglob
  if [[ "${#candidates[@]}" -ne 1 ]]; then
    echo "Expected one checkpoint matching ${VLA_RUN_DIR}/checkpoints/step-${VLA_CKPT_STEP}-*.pt; found ${#candidates[@]}." >&2
    echo "Set CKPT_NAME explicitly if needed." >&2
    exit 1
  fi
  CKPT_NAME="$(basename "${candidates[0]}")"
fi

python vla-scripts/extern/convert_polar_weights_to_hf.py \
  --openvla_model_path_or_id "${VLA_RUN_DIR}" \
  --ckpt_name "${CKPT_NAME}" \
  --output_hf_model_local_path "${OUTPUT_HF_MODEL_LOCAL_PATH}" \
  --llm_tokenizer_path "${LLM_TOKENIZER_PATH}" \
  --hf_token "${HF_TOKEN}" \
  --action_vocab_size "${ACTION_VOCAB_SIZE}" \
  --latent_action_token_len "${LATENT_ACTION_TOKEN_LEN}" \
  "$@"
