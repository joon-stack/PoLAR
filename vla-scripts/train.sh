GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NNODES=${NNODES:-1}
MASTER_PORT=${MASTER_PORT:-28596}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
RANK=${RANK:-0}

PRETRAIN_VLM=${PRETRAIN_VLM:-/path/to/prism-dinosiglip-224px-7b}
LAM_PATH=${LAM_PATH:-/path/to/polar-tokenizer.ckpt}
DATA_ROOT_DIR=${DATA_ROOT_DIR:-/path/to/rlds_data_collection}
RUN_ROOT_DIR=${RUN_ROOT_DIR:-runs/polar-vla}

# Latent VLA training for the PoLAR release path.
torchrun --nproc_per_node ${GPUS_PER_NODE} \
    --nnodes ${NNODES} \
    --node_rank ${RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    train.py \
    --vla.type prism-dinosiglip-224px+mx-bridge \
    --pretrain_vlm ${PRETRAIN_VLM} \
    --lam_path ${LAM_PATH} \
    --data_root_dir ${DATA_ROOT_DIR} \
    --run_root_dir ${RUN_ROOT_DIR}
