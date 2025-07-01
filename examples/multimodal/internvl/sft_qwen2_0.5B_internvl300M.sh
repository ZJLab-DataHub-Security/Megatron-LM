#!/bin/bash
# Run SFT on a pretrained multimodal model

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
MODEL_NAME="mcore-llava-qwen2.0-0.5b-instruct-internvl300M-sft"

# Check that the user has set an output path for model checkpoints.
if [[ -z $WORKSPACE ]]; then
    echo "Please set WORKSPACE for storing your model checkpoints."
    exit 1
fi

SOURCE=`pwd`
OUTPUT_BASE="${WORKSPACE}/output"
OUTPUT="${OUTPUT_BASE}/${MODEL_NAME}"

FINETUNE_DIR=${OUTPUT}/checkpoints
LOGS_DIR="${OUTPUT}/logs"
TENSORBOARD_DIR="${OUTPUT}/tensorboard"

if [[ -z $LOAD_NAME ]]; then
    echo "Please set LOAD_NAME for input model name."
    exit 1
fi

if [[ -z $LOAD_ITER ]]; then
    echo "Please set LOAD_ITER for pre-trained input model iteration."
    exit 1
fi

CHECKPOINT_DIR="${WORKSPACE}/${LOAD_NAME}/"

# DATA_TRAIN="${SOURCE}/examples/multimodal/sft_dataset.yaml"

# DEBUG=0
if [[ $DEBUG -eq 1 ]]; then
    BZ=32
    NW=0
    HD=0.0 #0.0
    LI=1
    EXTRA_ARGS=""
    NONDETERMINISTIC_ATTN=1
else
    BZ=128
    NW=2
    HD=0.1
    LI=10
    EXTRA_ARGS=""
    NONDETERMINISTIC_ATTN=1
fi

OPTIONS=" \
    --apply-layernorm-1p \
    --attention-softmax-in-fp32 \
    --use-checkpoint-args \
    --use-distributed-optimizer \
    --transformer-impl transformer_engine \
    --use-te \
    --normalization RMSNorm \
    --group-query-attention \
    --num-query-groups 2 \
    --no-masked-softmax-fusion \
    --num-workers ${NW} \
    --exit-duration-in-mins 230 \
    --use-flash-attn \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --rotary-percent 1.0 \
    --rotary-base 1000000 \
    --swiglu \
    --attention-dropout 0.0 \
    --hidden-dropout ${HD} \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 24 \
    --hidden-size 896 \
    --num-attention-heads 14 \
    --seq-length 4096 \
    --decoder-seq-length 4096 \
    --max-position-embeddings 32768 \
    --ffn-hidden-size 4864 \
    --train-iters 31 \
    --micro-batch-size 2 \
    --global-batch-size ${BZ} \
    --lr-decay-iters 31 \
    --lr-warmup-fraction .03 \
    --lr-warmup-iters 0 \
    --lr 4e-5 \
    --min-lr 0.0 \
    --lr-decay-style cosine \
    --log-interval ${LI} \
    --eval-iters 0 \
    --eval-interval 500 \
    --seed 42 \
    --tokenizer-type MultimodalTokenizer \
    --tokenizer-model /workspace/models/InternVL2-1B/ \
    --tokenizer-prompt-format qwen2p0 \
    --data-path ${DATA_TRAIN} \
    --prompt-path ${SOURCE}/examples/multimodal/internvl/manual_prompts.json \
    --save-interval 500 \
    --save ${FINETUNE_DIR} \
    --load ${FINETUNE_DIR} \
    --pretrained-checkpoint ${CHECKPOINT_DIR} \
    --dataloader-save ${FINETUNE_DIR}/dataloader \
    --split 100,0,0 \
    --clip-grad 1.0 \
    --weight-decay 1e-2 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --init-method-std 0.02 \
    --no-initialization \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --eod-mask-loss \
    --freeze-ViT \
    --patch-dim 14 \
    --img-h 448 \
    --img-w 448 \
    --dataloader-type external \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --language-model-type=qwen2.0_0.5B \
    --vision-model-type=internvit300M \
    --pixel-shuffle \
    --max-num-tiles 6 \
    --use-thumbnail \
    --force-system-message \
    --use-tiling \
    --bf16 \
    --disable-vision-class-token \
    --recompute-activations \
    --use-linspace-drop-path \
    --drop-path-rate 0.1 \
    ${EXTRA_ARGS} \
    --distributed-timeout-minutes 60 \
    --ckpt-format torch
"

export NVTE_APPLY_QK_LAYER_SCALING=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=${NONDETERMINISTIC_ATTN}

torchrun --nproc_per_node 8 examples/multimodal/train.py ${OPTIONS}
