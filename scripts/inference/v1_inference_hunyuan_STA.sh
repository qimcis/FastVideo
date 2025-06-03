#!/bin/bash

num_gpus=2
export FASTVIDEO_ATTENTION_CONFIG=assets/mask_strategy_hunyuan.json
export FASTVIDEO_ATTENTION_BACKEND=SLIDING_TILE_ATTN
export MODEL_BASE=hunyuanvideo-community/HunyuanVideo
# export MODEL_BASE=hunyuanvideo-community/HunyuanVideo
# Note that the tp_size and sp_size should be the same and equal to the number
# of GPUs. They are used for different parallel groups. sp_size is used for
# dit model and tp_size is used for encoder models.
fastvideo generate \
    --model-path $MODEL_BASE \
    --sp-size ${num_gpus} \
    --tp-size ${num_gpus} \
    --height 768 \
    --width 1280 \
    --num-frames 117 \
    --num-inference-steps 50 \
    --guidance-scale 1 \
    --embedded-cfg-scale 6 \
    --flow-shift 7 \
    --prompt "A beautiful woman in a red dress walking down a street" \
    --seed 1024 \
    --output-path outputs_video/
