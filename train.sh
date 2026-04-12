#!/bin/bash
config_file=./config/vamoe.yaml
config='vamoe'
run_num='1'

# Add timestamp to experiment name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NAME='vit_rebuttal_zqu10_49_0.0002AdamW_CosLR_trainl2loss0.1_useMoE_ChannelMoE_GradClip_patch2channel768_cl_0515'_${TIMESTAMP}

checkpoint=""

LOG_DIR="./logs/${NAME}/"
mkdir -p -- "$LOG_DIR"


CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train.py \
            --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num --exp_dir=$LOG_DIR --checkpoint=$checkpoint \
            > ${LOG_DIR}train.log 2>&1
