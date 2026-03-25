#!/bin/bash

# DiT Distributed Training Launch Script (8P)
# Usage: bash scripts/train_distributed.sh

set -e

CONFIG_FILE=${1:-"./configs/train_dit_xl_2.yaml"}

echo "Starting DiT-XL/2 Distributed Training..."
echo "Config: $CONFIG_FILE"

# Training parameters
DATASET_PATH="/path/to/imagenet"
OUTPUT_PATH="./output/dit_xl2_dist"

# Create output directory
mkdir -p $OUTPUT_PATH

# Check rank table
if [ -z "$RANK_TABLE_FILE" ]; then
    echo "Error: RANK_TABLE_FILE environment variable not set"
    echo "Please set RANK_TABLE_FILE to your rank table JSON file path"
    exit 1
fi

echo "Using rank table: $RANK_TABLE_FILE"
echo "Device count: $RANK_SIZE"

# Run distributed training
python scripts/train.py \
    --config $CONFIG_FILE \
    data.dataset_path=$DATASET_PATH \
    train.output_path=$OUTPUT_PATH \
    train.num_epochs=100 \
    data.batch_size=8 \
    model.model_type="DiT-XL/2" \
    model.hidden_size=1152 \
    model.num_layers=28 \
    model.num_heads=16 \
    model.head_dim=72 \
    train.optimizer.lr=1e-4 \
    train.scheduler.warmup_steps=1000 \
    train.ema.enable=true \
    train.ema.decay=0.9999 \
    env.distributed=True \
    env.dtype=fp32 \
    env.seed=42

echo "Distributed training completed!"
