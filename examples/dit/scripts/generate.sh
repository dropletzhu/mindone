#!/bin/bash

# DiT Generation Launch Script
# Usage: bash scripts/generate.sh

set -e

# Model checkpoint (required)
CHECKPOINT=${1:-"./output/dit_xl2/checkpoints/dit_final.ckpt"}
OUTPUT_DIR=${2:-"./generated_images"}
NUM_IMAGES=${3:-4}

echo "Starting DiT Image Generation..."
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT_DIR"
echo "Num images: $NUM_IMAGES"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run generation
python scripts/generate.py \
    --checkpoint $CHECKPOINT \
    --output_dir $OUTPUT_DIR \
    --num_images $NUM_IMAGES \
    --batch_size 1 \
    --num_inference_steps 50 \
    --guidance_scale 4.0 \
    --seed 42 \
    --image_size 256 \
    --dtype fp32 \
    --scheduler ddim \
    --class_labels "tench,goldfish,hen,cock"

echo "Generation completed!"
