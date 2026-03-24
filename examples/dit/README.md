# DiT (Diffusion Transformer) Training and Inference

A comprehensive implementation of DiT for class-conditional image generation using MindSpore.

## 📖 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Training](#training)
- [Inference](#inference)
- [Model Architecture](#model-architecture)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Advanced Features](#advanced-features)
- [Performance Tips](#performance-tips)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

## Overview

DiT (Diffusion Transformer) is a state-of-the-art diffusion model that uses a transformer backbone instead of traditional UNet architectures. It achieves excellent results on ImageNet 256×256 class-conditional generation.

### Key Features

- **Transformer-based Architecture**: Uses pure transformer blocks with adaptive layer normalization (adaLN)
- **Patch-based Processing**: Converts images into token-like patches for efficient processing
- **Class-conditional Generation**: Supports classifier-free guidance for flexible generation
- **Multiple Model Variants**: DiT-XL/2, DiT-L/2, DiT-B/2, DiT-S/2
- **Training Stability**: Supports EMA, gradient checkpointing, and mixed precision training
- **MindSpore Optimized**: Fully compatible with MindSpore framework

### Model Variants

| Model | Hidden Size | Layers | Heads | Parameters | Memory (FP32) |
|-------|-------------|--------|-------|------------|---------------|
| DiT-XL/2 | 1152 | 28 | 16 | 675M | ~16GB |
| DiT-L/2 | 1024 | 24 | 16 | 457M | ~11GB |
| DiT-B/2 | 768 | 12 | 12 | 112M | ~3GB |
| DiT-S/2 | 384 | 12 | 6 | 39M | ~1GB |

## Quick Start

### Training a DiT-XL/2 Model (MindSpore)

```bash
# 1. Navigate to the DiT directory
cd examples/dit

# 2. Run training with default configuration
bash scripts/train_dit.sh

# 3. Or customize with command-line arguments
python scripts/train.py \
    --config configs/train_dit_xl_2.yaml \
    data.dataset_path=/path/to/imagenet \
    train.output_path=./output/dit_xl2
```

### Training with PyTorch (Alternative)

If MindSpore is not available, you can use the PyTorch-based DiT implementation at the root `DiT/` directory:

```bash
# Navigate to DiT directory
cd DiT

# Run training on CPU
RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=29500 \
python3 train.py \
    --data-path /path/to/imagenet \
    --results-dir ./output/dit_xl2 \
    --model DiT-XL/2 \
    --device cpu \
    --num-workers 4 \
    --epochs 100 \
    --global-batch-size 256

# Or run on GPU (if available)
python3 train.py \
    --data-path /path/to/imagenet \
    --results-dir ./output/dit_xl2 \
    --model DiT-XL/2 \
    --device cuda \
    --num-workers 4
```

### Generating Images

```bash
# Generate images from a trained checkpoint
python scripts/generate.py \
    --checkpoint ./output/dit_xl2/checkpoints/dit_final.ckpt \
    --output_dir ./generated_images \
    --num_images 4 \
    --num_inference_steps 50 \
    --guidance_scale 4.0

# Generate specific classes
python scripts/generate.py \
    --checkpoint ./output/dit_xl2/checkpoints/dit_final.ckpt \
    --class_labels "tench,goldfish,hen" \
    --num_images 2
```

## Project Structure

```
examples/dit/
├── configs/                    # Training configurations
│   └── train_dit_xl_2.yaml    # DiT-XL/2 default config
├── dit/                       # Main package
│   ├── __init__.py
│   ├── dataset.py             # ImageNet dataset implementation
│   └── train_pipeline.py      # Training pipeline with loss computation
├── scripts/                   # Executable scripts
│   ├── train.py               # Training entry point
│   ├── train_dit.sh           # Training launcher
│   ├── generate.py            # Inference entry point
│   └── generate.sh            # Inference launcher
└── README.md                  # This file
```

## Installation

### Requirements

- MindSpore >= 2.0 (for MindSpore version)
- PyTorch >= 2.0 (for PyTorch version)
- Python >= 3.9
- CUDA >= 11.7 (for GPU training)
- Ascend NPU (for Ascend training)

### Install Dependencies (MindSpore)

```bash
# Install MindSpore (CPU/GPU/Ascend)
pip install mindspore             # CPU
pip install mindspore-gpu         # GPU
pip install mindspore-ascend      # Ascend NPU

# Install MindONE in development mode
cd /home/ubuntu/xql/mindone
pip install -e ".[training]"
```

### Install Dependencies (PyTorch)

```bash
# Install PyTorch with CUDA
pip install torch torchvision

# Or install CPU version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Training

### Preparing the Dataset

DiT requires ImageNet dataset for training. Download from [Image-Net](https://image-net.org/) and organize as follows:

```
imagenet/
├── train/
│   ├── n01440764/
│   │   ├── ILSVRC2012_val_00000211.JPEG
│   │   └── ...
│   ├── n01443537/
│   │   └── ...
│   └── ...
└── val/
    ├── n01440764/
    │   └── ...
    └── ...
```

#### Using ImageNet Mini (Smaller Dataset for Testing)

For quick testing or development, you can use [ImageNet Mini](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000):

```bash
# Download via kagglehub
pip install kagglehub
python -c "import kagglehub; kagglehub.dataset_download('ifigotin/imagenetmini-1000')"

# The dataset will be downloaded to ~/.cache/kagglehub/datasets/ifigotin/imagenetmini-1000/
# Use: data.dataset_path=/path/to/imagenetmini/imagenet-mini
```

### Training Configuration

Create or modify the configuration file (`configs/train_dit_xl_2.yaml`):

```yaml
# Model configuration
model:
  model_type: "DiT-XL/2"
  vae_model: "stabilityai/sd-vae-ft-mse"
  latent_channels: 4
  patch_size: 2
  hidden_size: 1152
  num_layers: 28
  num_heads: 16
  head_dim: 72
  class_dropout_prob: 0.1
  enable_gradient_checkpointing: false

# Data configuration
data:
  dataset_path: "/path/to/imagenet"
  latent_path: null  # Pre-computed latents path for faster training
  image_size: 256
  num_classes: 1000
  batch_size: 16
  num_workers: 8
  shuffle: true
  drop_remainder: true
  prefetch_size: 16

# Training configuration
train:
  output_path: "./output/dit_xl2"
  resume_ckpt: null
  num_epochs: 100
  gradient_accumulation: 1
  max_grad_norm: 1.0
  
  # Optimizer settings
  optimizer:
    type: "adamw"
    lr: 1e-4
    weight_decay: 0.0
    beta1: 0.9
    beta2: 0.999
  
  # Learning rate scheduler
  scheduler:
    type: "cosine"
    warmup_steps: 1000
    min_lr: 1e-6
  
  # EMA settings
  ema:
    enable: true
    decay: 0.9999
    update_interval: 1
  
  # Logging and checkpointing
  log_interval: 50
  save_checkpoint_steps: 5000
  save_checkpoint_total_limit: 5

# Environment settings
env:
  mode: 0  # 0: PYNATIVE_MODE, 1: GRAPH_MODE
  device: "Ascend"  # "Ascend", "GPU", "CPU"
  distributed: false
  seed: 42
  dtype: "fp32"

# VAE settings
vae:
  tiling: false
  scaling_factor: 0.18215

# Scheduler settings
scheduler:
  type: "ddpm"
  beta_start: 0.0001
  beta_end: 0.02
  beta_schedule: "linear"
  num_train_timesteps: 1000
  clip_sample: false
```

### Running Training

#### Single Device Training

```bash
python scripts/train.py \
    --config configs/train_dit_xl_2.yaml \
    data.dataset_path=/path/to/imagenet \
    train.output_path=./output/dit_xl2
```

#### Multi-Device Training

```bash
# Set up distributed environment
export RANK_TABLE_FILE=./rank_table.json
export RANK_SIZE=8

# Run distributed training
python scripts/train.py \
    --config configs/train_dit_xl_2.yaml \
    data.dataset_path=/path/to/imagenet \
    env.distributed=True \
    train.output_path=./output/dit_xl2_dist
```

#### Resume Training from Checkpoint

```bash
python scripts/train.py \
    --config configs/train_dit_xl_2.yaml \
    data.dataset_path=/path/to/imagenet \
    train.resume_ckpt=./output/dit_xl2/checkpoints/dit_5000.ckpt
```

### Training with Pre-computed Latents

For faster training, you can pre-compute VAE latents:

```bash
# Generate latents first
python scripts/precompute_latents.py \
    --dataset_path /path/to/imagenet \
    --output_path ./latents \
    --vae_model stabilityai/sd-vae-ft-mse

# Train with latents
python scripts/train.py \
    --config configs/train_dit_xl_2.yaml \
    data.dataset_path=/path/to/imagenet \
    data.latent_path=./latents
```

## Inference

### Basic Usage

```bash
python scripts/generate.py \
    --checkpoint ./output/dit_xl2/checkpoints/dit_final.ckpt \
    --output_dir ./generated_images \
    --num_images 4 \
    --num_inference_steps 50 \
    --guidance_scale 4.0
```

### Generation Options

```bash
# Specify classes (ImageNet labels)
python scripts/generate.py \
    --checkpoint ./output/dit_xl2/checkpoints/dit_final.ckpt \
    --class_labels "tench,goldfish,hen,ostrich" \
    --num_images 2

# Random classes
python scripts/generate.py \
    --checkpoint ./output/dit_xl2/checkpoints/dit_final.ckpt \
    --class_labels "random" \
    --num_images 4

# Different schedulers
python scripts/generate.py \
    --checkpoint ./output/dit_xl2/checkpoints/dit_final.ckpt \
    --scheduler "ddim" \
    --eta 0.0

# Mixed precision for faster generation
python scripts/generate.py \
    --checkpoint ./output/dit_xl2/checkpoints/dit_final.ckpt \
    --dtype "fp16"
```

### Python API for Inference

```python
import mindspore as ms
from mindone.diffusers import DiTPipeline, DiTTransformer2DModel

# Load pipeline from checkpoint
pipeline = DiTPipeline.from_pretrained("./output/dit_xl2")

# Generate single class
output = pipeline(
    class_labels=[0],  # ImageNet class: tench
    num_inference_steps=50,
    guidance_scale=4.0
)

# Save generated images
for idx, image in enumerate(output.images):
    image.save(f"generated_{idx}.png")

# Generate multiple classes
output = pipeline(
    class_labels=[0, 1, 2],  # tench, goldfish, great_white_shark
    num_inference_steps=50,
    guidance_scale=4.0,
    num_images_per_prompt=2
)
```

### Supported Schedulers

- **DDPM**: High-quality but slower
- **DDIM**: Faster sampling with similar quality
- **DPM-Solver**: Fast sampler with good quality

## Model Architecture

### DiTTransformer2DModel

```
Input (Image) → PatchEmbed → [DiTBlock × N] → Norm → Unpatchify → Output (Noise Prediction)
                         ↑
                    Timestep + Class Embedding (AdaLN)
```

#### Key Components

1. **PatchEmbed**: Converts images into flattened patches
2. **DiTBlock**: Transformer block with adaLN-Zero conditioning
3. **AdaLN-Zero**: Adaptive layer normalization with zero-initialized modulation

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_attention_heads` | int | 16 | Number of attention heads |
| `attention_head_dim` | int | 72 | Dimension of each attention head |
| `in_channels` | int | 4 | Input latent channels |
| `out_channels` | int | None | Output channels (defaults to in_channels) |
| `num_layers` | int | 28 | Number of transformer layers |
| `dropout` | float | 0.0 | Dropout probability |
| `patch_size` | int | 2 | Size of image patches |
| `norm_type` | str | "ada_norm_zero" | Normalization type |
| `num_embeds_ada_norm` | int | 1000 | Max timesteps for conditioning |

## Configuration

### Model Configuration

```yaml
model:
  model_type: "DiT-XL/2"           # Model variant
  vae_model: "stabilityai/sd-vae-ft-mse"
  latent_channels: 4
  patch_size: 2
  hidden_size: 1152
  num_layers: 28
  num_heads: 16
  head_dim: 72
  class_dropout_prob: 0.1          # CFG dropout probability
  enable_gradient_checkpointing: false
```

### Training Configuration

```yaml
train:
  num_epochs: 100
  gradient_accumulation: 1         # Accumulate gradients over N steps
  max_grad_norm: 1.0               # Gradient clipping
  
  optimizer:
    type: "adamw"
    lr: 1e-4                       # Learning rate
    weight_decay: 0.0
    beta1: 0.9
    beta_2: 0.999
  
  scheduler:
    type: "cosine"                 # or "constant", "linear"
    warmup_steps: 1000
    min_lr: 1e-6
  
  ema:
    enable: true
    decay: 0.9999                  # EMA decay rate
    update_interval: 1             # Update EMA every N steps
```

### Data Configuration

```yaml
data:
  dataset_path: "/path/to/imagenet"
  latent_path: null               # Pre-computed latents (faster training)
  image_size: 256
  num_classes: 1000
  batch_size: 16
  num_workers: 8
  shuffle: true
  drop_remainder: true            # Drop incomplete batches
  prefetch_size: 16
```

## API Reference

### Training Pipeline

#### DiTDiffusionWithLoss

```python
from dit.train_pipeline import DiTDiffusionWithLoss

class DiTDiffusionWithLoss(nn.Cell):
    def __init__(
        self,
        network: nn.Cell,
        vae: Optional[nn.Cell] = None,
        noise_scheduler: Optional[Any] = None,
        latent_scaling_factor: float = 0.18215,
        class_dropout_prob: float = 0.1,
        latent_channels: int = 4,
    )
    
    def construct(
        self,
        images: Tensor,
        class_labels: Tensor,
        latent: Optional[Tensor] = None,
        guidance_scale: Optional[float] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Forward pass for training."""
```

**Example Usage:**

```python
import mindspore as ms
from mindone.diffusers import DiTTransformer2DModel, AutoencoderKL, DDPMScheduler
from dit.train_pipeline import DiTDiffusionWithLoss

# Initialize models
dit = DiTTransformer2DModel(
    in_channels=4,
    num_attention_heads=16,
    attention_head_dim=72,
    num_layers=28,
)

vae = AutoencoderKL.from_pretrained("vae_path")
scheduler = DDPMScheduler(num_train_timesteps=1000)

# Create training pipeline
train_pipeline = DiTDiffusionWithLoss(
    network=dit,
    vae=vae,
    noise_scheduler=scheduler,
    class_dropout_prob=0.1
)

# Training step
loss, logs = train_pipeline(images, class_labels)
```

### Inference Pipeline

#### DiTPipeline

```python
from mindone.diffusers import DiTPipeline

pipeline = DiTPipeline.from_pretrained("model_path")

output = pipeline(
    class_labels: List[int],
    num_inference_steps: int = 50,
    guidance_scale: float = 4.0,
    num_images_per_prompt: int = 1,
    height: int = 256,
    width: int = 256,
    eta: float = 0.0,
    generator: Optional[np.random.Generator] = None,
)
```

### Transformer Model

#### DiTTransformer2DModel

```python
from mindone.diffusers import DiTTransformer2DModel

model = DiTTransformer2DModel(
    num_attention_heads: int = 16,
    attention_head_dim: int = 72,
    in_channels: int = 4,
    out_channels: Optional[int] = None,
    num_layers: int = 28,
    dropout: float = 0.0,
    norm_num_groups: int = 32,
    attention_bias: bool = True,
    sample_size: int = 32,
    patch_size: int = 2,
    activation_fn: str = "gelu-approximate",
    num_embeds_ada_norm: Optional[int] = 1000,
    upcast_attention: bool = False,
    norm_type: str = "ada_norm_zero",
    norm_elementwise_affine: bool = False,
    norm_eps: float = 1e-5,
)
```

## Advanced Features

### Classifier-Free Guidance

DiT supports classifier-free guidance for improved generation quality:

```python
# Training with CFG
train_pipeline = DiTDiffusionWithLoss(
    network=dit,
    vae=vae,
    noise_scheduler=scheduler,
    class_dropout_prob=0.1  # 10% dropout for null class
)

# Inference with CFG
output = pipeline(
    class_labels=[0],
    guidance_scale=4.0  # Higher = more adherence to class
)
```

### Exponential Moving Average (EMA)

EMA improves training stability:

```yaml
train:
  ema:
    enable: true
    decay: 0.9999  # Exponential decay rate
    update_interval: 1  # Update frequency
```

### Gradient Checkpointing

Reduces memory usage during training:

```yaml
model:
  enable_gradient_checkpointing: true
```

### Mixed Precision Training

```yaml
env:
  dtype: "fp16"  # or "bf16"
```

### Flow Matching Loss

Alternative to DDPM loss:

```python
from dit.train_pipeline import FlowMatchingLoss

train_pipeline = FlowMatchingLoss(
    network=dit,
    vae=vae,
)
```

## Performance Tips

### Training Speed

1. **Use Pre-computed Latents**: Save VAE encoding time
   ```bash
   python scripts/precompute_latents.py --dataset_path /path/to/data
   ```

2. **Enable Gradient Checkpointing**: Trade compute for memory
   ```yaml
   model:
     enable_gradient_checkpointing: true
   ```

3. **Increase Batch Size**: More samples per step
   ```bash
   python scripts/train.py data.batch_size=32
   ```

4. **Use Distributed Training**: Scale across devices
   ```bash
   env.distributed=True RANK_SIZE=8
   ```

### Inference Speed

1. **Use DDIM Scheduler**: Faster than DDPM
   ```bash
   --scheduler "ddim" --num_inference_steps 30
   ```

2. **Mixed Precision**: Use FP16
   ```bash
   --dtype "fp16"
   ```

3. **Reduce Steps**: Fewer denoising steps
   ```bash
   --num_inference_steps 20
   ```

### Memory Optimization

1. **Gradient Accumulation**: Simulate larger batches
   ```yaml
   train:
     gradient_accumulation: 4
     batch_size: 4  # Actual batch size
   ```

2. **Enable Offloading**: Offload models to CPU
   ```python
   pipeline.enable_model_cpu_offload()
   ```

## Troubleshooting

### Common Issues

#### MindSpore CPU: Missing Kernel Operations

**Problem:** MindSpore CPU version may be missing some kernel operations (e.g., Conv2D, Arange) required for DiT training.

**Solutions:**
```bash
# Option 1: Use GPU version of MindSpore
pip install mindspore-gpu

# Option 2: Use Ascend NPU version
pip install mindspore-ascend

# Option 3: Use PyTorch version instead
cd DiT
python train.py --device cpu ...
```

#### Out of Memory (OOM) During Training

**Solution:**
```bash
# Reduce batch size
python scripts/train.py data.batch_size=8

# Enable gradient checkpointing
python scripts/train.py model.enable_gradient_checkpointing=true

# Use mixed precision
python scripts/train.py env.dtype="fp16"
```

#### Poor Generation Quality

**Possible Causes:**
- Training not converged (need more epochs)
- Learning rate too high/low
- Batch size too small

**Solutions:**
```bash
# Train longer
python scripts/train.py train.num_epochs=200

# Adjust learning rate
python scripts/train.py train.optimizer.lr=5e-5

# Increase batch size
python scripts/train.py data.batch_size=32
```

#### Checkpoint Loading Fails

**Solution:**
```python
# Verify checkpoint path
import os
print(os.path.exists("./checkpoint.ckpt"))

# Check file is valid MindSpore checkpoint
import mindspore as ms
param_dict = ms.load_checkpoint("./checkpoint.ckpt")
print(f"Parameters: {len(param_dict)}")
```

#### Distributed Training Issues

**Solution:**
```bash
# Verify rank table
cat ./rank_table.json

# Check device connectivity
mpirun --allow-run-as-root -n 8 python scripts/train.py
```

### Getting Help

- **Documentation**: Check `mindone/diffusers/models/transformers/dit_transformer_2d.py`
- **Tests**: See `tests/diffusers_tests/pipelines/dit/test_dit.py`
- **Examples**: Browse `examples/accelerated_dit_pipelines/`

### Debug Mode

Enable verbose logging:

```bash
python scripts/train.py \
    --config configs/train_dit_xl_2.yaml \
    train.log_interval=10
```

## Citation

If you use DiT in your research, please cite:

```bibtex
@article{peebles2022dit,
  title={Scalable Diffusion Models with Transformers},
  author={Peebles, William and Xie, Saining},
  journal={arXiv preprint arXiv:2212.09748},
  year={2022}
}
```

## License

This code is adapted from [Hugging Face Diffusers](https://github.com/huggingface/diffusers) with modifications for MindSpore compatibility.

Licensed under the Apache License, Version 2.0.

## Acknowledgments

- DiT paper: [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
- Hugging Face Diffusers: [https://github.com/huggingface/diffusers](https://github.com/huggingface/diffusers)
- MindSpore: [https://www.mindspore.cn/](https://www.mindspore.cn/)
