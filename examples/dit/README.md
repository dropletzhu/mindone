# DiT (Diffusion Transformer) Training and Inference

基于 MindSpore 框架实现的 DiT 模型，用于 ImageNet 256×256 类别条件图像生成。

## 目录

- [概述](#概述)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [环境安装](#环境安装)
- [训练](#训练)
- [推理](#推理)
- [模型架构](#模型架构)
- [命令行参数](#命令行参数)
- [常见问题](#常见问题)

## 概述

DiT (Diffusion Transformer) 是一种使用 Transformer 主干网络替代传统 UNet 架构的扩散模型。

### 模型变体

| 模型 | 隐藏层大小 | 层数 | 注意力头数 | 参数数量 | 显存占用 (FP32) |
|------|-----------|------|-----------|---------|----------------|
| DiT-XL/2 | 1152 | 28 | 16 | 675M | ~16GB |
| DiT-L/2 | 1024 | 24 | 16 | 457M | ~11GB |
| DiT-B/2 | 768 | 12 | 12 | 112M | ~3GB |
| DiT-S/2 | 384 | 12 | 6 | 39M | ~1GB |

## 快速开始

### 训练

```bash
cd examples/dit

# 单卡训练
python scripts/train.py \
    --config configs/train_dit_xl_2.yaml \
    data.dataset_path=/path/to/imagenet \
    train.output_path=./output/dit_xl2

# 多卡训练
export RANK_TABLE_FILE=./rank_table_8p.json
export RANK_SIZE=8

python scripts/train.py \
    --config configs/train_dit_xl_2.yaml \
    data.dataset_path=/path/to/imagenet \
    env.distributed=True \
    train.output_path=./output/dit_xl2_dist
```

### 推理

```bash
# 使用训练好的检查点生成图像
python scripts/generate.py \
    --checkpoint ./output/dit_xl2/checkpoints/dit_final.ckpt \
    --output_dir ./generated_images \
    --num_images 4 \
    --num_inference_steps 50 \
    --guidance_scale 4.0

# 指定类别生成
python scripts/generate.py \
    --checkpoint ./output/dit_xl2/checkpoints/dit_final.ckpt \
    --class_labels "tench,goldfish,hen" \
    --num_images 3
```

## 项目结构

```
examples/dit/
├── configs/
│   └── train_dit_xl_2.yaml    # DiT-XL/2 训练配置
├── dit/
│   ├── __init__.py
│   ├── dataset.py             # ImageNet 数据集实现
│   └── train_pipeline.py      # 训练 pipeline
├── scripts/
│   ├── train.py               # 训练入口
│   ├── generate.py            # 推理入口
│   └── train_dit.sh          # 训练启动脚本
└── README.md
```

## 环境安装

### 依赖要求

- MindSpore >= 2.0
- Python >= 3.9
- Ascend NPU 驱动和固件

### 安装步骤

```bash
# 1. 安装 MindSpore (Ascend NPU 版本)
pip install mindspore-ascend

# 2. 安装 MindONE
cd /path/to/mindone
pip install -e ".[training]"

### 模型文件

VAE 模型和预训练检查点建议下载到本地：

```bash
# 使用 ModelScope 下载 VAE 模型
python -c "from modelscope.hub.snapshot_download import snapshot_download; snapshot_download('stabilityai/sd-vae-ft-mse', cache_dir='./models')"

# 推荐存放目录
# /home/ma-user/work/temp/sd-vae-ft-mse
# /home/ma-user/work/temp/imagenet
```

## 训练

### 数据集准备

DiT 训练需要 ImageNet 数据集。下载后按以下结构组织：

```
imagenet/
├── train/
│   ├── n01440764/
│   │   ├── ILSVRC2012_val_00000211.JPEG
│   │   └── ...
│   └── ...
└── val/
    └── ...
```

### 配置文件

`configs/train_dit_xl_2.yaml`:

```yaml
# 环境设置
env:
  mode: 0                    # 0: PYNATIVE_MODE, 1: GRAPH_MODE
  device: "Ascend"           # Ascend NPU 设备
  distributed: false         # 是否分布式训练
  seed: 42
  dtype: "fp32"             # fp32, fp16, bf16

# 数据设置
data:
  dataset_path: "/path/to/imagenet"
  latent_path: null
  image_size: 256
  batch_size: 16
  num_workers: 8

# 模型设置
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

# 训练设置
train:
  output_path: "./output/dit_xl2"
  num_epochs: 100
  optimizer:
    type: "adamw"
    lr: 1e-4
  scheduler:
    type: "cosine"
    warmup_steps: 1000
    min_lr: 1e-6
  ema:
    enable: true
    decay: 0.9999
  log_interval: 50
  save_checkpoint_steps: 5000

# VAE 设置
vae:
  tiling: false
  scaling_factor: 0.18215

# 调度器设置
scheduler:
  type: "ddpm"
  beta_start: 0.0001
  beta_end: 0.02
  num_train_timesteps: 1000
```

### 训练命令

```bash
# 基础训练
python scripts/train.py \
    --config configs/train_dit_xl_2.yaml \
    data.dataset_path=/path/to/imagenet \
    train.output_path=./output/dit_xl2

# 从检查点恢复训练
python scripts/train.py \
    --config configs/train_dit_xl_2.yaml \
    data.dataset_path=/path/to/imagenet \
    train.resume_ckpt=./output/dit_xl2/checkpoints/dit_5000.ckpt

# 分布式训练
export RANK_TABLE_FILE=./rank_table_8p.json
export RANK_SIZE=8
python scripts/train.py \
    --config configs/train_dit_xl_2.yaml \
    data.dataset_path=/path/to/imagenet \
    env.distributed=True \
    train.output_path=./output/dit_xl2_dist
```

## 推理

### 推理命令

```bash
# 基础推理 (使用本地 VAE 模型)
python scripts/generate.py \
    --checkpoint ./output/dit_xl2/checkpoints/dit_final.ckpt \
    --vae_model /home/ma-user/work/temp/sd-vae-ft-mse \
    --output_dir ./generated_images \
    --num_images 4 \
    --num_inference_steps 50 \
    --guidance_scale 4.0

# 指定类别 (ImageNet 标签)
python scripts/generate.py \
    --checkpoint ./output/dit_xl2/checkpoints/dit_final.ckpt \
    --vae_model /home/ma-user/work/temp/sd-vae-ft-mse \
    --class_labels "tench,goldfish,hen,ostrich" \
    --num_images 4

# 使用 DDIM 调度器加速采样
python scripts/generate.py \
    --checkpoint ./output/dit_xl2/checkpoints/dit_final.ckpt \
    --vae_model /home/ma-user/work/temp/sd-vae-ft-mse \
    --scheduler ddim \
    --num_inference_steps 20 \
    --output_dir ./generated_images

# 混合精度推理
python scripts/generate.py \
    --checkpoint ./output/dit_xl2/checkpoints/dit_final.ckpt \
    --vae_model /home/ma-user/work/temp/sd-vae-ft-mse \
    --dtype fp16 \
    --output_dir ./generated_images
```

### Python API 推理

```python
import mindspore as ms
from mindone.diffusers import DiTPipeline

# 加载 pipeline
pipeline = DiTPipeline.from_pretrained("./output/dit_xl2")

# 生成图像
output = pipeline(
    class_labels=[0, 1, 2],  # ImageNet 类别
    num_inference_steps=50,
    guidance_scale=4.0,
    return_dict=True,
)

# 保存图像
for idx, image in enumerate(output.images):
    image.save(f"generated_{idx}.png")
```

### 支持的调度器

| 调度器 | 特点 |
|-------|------|
| DDPM | 质量最高，速度最慢 |
| DDIM | 质量接近 DDPM，速度更快 |
| DPM-Solver | 快速采样 |

## 模型架构

```
Input (Image) → PatchEmbed → [DiTBlock × N] → Norm → Unpatchify → Output
                              ↑
                     Timestep + Class Embedding (AdaLN)
```

### 核心组件

1. **PatchEmbed**: 将图像转换为 patch tokens
2. **DiTBlock**: 带 adaLN-Zero 条件化的 Transformer 块
3. **AdaLN-Zero**: 零初始化的自适应层归一化

## 命令行参数

### 训练参数 (train.py)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--config` | str | - | 配置文件路径 |
| `--data.dataset_path` | str | - | ImageNet 数据集路径 (必需) |
| `--train.output_path` | str | - | 输出路径 (必需) |
| `--model.model_type` | str | DiT-XL/2 | 模型类型 |
| `--model.latent_channels` | int | 4 | 潜在通道数 |
| `--model.num_layers` | int | 28 | Transformer 层数 |
| `--model.num_heads` | int | 16 | 注意力头数 |
| `--model.head_dim` | int | 72 | 注意力头维度 |
| `--data.batch_size` | int | 16 | 批次大小 |
| `--data.image_size` | int | 256 | 图像大小 |
| `--train.num_epochs` | int | 100 | 训练轮数 |
| `--train.optimizer.lr` | float | 1e-4 | 学习率 |
| `--train.ema.enable` | bool | true | 是否启用 EMA |
| `--train.ema.decay` | float | 0.9999 | EMA 衰减率 |
| `--env.device` | str | Ascend | 设备类型 |
| `--env.dtype` | str | fp32 | 数据类型 (fp32/fp16/bf16) |
| `--env.distributed` | bool | false | 分布式训练 |
| `--scheduler.type` | str | ddpm | 调度器类型 |

### 推理参数 (generate.py)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--checkpoint` | str | - | 检查点路径 (必需) |
| `--output_dir` | str | ./generated_images | 输出目录 |
| `--num_images` | int | 4 | 生成图像数量 |
| `--num_inference_steps` | int | 50 | 推理步数 |
| `--guidance_scale` | float | 4.0 | CFG 引导强度 |
| `--class_labels` | str | - | 类别标签 (逗号分隔) |
| `--scheduler` | str | ddim | 调度器类型 |
| `--dtype` | str | fp32 | 数据类型 |
| `--seed` | int | 42 | 随机种子 |
| `--image_size` | int | 256 | 输出图像大小 |
| `--vae_model` | str | stabilityai/sd-vae-ft-mse | VAE 模型路径或 HuggingFace ID |

## 常见问题

### 显存不足 (OOM)

```bash
# 减小批次大小
python scripts/train.py data.batch_size=8

# 启用梯度检查点
python scripts/train.py model.enable_gradient_checkpointing=true

# 使用混合精度
python scripts/train.py env.dtype="fp16"
```

### 分布式训练失败

```bash
# 检查 rank table
cat ./rank_table.json

# 验证设备连接
mpirun --allow-run-as-root -n 8 python scripts/train.py
```

### 生成质量问题

```bash
# 增加推理步数
python scripts/generate.py --num_inference_steps 100

# 调整引导强度
python scripts/generate.py --guidance_scale 4.0

# 使用更大模型
python scripts/train.py model.model_type="DiT-XL/2"
```

## 引用

```bibtex
@article{peebles2022dit,
  title={Scalable Diffusion Models with Transformers},
  author={Peebles, William and Xie, Saining},
  journal={arXiv preprint arXiv:2212.09748},
  year={2022}
}
```

## 许可证

本代码改编自 [Hugging Face Diffusers](https://github.com/huggingface/diffusers)，遵守 Apache License 2.0。
