# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
#
# This code is adapted from https://github.com/huggingface/diffusers
# with modifications to run diffusers on mindspore.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Training script for DiT (Diffusion Transformer) model."""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from jsonargparse import ActionConfigFile, ArgumentParser

import mindspore as ms
import mindspore.dataset as ds
from mindspore import GRAPH_MODE, Model, get_context, nn, set_context, set_seed
from mindspore.communication.management import get_group_size, get_rank

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from mindone.data import create_dataloader
from mindone.trainers import create_optimizer, create_scheduler
from mindone.trainers.callback import EvalSaveCallback, OverflowMonitor, StopAtStepCallback
from mindone.utils import EMA, count_params, init_env, set_logger
from mindone.utils.seed import set_random_seed

from dit.dataset import ImageNetDataset, ImageNetLatentDataset
from dit.train_pipeline import DiTDiffusionWithLoss

logger = logging.getLogger(__name__)

parser = ArgumentParser()
parser.add_argument("--config", action=ActionConfigFile)


def add_model_args(parser: ArgumentParser) -> ArgumentParser:
    """Add model configuration arguments."""
    model_group = parser.add_argument_group(title="model")
    model_group.add_argument("--model.model_type", type=str, default="DiT-XL/2", help="Model variant")
    model_group.add_argument("--model.vae_model", type=str, default="stabilityai/sd-vae-ft-mse", help="VAE model path")
    model_group.add_argument("--model.latent_channels", type=int, default=4)
    model_group.add_argument("--model.patch_size", type=int, default=2)
    model_group.add_argument("--model.hidden_size", type=int, default=1152)
    model_group.add_argument("--model.num_layers", type=int, default=28)
    model_group.add_argument("--model.num_heads", type=int, default=16)
    model_group.add_argument("--model.head_dim", type=int, default=72)
    model_group.add_argument("--model.class_dropout_prob", type=float, default=0.1)
    model_group.add_argument("--model.enable_gradient_checkpointing", type=bool, default=False)
    return parser


def add_data_args(parser: ArgumentParser) -> ArgumentParser:
    """Add data configuration arguments."""
    data_group = parser.add_argument_group(title="data")
    data_group.add_argument("--data.dataset_path", type=str, required=True, help="ImageNet dataset path")
    data_group.add_argument("--data.latent_path", type=str, default=None, help="Pre-computed latents path")
    data_group.add_argument("--data.image_size", type=int, default=256)
    data_group.add_argument("--data.num_classes", type=int, default=1000)
    data_group.add_argument("--data.batch_size", type=int, default=16)
    data_group.add_argument("--data.num_workers", type=int, default=8)
    data_group.add_argument("--data.shuffle", type=bool, default=True)
    data_group.add_argument("--data.drop_remainder", type=bool, default=True)
    data_group.add_argument("--data.prefetch_size", type=int, default=16)
    return parser


def add_train_args(parser: ArgumentParser) -> ArgumentParser:
    """Add training configuration arguments."""
    train_group = parser.add_argument_group(title="train")
    train_group.add_argument("--train.output_path", type=str, required=True)
    train_group.add_argument("--train.resume_ckpt", type=str, default=None)
    train_group.add_argument("--train.num_epochs", type=int, default=100)
    train_group.add_argument("--train.gradient_accumulation", type=int, default=1)
    train_group.add_argument("--train.max_grad_norm", type=float, default=1.0)
    train_group.add_argument("--train.optimizer.type", type=str, default="adamw")
    train_group.add_argument("--train.optimizer.lr", type=float, default=1e-4)
    train_group.add_argument("--train.optimizer.weight_decay", type=float, default=0.0)
    train_group.add_argument("--train.optimizer.beta1", type=float, default=0.9)
    train_group.add_argument("--train.optimizer.beta2", type=float, default=0.999)
    train_group.add_argument("--train.scheduler.type", type=str, default="cosine")
    train_group.add_argument("--train.scheduler.warmup_steps", type=int, default=1000)
    train_group.add_argument("--train.scheduler.min_lr", type=float, default=1e-6)
    train_group.add_argument("--train.ema.enable", type=bool, default=True)
    train_group.add_argument("--train.ema.decay", type=float, default=0.9999)
    train_group.add_argument("--train.ema.update_interval", type=int, default=1)
    train_group.add_argument("--train.log_interval", type=int, default=50)
    train_group.add_argument("--train.save_checkpoint_steps", type=int, default=5000)
    train_group.add_argument("--train.save_checkpoint_total_limit", type=int, default=5)
    return parser


def add_env_args(parser: ArgumentParser) -> ArgumentParser:
    """Add environment configuration arguments."""
    env_group = parser.add_argument_group(title="env")
    env_group.add_argument("--env.mode", type=int, default=0)
    env_group.add_argument("--env.device", type=str, default="Ascend")
    env_group.add_argument("--env.distributed", type=bool, default=False)
    env_group.add_argument("--env.seed", type=int, default=42)
    env_group.add_argument("--env.dtype", type=str, default="fp32")
    return parser


def add_vae_args(parser: ArgumentParser) -> ArgumentParser:
    """Add VAE configuration arguments."""
    vae_group = parser.add_argument_group(title="vae")
    vae_group.add_argument("--vae.tiling", type=bool, default=False)
    vae_group.add_argument("--vae.scaling_factor", type=float, default=0.18215)
    return parser


def add_scheduler_args(parser: ArgumentParser) -> ArgumentParser:
    """Add scheduler configuration arguments."""
    sched_group = parser.add_argument_group(title="scheduler")
    sched_group.add_argument("--scheduler.type", type=str, default="ddpm")
    sched_group.add_argument("--scheduler.beta_start", type=float, default=0.0001)
    sched_group.add_argument("--scheduler.beta_end", type=float, default=0.02)
    sched_group.add_argument("--scheduler.beta_schedule", type=str, default="linear")
    sched_group.add_argument("--scheduler.num_train_timesteps", type=int, default=1000)
    sched_group.add_argument("--scheduler.clip_sample", type=bool, default=False)
    return parser


parser = add_env_args(parser)
parser = add_data_args(parser)
parser = add_model_args(parser)
parser = add_train_args(parser)
parser = add_vae_args(parser)
parser = add_scheduler_args(parser)


def create_dit_model(
    model_type: str = "DiT-XL/2",
    latent_channels: int = 4,
    patch_size: int = 2,
    hidden_size: int = 1152,
    num_layers: int = 28,
    num_heads: int = 16,
    head_dim: int = 72,
    sample_size: int = 32,
    num_embeds_ada_norm: int = 1000,
    **kwargs,
) -> nn.Cell:
    """Create DiT model based on configuration.

    Args:
        model_type: Model variant (DiT-XL/2, DiT-L/2, DiT-B/2, DiT-S/2)
        latent_channels: Number of latent channels
        patch_size: Patch size for patch embedding
        hidden_size: Hidden dimension size
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        head_dim: Dimension per head
        sample_size: Sample size (latent grid size)
        num_embeds_ada_norm: Number of timestep embeddings

    Returns:
        DiT transformer model
    """
    from mindone.diffusers import DiTTransformer2DModel

    model = DiTTransformer2DModel(
        in_channels=latent_channels,
        out_channels=latent_channels,
        patch_size=patch_size,
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        num_layers=num_layers,
        sample_size=sample_size,
        num_embeds_ada_norm=num_embeds_ada_norm,
        activation_fn="gelu-approximate",
    )

    return model


def create_scheduler_fn(
    scheduler_type: str = "ddpm",
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    beta_schedule: str = "linear",
    num_train_timesteps: int = 1000,
    clip_sample: bool = False,
) -> Any:
    """Create noise scheduler for training.

    Args:
        scheduler_type: Type of scheduler (ddpm, ddim)
        beta_start: Starting beta value
        beta_end: Ending beta value
        beta_schedule: Beta schedule type
        num_train_timesteps: Number of training timesteps
        clip_sample: Whether to clip samples

    Returns:
        Noise scheduler
    """
    from mindone.diffusers import DDPMScheduler, DDIMScheduler

    if scheduler_type.lower() == "ddpm":
        scheduler = DDPMScheduler(
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            num_train_timesteps=num_train_timesteps,
            clip_sample=clip_sample,
        )
    elif scheduler_type.lower() == "ddim":
        scheduler = DDIMScheduler(
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            num_train_timesteps=num_train_timesteps,
            clip_sample=clip_sample,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler


def create_vae(vae_model: str, tiling: bool = False) -> nn.Cell:
    """Create VAE model.

    Args:
        vae_model: VAE model path or identifier
        tiling: Whether to enable VAE tiling

    Returns:
        VAE model
    """
    from mindone.diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(vae_model)
    if tiling:
        vae.enable_tiling()
    else:
        vae.disable_tiling()

    return vae


def initialize_dataset(
    dataset_path: str,
    latent_path: Optional[str],
    image_size: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    drop_remainder: bool,
    prefetch_size: int,
    latent_channels: int = 4,
    sample_size: int = 32,
    is_distributed: bool = False,
    rank_id: int = 0,
    device_num: int = 1,
) -> tuple:
    """Initialize dataset and dataloader.

    Args:
        dataset_path: Path to ImageNet dataset
        latent_path: Path to pre-computed latents (optional)
        image_size: Image size for processing
        batch_size: Batch size
        num_workers: Number of workers
        shuffle: Whether to shuffle
        drop_remainder: Whether to drop remainder
        prefetch_size: Prefetch size
        latent_channels: Latent channels
        sample_size: Sample size for latent dataset
        is_distributed: Whether distributed training
        rank_id: Rank ID
        device_num: Number of devices

    Returns:
        Tuple of (dataloader, dataset_size)
    """
    if latent_path:
        dataset = ImageNetLatentDataset(
            latent_path=latent_path,
            latent_channels=latent_channels,
            sample_size=sample_size,
        )
        latent_mode = True
    else:
        dataset = ImageNetDataset(
            dataset_path=dataset_path,
            image_size=image_size,
            split="train",
        )
        latent_mode = False

    dataloader = create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_remainder=drop_remainder,
        num_workers=num_workers,
        prefetch_size=prefetch_size,
        device_num=device_num,
        rank_id=rank_id,
        python_multiprocessing=False,
    )

    return dataloader, len(dataset)


def main(cfg):
    """Main training function."""
    output_path = Path(cfg.train.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    device_id, rank_id, device_num = init_env(
        mode=cfg.env.mode,
        device=cfg.env.device,
        distributed=cfg.env.distributed,
        seed=cfg.env.seed,
    )
    set_random_seed(cfg.env.seed)
    set_seed(cfg.env.seed)
    ds.set_seed(cfg.env.seed)

    set_logger(name="", output_dir=str(output_path), rank=rank_id)

    logger.info(f"Output directory: {output_path}")
    logger.info(f"Device: {cfg.env.device}, Rank: {rank_id}, Device Num: {device_num}")

    model_dtype = ms.float32
    if cfg.env.dtype == "fp16":
        model_dtype = ms.float16
    elif cfg.env.dtype == "bf16":
        model_dtype = ms.bfloat16

    sample_size = cfg.data.image_size // 8
    logger.info(f"Latent sample size: {sample_size}")

    logger.info("Creating VAE...")
    vae = create_vae(cfg.model.vae_model, tiling=cfg.vae.tiling)
    vae.set_train(False)
    for param in vae.get_parameters():
        param.requires_grad = False

    logger.info("Creating scheduler...")
    scheduler = create_scheduler_fn(
        scheduler_type=cfg.scheduler.type,
        beta_start=cfg.scheduler.beta_start,
        beta_end=cfg.scheduler.beta_end,
        beta_schedule=cfg.scheduler.beta_schedule,
        num_train_timesteps=cfg.scheduler.num_train_timesteps,
        clip_sample=cfg.scheduler.clip_sample,
    )

    logger.info("Creating DiT model...")
    network = create_dit_model(
        model_type=cfg.model.model_type,
        latent_channels=cfg.model.latent_channels,
        patch_size=cfg.model.patch_size,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        head_dim=cfg.model.head_dim,
        sample_size=sample_size,
        num_embeds_ada_norm=cfg.scheduler.num_train_timesteps,
    )
    network.set_train(True)

    total_params, trainable_params = count_params(network)
    logger.info(f"Total parameters: {total_params:,}, Trainable parameters: {trainable_params:,}")

    logger.info("Creating dataset...")
    dataloader, dataset_size = initialize_dataset(
        dataset_path=cfg.data.dataset_path,
        latent_path=cfg.data.latent_path,
        image_size=cfg.data.image_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=cfg.data.shuffle,
        drop_remainder=cfg.data.drop_remainder,
        prefetch_size=cfg.data.prefetch_size,
        latent_channels=cfg.model.latent_channels,
        sample_size=sample_size,
        is_distributed=cfg.env.distributed,
        rank_id=rank_id,
        device_num=device_num,
    )
    logger.info(f"Dataset size: {dataset_size}")

    steps_per_epoch = dataset_size // (cfg.data.batch_size * device_num)
    total_steps = steps_per_epoch * cfg.train.num_epochs
    warmup_steps = cfg.train.scheduler.warmup_steps

    logger.info(f"Steps per epoch: {steps_per_epoch}, Total steps: {total_steps}, Warmup steps: {warmup_steps}")

    logger.info("Creating optimizer...")
    optimizer = create_optimizer(
        network.trainable_params(),
        opt=cfg.train.optimizer.type,
        lr=cfg.train.optimizer.lr,
        weight_decay=cfg.train.optimizer.weight_decay,
        beta1=cfg.train.optimizer.beta1,
        beta2=cfg.train.optimizer.beta2,
    )

    logger.info("Creating scheduler...")
    scheduler_fn = create_scheduler(
        optimizer,
        num_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr=cfg.train.scheduler.min_lr,
        scheduler_type=cfg.train.scheduler.type,
    )

    logger.info("Creating training pipeline...")
    train_pipeline = DiTDiffusionWithLoss(
        network=network,
        vae=vae,
        noise_scheduler=scheduler,
        latent_scaling_factor=cfg.vae.scaling_factor,
        class_dropout_prob=cfg.model.class_dropout_prob,
        latent_channels=cfg.model.latent_channels,
    )

    if cfg.train.ema.enable:
        logger.info(f"Creating EMA with decay={cfg.train.ema.decay}...")
        ema = EMA(network, decay=cfg.train.ema.decay, update_interval=cfg.train.ema.update_interval)
    else:
        ema = None

    callbacks = []

    if rank_id == 0:
        callbacks.append(OverflowMonitor())

        save_callback = EvalSaveCallback(
            network=network,
            ema_network=ema,
            save_path=str(output_path / "checkpoints"),
            ckpt_save_interval=cfg.train.save_checkpoint_steps,
            ckpt_save_total_limit=cfg.train.save_checkpoint_total_limit,
            step_interval=cfg.train.log_interval,
            log_interval=cfg.train.log_interval,
            metric_for_best_model="loss",
            greater_or_smaller="smaller",
            model_name="dit",
        )
        callbacks.append(save_callback)

    if hasattr(cfg.train, "max_steps") and cfg.train.max_steps > 0:
        callbacks.append(StopAtStepCallback(steps=cfg.train.max_steps))

    logger.info("Creating model...")
    model = Model(train_pipeline, optimizer=optimizer, scheduler=scheduler_fn, amp_level="O2" if model_dtype == ms.float16 else "OFF")

    logger.info("Starting training...")
    start_time = datetime.now()

    model.train(
        cfg.train.num_epochs,
        dataloader,
        callbacks=callbacks,
        dataset_sink_mode=(cfg.env.mode == GRAPH_MODE),
    )

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"Training completed in {elapsed / 3600:.2f} hours")

    if rank_id == 0 and ema is not None:
        final_ckpt_path = output_path / "checkpoints" / "dit_final.ckpt"
        ema_state = ema.export()
        ms.save_checkpoint(ema_state, str(final_ckpt_path))
        logger.info(f"Final EMA checkpoint saved to {final_ckpt_path}")


if __name__ == "__main__":
    cfg = parser.parse_args()
    main(cfg)
