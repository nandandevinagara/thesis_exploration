"""Command-line argument utilities for the VLM foveation experiments."""

import argparse


def parse_args():
    """Define and parse finetuning-specific command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Finetune Grounding DINO with custom data."
    )
    parser.add_argument(
        "--config",
        default="configs/grounding_dino_tiny.py",
        help="Path to model config file.",
    )
    parser.add_argument(
        "--pretrained_ckpt",
        default="checkpoints/grounding_dino_tiny.pth",
        help="Starting checkpoint for finetuning.",
    )
    parser.add_argument(
        "--dataset_root",
        default="datasets/custom_vl",
        help="Root folder containing train/val splits.",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of finetuning epochs."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for optimizer."
    )
    parser.add_argument(
        "--device", default="cuda", help="Target device identifier (cuda, cuda:0, cpu)."
    )
    parser.add_argument(
        "--output_dir",
        default="runs/finetune_dino",
        help="Directory to store checkpoints and logs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Seed for deterministic training components.",
    )
    return parser.parse_args()
