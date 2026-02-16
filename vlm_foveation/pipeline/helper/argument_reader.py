"""Command-line argument utilities for the VLM foveation experiments."""

import argparse


def get_args():
    """Return parsed command-line arguments for the experiment entry point."""
    parser = argparse.ArgumentParser(
        description="Configure datasets, models, and logging for VLM foveation studies."
    )
    parser.add_argument(
        "--model_name",
        default="llava-1.6-34b",
        help="Name or path of the target VLM checkpoint.",
    )
    parser.add_argument(
        "--dataset",
        default="VQA-MHUG",
        help="Dataset identifier such as VQA-MHUG, AiR-D, or VOILA-COCO.",
    )
    parser.add_argument(
        "--log_dir",
        default="logs",
        help="Directory for saving logs, metrics, and artifacts.",
    )
    parser.add_argument(
        "--sampling_method",
        default="uniform",
        help="Foveation sampling policy (uniform, saliency-guided, etc.).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility across sampling and evaluation.",
    )
    return parser.parse_args()
