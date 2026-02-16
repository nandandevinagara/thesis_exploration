# VLM Foveation Experiments

Research sandbox for studying foveation-aware Vision–Language Models (VLMs). The codebase
keeps dataset preparation, sampling pipelines, model calls, and evaluation logic modular so
new experiments can be added without refactoring core components.

## Quick Start
1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the entry point with your preferred configuration:
   ```bash
   python pipeline/main.py \
       --model_name llava-1.6-34b \
       --dataset VQA-MHUG \
       --log_dir runs/llava_baseline \
       --sampling_method uniform \
       --seed 42
   ```

## Pipeline Overview
- **Dataset loading**: `pipeline/main.py` will use TODO hooks to locate the dataset described in
  `datasets/` and prepare batches for foveation.
- **Foveation / sampling**: Placeholder logic will evolve into saliency-aware downsampling and
  multi-resolution crops living under `pipeline/`.
- **VLM inference**: Calls into models referenced by `--model_name` (e.g., LLaVA, Qwen-VL) once
  adapters are integrated.
- **Evaluation**: Future modules will log accuracy, compute, and qualitative artifacts to
  `--log_dir` for comparison across experiments.

Keep contributions focused on reproducibility: document dataset preprocessing steps in
`datasets/`, capture experiment manifests in `experiments/`, and prefer lightweight helper
functions so the main pipeline stays readable.
