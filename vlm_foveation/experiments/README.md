# Experiment Tracking

Use this folder to organize experiment definitions, configuration files, and analysis reports.
A typical layout might include:

```
experiments/
├── baselines/
├── foveation_methods/
└── efficiency_vs_accuracy/
```

Suggested conventions:
- Store YAML/JSON configs describing model_name, dataset, sampling_method, and seed.
- Keep notebooks or scripts that aggregate metrics for each study area (baseline versus
  foveated variants, saliency heuristics, etc.).
- Document trade-offs in accuracy, latency, and compute for every configuration so future
  researchers can replicate or extend findings quickly.
- Summaries of best-performing settings should link back to the exact command (arguments and
  git commit) used to run `pipeline/main.py`.
