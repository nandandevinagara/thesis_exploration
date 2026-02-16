# Dataset Guidelines

Store all raw and preprocessed Vision–Language datasets inside this directory. Each dataset
should live in its own subfolder to keep annotations, images, and derived features scoped to
that corpus.

```
datasets/
├── VQA-MHUG/
├── AiR-D/
└── VOILA-COCO/
```

Recommended layout inside each dataset folder:
- `images/` or `frames/` for visual assets after optional foveation.
- `annotations/` for JSON/CSV question-answer pairs or grounding metadata.
- `splits/` to cache train/val/test indices that the pipeline can reuse.
- `README.md` capturing preprocessing notes unique to the dataset.

Keep dataset assets out of version control; only commit lightweight metadata or scripts that
reproduce the preprocessing pipeline.
