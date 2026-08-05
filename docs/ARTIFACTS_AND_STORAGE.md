# Artifacts and storage

## Local archive

```text
~/synthetic-industrial-defect-generation-local-archive/
├── original_unorganized_snapshot/
├── raw_data/
├── full_outputs/
├── failed_runs/
├── model_checkpoints/
├── notebooks_with_outputs/
└── external_code/
```

The pre-cleanup snapshot preserves the original repository state.

## What belongs in Git

- code
- Slurm files
- environment exports
- small summaries/loss histories
- selected representative outputs
- preparation/generation scripts
- documentation

## What does not belong in normal Git history

- full SDI data
- generated epoch sequences
- checkpoints/models
- copied third-party repositories
- failed-run output trees
- large notebook outputs
- compressed dataset archives

A compressed dataset is still a large opaque binary and may also raise hosting or redistribution issues.

## Handover warning

A laptop-only archive is not enough. Copy irreplaceable artifacts into approved shared project storage accessible to the professor and future researcher.
