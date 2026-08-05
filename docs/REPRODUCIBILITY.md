# Reproducibility guide

## Environment

The `environment/` folder contains local `scratchgen` exports:

- `scratchgen_environment.yml`
- `scratchgen_full_macos.yml`
- `scratchgen_requirements_macos.txt`
- `scratchgen_runtime.txt`

The macOS export may not reproduce the RC environment exactly.

## Data layout

```text
Implementation_trial/cDCGAN_SDI/data/prepared_A/
├── normal/
├── scratches/
└── spots/
```

The procedural dataset can be regenerated with:

```bash
python Implementation_trial/cDCGAN_SDI/experiments/04_controlled_simple_scratch/generate_simple_scratch_dataset.py
```

Review paths and seeds before running.

## Slurm workflow

```bash
sbatch run_file.slurm
squeue --me
sacct -j JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed,Start,End
tail -f log_file.out
scancel JOB_ID
```

## Record for every new run

- hypothesis
- data version and counts
- preprocessing/augmentation
- batch composition
- seed
- epochs and steps
- code commit
- Slurm job ID
- RC output path
- environment
- fixed/random latent outputs
- qualitative and quantitative results
