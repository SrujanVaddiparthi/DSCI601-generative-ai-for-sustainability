# RIT RC Experiment Run Ledger

This ledger is a handover-oriented index of the main cDCGAN experiments. It intentionally omits passwords, Duo information, SSH keys, and private access credentials.

Use `<RIT_USERNAME>` in documentation and commands. Fill in job IDs and exact resource requests only from verified Slurm records or saved logs.

## Shared project layout

Expected shared locations:

```text
/shared/rc/defgengan/
├── data/
├── scripts/
├── logs/
└── outputs/
```

Typical connection command:

```bash
ssh <RIT_USERNAME>@sporcsubmit.rc.rit.edu
```

## Experiment ledger

| Experiment | Repository script / Slurm file | Epochs | Status supported by retained artifacts | RC job ID | Requested resources | Shared output directory / notes |
|---|---|---:|---|---|---|---|
| Three-class early run, 32 GB | `experiments/01_three_class/train_cdcgan_3class_cluster_v2.py`; `run_cdcgan_3class.slurm` | planned 82 | OOM around epoch 13 | TBD | memory 32 GB; remaining fields TBD | Historical failed run; verify exact output folder from logs |
| Three-class early run, 64 GB | same baseline files or immediate predecessor | planned 82 | OOM around epoch 28; checkpoints around epochs 19–21 were saved | TBD | memory 64 GB; remaining fields TBD | Historical failed run; verify exact output folder from logs |
| Three-class stabilized baseline | `experiments/01_three_class/train_cdcgan_3class_cluster_v2.py`; `run_cdcgan_3class.slurm` | 82 or retained run-specific value | completed after training-loop refactor | TBD | TBD | Retained representative outputs under `01_three_class/` |
| Normal vs scratch: binary baseline | `experiments/02_normal_vs_scratch/exp4_binary_baseline/` and `run_cdcgan_normalVscratch.slurm` | 200 | completed; representative epoch-200 grid retained | TBD | TBD | Historical output name: `exp4_normal_vs_scratch` |
| Normal vs scratch: stratified | `experiments/02_normal_vs_scratch/exp5_stratified/` and `run_cdcgan_normalVscratch.slurm` | 200 | completed; full output archived locally | TBD | TBD | Historical output name: `exp5_normal_vs_scratch_stratified` |
| Normal vs scratch: final | `experiments/02_normal_vs_scratch/exp6_final/` and `run_cdcgan_normalVscratch.slurm` | 200 | completed; representative epoch-200 grid retained | TBD | TBD | Historical output name: `exp6_normal_vs_scratch_final` |
| Normal vs spot: baseline | `experiments/03_normal_vs_spot/exp1_baseline/` and `run_cdcgan_normalVspot.slurm` | 200 | completed; representative epoch-200 grid retained | TBD | TBD | Historical output name: `exp_spot_1_baseline_normal_vs_spot` |
| Normal vs spot: fixed enhancement | `experiments/03_normal_vs_spot/exp2_fixed_enhancement/` and `run_cdcgan_normalVspot.slurm` | 200 | completed; full output archived locally | TBD | TBD | Historical output name: `exp_spot_2_fixed_enhancement_normal_vs_spot` |
| Normal vs spot: stratified randomized | `experiments/03_normal_vs_spot/exp3_stratified_randomized/` and `run_cdcgan_normalVspot.slurm` | 200 | completed; full output archived locally | TBD | TBD | Historical output name: `exp_spot_3_stratified_randomized` |
| Normal vs spot: augmented | `experiments/03_normal_vs_spot/exp4_augmented/` and `run_cdcgan_normalVspot.slurm` | 200 | completed; representative epoch-200 grid retained | TBD | TBD | Historical output name: `exp_spot_4_stratified_randomized_augmented` |
| Exp7: balanced procedural scratch | `experiments/04_controlled_simple_scratch/exp7_balanced/` | 82 | completed; fixed-latent epoch-82 result and losses retained | TBD | TBD | 700 normal + 700 scratch; 8 normal / 8 scratch per batch |
| Exp8: scratch-heavy procedural run | `experiments/04_controlled_simple_scratch/exp8_scratch_heavy/` | 82 | completed; fixed-latent epoch-82 result retained | TBD | TBD | 100 normal + 700 scratch; 2 normal / 14 scratch per batch |

## How to fill the missing job details

On the cluster, use verified Slurm history:

```bash
sacct -S 2026-01-01 --format=JobID,JobName,Partition,Account,AllocCPUS,State,ExitCode,Elapsed,Start,End,ReqMem,MaxRSS
```

Then narrow by job name or known date range. For a specific job:

```bash
sacct -j <JOB_ID> --format=JobID,JobName,Partition,Account,AllocCPUS,State,ExitCode,Elapsed,Start,End,ReqMem,MaxRSS
```

Keep the command on one shell line, or use a backslash only when it is the final character on the preceding line.

## Pre-handover cluster checklist

- Confirm the successor is added to the correct RC project and allocation.
- Confirm access to `/shared/rc/defgengan`.
- Confirm scripts, data, logs, and outputs are stored under shared project storage rather than only in a personal home directory.
- Record the working software-environment activation command.
- Record the final Slurm account, partition, GPU request, CPU count, memory request, and wall-time request.
- Copy any unique artifacts not already represented in Git or the local project archive.
- Do not place passwords, private keys, Duo codes, or session tokens in this repository.
