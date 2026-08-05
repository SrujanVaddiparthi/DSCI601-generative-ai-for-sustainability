# Reproducibility guide

## Reproducibility scope

This project contains research experiments, engineering fixes, and shared-cluster execution records. Reproducing a result requires recording both the scientific configuration and the Slurm attempt that produced it.

## Local environment records

The `environment/` folder contains local `scratchgen` exports:

- `scratchgen_environment.yml`
- `scratchgen_full_macos.yml`
- `scratchgen_requirements_macos.txt`
- `scratchgen_runtime.txt`

The macOS export may not reproduce the RC environment exactly.

## RC software environment

Archived Slurm scripts activate:

```bash
spack env activate default-ml-x86_64-25052701
```

This environment was used by archived project scripts and successful runs, but it must be reverified before reuse. The evidence base does not establish a single authoritative Python, TensorFlow, CUDA, or cuDNN version for all successful jobs. Record those values as `TBD` for historical runs and print them explicitly in future logs.

Suggested future environment header:

```python
import platform
import tensorflow as tf

print("Python:", platform.python_version())
print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
print("Build info:", tf.sysconfig.get_build_info())
```

## Data layout

Real SDI-derived prepared data:

```text
/shared/rc/defgengan/data/prepared_A/
├── normal/
├── scratches/
└── spots/
```

Procedural controlled data:

```text
/shared/rc/defgengan/data/simple_scratch_v1/
├── normal/
├── scratches/
├── metadata.csv
└── dataset_preview.png
```

Archive:

```text
/shared/rc/defgengan/data/simple_scratch_v1.tar.gz
```

The procedural dataset can be regenerated from the repository with:

```bash
python Implementation_trial/cDCGAN_SDI/experiments/04_controlled_simple_scratch/generate_simple_scratch_dataset.py
```

Review output paths, seeds, image counts, and generation parameters before running.

## Archived Slurm defaults

```text
partition: tier3
account: defgengan
GPU: A100, one GPU
CPUs per task: 8
memory: 64G
wall time: 5 hours
```

Early OOM attempts used 32G and 64G. Use the per-attempt ledger rather than assuming all historical runs used the final defaults.

## Clean submission workflow

```bash
ssh <RIT_USERNAME>@sporcsubmit.rc.rit.edu
cd /shared/rc/defgengan/scripts
sbatch run_file.slurm
squeue --me
sacct -j <JOB_ID> --format=JobID,JobName,State,ExitCode,Elapsed,Start,End,ReqMem,MaxRSS
```

The batch script performs Spack activation. A clean-shell direct submission resolved the Exp7 environment failure. Do not manually activate the environment before `sbatch` unless current RC guidance requires it.

Follow logs:

```bash
tail -f /shared/rc/defgengan/logs/<JOB_NAME>_<JOB_ID>.out
tail -n 100 /shared/rc/defgengan/logs/<JOB_NAME>_<JOB_ID>.err
```

Cancel only the intended job:

```bash
scancel <JOB_ID>
```

## Retained experiment scripts and outputs

| Experiment | Script | Output root |
|---|---|---|
| Three-class stabilized | `train_cdcgan_3class_cluster_v2.py` | `exp3_3class_fullrun_v2` |
| Normal-vs-scratch Exp4 | `train_cdcgan_normalVscratch_cluster_v2.py` | `exp4_normal_vs_scratch` |
| Normal-vs-scratch Exp5 | `train_cdcgan_normalVscratch_cluster_v3_stratified.py` | `exp5_normal_vs_scratch_stratified` |
| Normal-vs-scratch Exp6 | `train_cdcgan_normalVscratch_cluster_v4_exp6.py` | `exp6_normal_vs_scratch_final` |
| Normal-vs-spot Exp1 | `train_cdcgan_normalVspot_cluster_v1_baseline.py` | `exp_spot_1_baseline_normal_vs_spot` |
| Normal-vs-spot Exp2 | `train_cdcgan_normalVspot_cluster_v2_fixed_enhancement.py` | `exp_spot_2_fixed_enhancement_normal_vs_spot` |
| Normal-vs-spot Exp3 | `train_cdcgan_normalVspot_cluster_v3_stratified.py` | `exp_spot_3_stratified_randomized` |
| Normal-vs-spot Exp4 | `train_cdcgan_normalVspot_cluster_v4_augmented.py` | `exp_spot_4_stratified_randomized_augmented` |
| Exp7 | `train_cdcgan_simple_scratch_exp7.py` | `exp7_simple_scratch_geometry` |
| Exp8 | `train_cdcgan_simple_scratch_exp8.py` | `exp8_simple_scratch_100N_700S` |

All output roots are under:

```text
/shared/rc/defgengan/outputs/
```

## Exp7 and Exp8 verified configurations

### Exp7

```text
job: 21443801 successful; 21443794 was a prior setup failure
classes: 700 normal, 700 scratch
batch size: 16
batch composition: 8 normal, 8 scratch
epochs: 82
seed: 42
latent dimension: 100
```

### Exp8

```text
job: 21443819
classes used: 100 normal, 700 scratch
batch size: 16
batch composition: 2 normal, 14 scratch
epochs: 82
seed: 42
latent dimension: 100
```

## Record for every new run

### Scientific configuration

- hypothesis;
- experiment identifier and parent experiment;
- data source/version, path, checksums, and counts;
- preprocessing, enhancement, and augmentation;
- class sampling and batch composition;
- random seeds;
- architecture and parameter counts;
- optimizer and learning-rate settings;
- epochs and steps per epoch;
- checkpoint and sample schedule;
- fixed and random latent vectors;
- qualitative and quantitative evaluation protocol.

### Code provenance

- repository commit;
- exact Python script path;
- exact Slurm file path;
- SHA-256 hashes of submitted files;
- command-line arguments;
- environment name and printed versions.

### Slurm provenance

- job ID and job name;
- partition and account;
- GPU request;
- CPUs;
- requested memory and wall time;
- start/end/elapsed time;
- final state and exit code;
- MaxRSS;
- stdout/stderr filenames;
- exact output root.

## Completion criteria

A run is not considered fully reproducible merely because `sacct` reports `COMPLETED`. Verify that:

- final checkpoint/model files exist;
- loss arrays and plots exist;
- expected fixed/random latent grids exist;
- run summary/configuration exists;
- logs contain the environment and final completion message;
- artifacts are copied to approved shared storage;
- the ledger maps the experiment to the exact Slurm attempt.
