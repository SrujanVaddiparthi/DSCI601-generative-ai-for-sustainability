# Artifacts and storage

## Storage principles

The project uses three distinct artifact locations:

1. Git repository: code, documentation, compact summaries, and selected representative outputs.
2. Shared RC project storage: datasets, scripts, logs, checkpoints, and full experiment outputs needed by authorized collaborators.
3. Personal/local archives: original snapshots, large notebooks, copied external code, and other material that should not be treated as the authoritative shared handover location.

## Verified shared RC layout

```text
/shared/rc/defgengan/
├── data/       approximately 85M at audit time
├── scripts/    approximately 230K
├── logs/       approximately 454K
└── outputs/    approximately 1.2G
```

Total audited project storage was approximately 1.3G.

The shared root and its main subdirectories were group-accessible under the RC project group at audit time.

## Verified data under shared storage

```text
/shared/rc/defgengan/data/prepared_A/
├── normal/
├── scratches/
└── spots/

/shared/rc/defgengan/data/simple_scratch_v1/
├── normal/
├── scratches/
├── dataset_preview.png
└── metadata.csv

/shared/rc/defgengan/data/simple_scratch_v1.tar.gz
```

The inventory also showed a small macOS metadata file:

```text
/shared/rc/defgengan/data/._simple_scratch_v1
```

Future cleanup may remove such `._*` files after verifying they are not needed.

## Verified scripts under shared storage

```text
train_cdcgan_3class_cluster.py
train_cdcgan_3class_cluster_v2.py
run_cdcgan_3class.slurm
train_cdcgan_normalVscratch_cluster_v2.py
train_cdcgan_normalVscratch_cluster_v3_stratified.py
train_cdcgan_normalVscratch_cluster_v4_exp6.py
run_cdcgan_normalVscratch.slurm
train_cdcgan_normalVspot_cluster_v1_baseline.py
train_cdcgan_normalVspot_cluster_v2_fixed_enhancement.py
train_cdcgan_normalVspot_cluster_v3_stratified.py
train_cdcgan_normalVspot_cluster_v4_augmented.py
run_cdcgan_normalVspot.slurm
train_cdcgan_simple_scratch_exp7.py
run_cdcgan_simple_scratch_exp7.slurm
train_cdcgan_simple_scratch_exp8.py
run_cdcgan_simple_scratch_exp8.slurm
```

Python bytecode under `scripts/__pycache__` is not a handover artifact and can be regenerated.

## Verified output roots

```text
exp3_3class_baseline
exp3_3class_fullrun
exp3_3class_fullrun_v2
exp4_normal_vs_scratch
exp5_normal_vs_scratch_stratified
exp6_normal_vs_scratch_final
exp_spot_1_baseline_normal_vs_spot
exp_spot_2_fixed_enhancement_normal_vs_spot
exp_spot_3_stratified_randomized
exp_spot_4_stratified_randomized_augmented
exp7_simple_scratch_geometry
exp8_simple_scratch_100N_700S
```

Each root had experiment subdirectories such as `checkpoints`, `final_models`, `generated_samples`, and `losses`. Exp7 and Exp8 also had `fixed_z` and `random_z` sample directories.

Directory presence does not guarantee that every run completed or every subdirectory is populated. Use `RC_RUN_LEDGER.md`, logs, and file-level inspection together.

## Logs

Shared logs follow the archived Slurm pattern:

```text
/shared/rc/defgengan/logs/%x_%j.out
/shared/rc/defgengan/logs/%x_%j.err
```

Retain both stdout and stderr for every important job. A nonempty stderr file does not necessarily mean failure because TensorFlow writes informational and warning messages there.

## Personal RC home directory

The audit found this project-related directory under personal storage:

```text
/home/<RIT_USERNAME>/exp3_3class_fullrun
```

It also found the audit bundle under the personal home directory. Personal storage is not the authoritative collaboration location. Compare any unique files against shared project storage, copy irreplaceable artifacts to approved shared storage, and avoid copying private account material.

The audit inventory listed `.ssh`, shell-history, and configuration files. These are explicitly excluded from handover and must never be copied into the repository or shared project archive.

## Local archive

The existing local archive plan remains useful:

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

The pre-cleanup snapshot preserves the original repository state. A laptop-only archive is not sufficient for collaboration or long-term handover.

## What belongs in Git

- source code;
- Slurm files;
- environment exports;
- small JSON/CSV/NumPy summaries and loss histories;
- selected representative fixed/random latent outputs;
- preparation and procedural-data-generation scripts;
- documentation;
- file manifests and hashes.

## What does not belong in normal Git history

- full SDI data;
- full generated epoch sequences;
- checkpoints and final models;
- copied third-party repositories;
- failed-run output trees;
- large notebook outputs;
- compressed dataset archives;
- credentials, authentication files, shell history, or private configuration.

A compressed dataset remains an opaque binary and may also raise redistribution or licensing issues.

## Handover checklist

- Confirm `/shared/rc/defgengan` remains active and accessible to the successor.
- Verify the successor can read project data and write to the intended output/log locations.
- Compare `/home/<RIT_USERNAME>/exp3_3class_fullrun` against the shared copy and archive only unique, nonprivate artifacts.
- Preserve job logs associated with ledger entries.
- Generate checksums for archives and final retained models.
- Record external dataset licenses and redistribution constraints.
- Do not treat personal home storage or a laptop archive as the only copy of irreplaceable work.
