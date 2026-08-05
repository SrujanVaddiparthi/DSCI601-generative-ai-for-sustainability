# Project handover

## Status

The main experiment sequence is complete. The project progressed from learning exercises to real SDI experiments and then to controlled diagnostic experiments. The repository, final report, and shared RC artifacts preserve the main scientific story; some historical experiment-to-job mappings remain `TBD` because a generic Slurm job name was reused.

## Work completed

1. MNIST conditional-GAN prototype.
2. TensorFlow CycleGAN tutorial reproduction.
3. SDI Type-A preparation.
4. Three-class cDCGAN.
5. Binary normal-vs-scratch and normal-vs-spot tracks.
6. Oversampling, stratified batching, enhancement, and mild augmentation.
7. Migration to the RIT RC cluster.
8. OOM diagnosis and refactor toward `tf.data` and `GradientTape`.
9. Procedural simple-scratch dataset.
10. Balanced and scratch-heavy controlled experiments.
11. Fixed-latent comparisons.
12. RC audit and run-ledger reconstruction.

## Strongest findings

- Shared background texture was easier than defect geometry.
- Binary decomposition improved class separation.
- Spots were easier than scratches.
- Low-data interventions improved consistency more than diversity.
- Removing metal texture did not solve scratch generation.
- Increasing scratch exposure from 50% to 87.5% did not solve conditioning.
- Fixed-latent pairs often remained similar after changing only the label.

## Important infrastructure findings

- Early three-class runs exhausted both 32G and 64G host-memory allocations.
- The memory-growth causes were investigated as engineering hypotheses; no single root cause was formally proven.
- Refactoring toward `tf.data`, `GradientTape`, stable training functions, lower batch size, and controlled saving produced successful low-memory runs.
- Exp7 had a four-second Spack/OpenSSL setup failure (`21443794`) before Python execution, followed by a successful clean-shell submission (`21443801`).
- Exp8 completed successfully as job `21443819`.
- The archived Spack environment was `default-ml-x86_64-25052701`; it must be reverified before future use.

## Where a new researcher should begin

1. Root README.
2. `docs/EXPERIMENT_CATALOG.md`.
3. `Implementation_trial/cDCGAN_SDI/README.md`.
4. Exp7 and Exp8 scripts and fixed-latent grids.
5. `docs/TROUBLESHOOTING.md`.
6. `docs/RC_RUN_LEDGER.md`.
7. `docs/RC_ACCESS.md`.
8. `docs/ARTIFACTS_AND_STORAGE.md`.
9. `docs/REPRODUCIBILITY.md`.
10. `docs/NEXT_STEPS.md`.
11. `reports/final_report/Synthetic_Industrial_Surface_Defect_Generation_Final_Report.pdf`.

## Operational handover

### 1. Obtain individual access

The successor must use their own RIT identity and obtain current RC access. Credentials, multifactor-authentication access, private keys, and sessions must never be shared.

### 2. Join the existing project and allocations

An authorized ColdFront project manager or PI should add the successor to the `defgengan` project and the required storage/cluster allocations. The exact ownership, manager permissions, expiration, and renewal workflow must be confirmed with RIT Research Computing.

### 3. Verify shared storage

The authoritative project root is:

```text
/shared/rc/defgengan
```

Verify read/write access to the appropriate project directories:

```text
data/
scripts/
logs/
outputs/
```

Do not rely on `/home/<RIT_USERNAME>` as shared handover storage.

### 4. Verify the software environment

Archived scripts use:

```bash
spack env activate default-ml-x86_64-25052701
```

Confirm that the environment still exists and run a small GPU/environment test before launching a full experiment. Historical Python, TensorFlow, CUDA, and cuDNN versions remain `TBD` where not proven.

### 5. Start with a controlled test

Use a short debug run or the procedural dataset before changing the model. Record a new job ID rather than overwriting the historical ledger.

## Verified shared experiment roots

The shared output inventory includes all principal three-class, normal-vs-scratch, normal-vs-spot, Exp7, and Exp8 roots documented in `RC_RUN_LEDGER.md`.

The personal RC home inventory also showed:

```text
/home/<RIT_USERNAME>/exp3_3class_fullrun
```

Compare it with shared storage and copy only unique, nonprivate artifacts if needed.

## Recommended continuation

Introduce explicit spatial or geometric conditioning:

- scratch masks;
- angle, location, length, and width parameters;
- defect transfer;
- image-to-image insertion;
- spatially adaptive normalization;
- controlled diffusion or inpainting.

Use the procedural dataset first so every architectural change tests a clear hypothesis.

## Do not begin with

- more epochs on the unchanged model;
- only changing class ratios or latent dimension;
- judging success from GAN losses alone;
- treating plausible metal texture as successful defect generation;
- increasing requested RAM without diagnosing progressive growth;
- submitting duplicate jobs to improve queue priority;
- editing multiple environment variables before isolating a setup failure.

## Unresolved items

These remain `TBD` unless additional authoritative records are recovered:

- exact job IDs for normal-vs-scratch Exp4–Exp6;
- exact job IDs for normal-vs-spot Exp1–Exp4;
- exact historical Python/TensorFlow/CUDA/cuDNN versions;
- current ColdFront owner/manager permissions;
- allocation and project expiration/renewal status;
- successor and former maintainer eligibility after affiliation changes.

Contact RIT Research Computing for administrative or policy questions rather than inferring them from the archived project state.
