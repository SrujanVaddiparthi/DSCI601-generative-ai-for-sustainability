# Synthetic Industrial Surface Defect Generation Under Low-Data Constraints

This repository documents a staged investigation of whether a label-conditioned cDCGAN can synthesize defect-faithful industrial metal-surface images when real defect data is scarce and imbalanced.

## Data

Type-A SDI metal-surface subset:

- Normal: 702 images
- Scratch: 344 images
- Spot: 181 images
- 128 × 128 grayscale

## Main conclusion

The tested cDCGAN learned shared surface texture more readily than defect-specific geometry. Binary decomposition and targeted low-data strategies improved class consistency more clearly than diversity, especially for spot defects.

Two controlled scratch experiments then removed complex metal texture and increased scratch exposure. Neither produced reliable scratch geometry or strong class-label control. This conclusion is specific to the tested architecture, conditioning mechanism, loss formulation, and training workflow.

## Repository map

- `Implementation_trial/learning_experimentation/` — MNIST cDCGAN and CycleGAN learning work.
- `Implementation_trial/cDCGAN_SDI/` — main SDI implementation and selected results.
- `docs/HANDOVER.md` — current status and where a new researcher should begin.
- `docs/EXPERIMENT_CATALOG.md` — chronological experiment map.
- `docs/NEXT_STEPS.md` — recommended continuation.
- `docs/REPRODUCIBILITY.md` — environments, data, and run-record guidance.
- `docs/RC_ACCESS.md` — RIT Research Computing workflow used in the project.
- `docs/TROUBLESHOOTING.md` — cluster OOM diagnosis and training-pipeline stabilization.
- `docs/RC_RUN_LEDGER.md` — experiment and Slurm run-record template.
- `docs/ARTIFACTS_AND_STORAGE.md` — large-file and archive policy.
- `references/READING_LIST.md` — report references and reading priorities.

## Recommended reading order

1. This README
2. `docs/HANDOVER.md`
3. `docs/EXPERIMENT_CATALOG.md`
4. `Implementation_trial/cDCGAN_SDI/README.md`
5. Experiment 7 and 8 code and fixed-latent outputs
6. `docs/TROUBLESHOOTING.md`
7. `docs/RC_RUN_LEDGER.md`
8. `docs/NEXT_STEPS.md`
9. `reports/final_report/Synthetic_Industrial_Surface_Defect_Generation_Final_Report.pdf`

## Public-release checks

- Add the exact official SDI source URL and verify redistribution terms.
- Add the original URL/license for reviewed third-party code.
- Verify the intended public repository name/course number.
- Move irreplaceable laptop-only artifacts into approved shared storage.
- Scan for credentials and hard-coded personal paths.
