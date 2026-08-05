# Project handover

## Status

The main experiment sequence is complete. The project progressed from learning exercises to real SDI experiments and then to controlled diagnostic experiments.

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

## Strongest findings

- Shared background texture was easier than defect geometry.
- Binary decomposition improved class separation.
- Spots were easier than scratches.
- Low-data interventions improved consistency more than diversity.
- Removing metal texture did not solve scratch generation.
- Increasing scratch exposure from 50% to 87.5% did not solve conditioning.
- Fixed-latent pairs often remained similar after changing only the label.

## Where a new researcher should begin

1. Root README
2. `docs/EXPERIMENT_CATALOG.md`
3. `Implementation_trial/cDCGAN_SDI/README.md`
4. Exp7 and Exp8 scripts and fixed-latent grids
5. `docs/TROUBLESHOOTING.md`
6. `docs/RC_RUN_LEDGER.md`
7. `docs/RC_ACCESS.md`
8. `docs/NEXT_STEPS.md`
9. `reports/final_report/Synthetic_Industrial_Surface_Defect_Generation_Final_Report.pdf`

## Recommended continuation

Introduce explicit spatial or geometric conditioning:

- scratch masks
- angle/location/length/width parameters
- defect transfer
- image-to-image insertion
- spatially adaptive normalization
- controlled diffusion/inpainting

Use the procedural dataset first so every architectural change tests a clear hypothesis.

## Do not begin with

- more epochs on the unchanged model
- only changing class ratios or latent dimension
- judging success from GAN losses alone
- treating plausible metal texture as successful defect generation
