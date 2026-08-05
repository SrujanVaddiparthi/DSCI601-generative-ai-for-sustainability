# Experiment catalog

## Learning stage

### MNIST cDCGAN
Purpose: understand class conditioning and fixed-latent comparisons.
Status: learning experiment only.

### TensorFlow CycleGAN tutorial
Purpose: understand unpaired translation, cycle consistency, and PatchGAN.
Status: tutorial reproduction only.

## Track 01 — Three-class SDI

Classes: normal, scratch, spot.

Finding: shared surface texture was learned more readily than class identity. Diversity and class consistency were low.

Decision: split into binary tasks.

## Track 02 — Normal versus scratch

- Exp4: binary baseline
- Exp5: stratified batching
- Exp6: final targeted low-data configuration

Finding: separation improved, but scratch geometry remained weak and repetitive.

## Track 03 — Normal versus spot

- Exp1: baseline
- Exp2: fixed enhancement
- Exp3: stratified randomized enhancement
- Exp4: augmented configuration

Finding: spot outputs showed the clearest consistency improvement, though diversity remained low.

## Track 04 — Controlled simple scratch

### Exp7 balanced

- 700 normal / 700 scratch
- 8 normal + 8 scratch per batch
- 82 epochs

Finding: only occasional small line-like cues; geometry, consistency, and diversity remained weak.

### Exp8 scratch-heavy

- 100 normal / 700 scratch
- 2 normal + 14 scratch per batch
- 82 epochs

Finding: coherent scratches did not emerge. Point-, grid-, and checkerboard-like artifacts appeared under both labels, normal fidelity degraded, and fixed-latent pairs often remained similar.

## Controlled-experiment conclusion

Simplifying the background and greatly increasing scratch exposure were not sufficient. The tested label-only conditioning mechanism remained weak.
