# Main cDCGAN experiments on SDI Type-A metal surfaces

## Research questions

1. Can the model generate the requested defect type?
2. Which defect types are easier to learn?
3. What improves output quality under scarcity and imbalance?
4. Do scratch failures persist after background complexity and unequal exposure are removed?

## Data

- Normal: 702
- Scratch: 344
- Spot: 181
- 128 × 128 grayscale

The full dataset is intentionally excluded from Git. See `data/README.md`.

## Experiment progression

### 1. Three-class baseline

One model generated normal, scratch, and spot classes. It learned the shared metal domain more readily than class-specific defect structure.

### 2. Normal versus scratch

Binary decomposition, stratified sampling, enhancement, and mild augmentation improved class separation, but scratch structures remained limited and repetitive.

### 3. Normal versus spot

Localized spot-like cues were easier to model than elongated scratch geometry. Consistency improved more clearly than diversity.

### 4. Controlled simple-scratch experiments

A procedural dataset removed complex metal texture and varied scratch angle, position, length, width, and intensity.

- Exp7: 700 normal / 700 scratch; batches of 8 normal + 8 scratch.
- Exp8: 100 normal / 700 scratch; batches of 2 normal + 14 scratch.

Neither run produced reliable scratch geometry. Exp8 produced point/grid/checkerboard artifacts under both labels and degraded normal fidelity.

## Interpretation

The controlled experiments weaken two explanations:

- the metal background was the primary obstacle;
- scratches merely needed much greater exposure.

The evidence instead points toward limitations in the tested global-label conditioning mechanism and full-image generation setup.
