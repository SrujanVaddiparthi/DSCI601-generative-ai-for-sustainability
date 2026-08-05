# Recommended next steps

The completed cDCGAN experiments suggest that global class labels and full-image generation are insufficient for reliable scratch geometry and strong within-class diversity. The next phase should therefore move from class-only generation toward explicit control over defect location, geometry, foreground, and background.

## 1. Establish a mask-conditioned baseline on the procedural dataset

Begin with the procedurally generated controlled scratch dataset because the true scratch geometry is known.

Provide the model with:

- a normal or empty mask
- a scratch mask
- the source background image
- optional geometric attributes

Test:

- the same source and latent vector with different masks
- the same mask with different latent vectors
- an empty mask versus a scratch mask
- whether generated defects overlap the requested region
- whether changing only the label or mask changes the intended structure

This is the most direct way to test whether explicit spatial conditioning solves the failure observed in Exp7 and Exp8.

## 2. Add explicit geometric conditioning

Condition on normalized:

- center position
- angle
- length
- width
- severity or intensity

The procedural generator already provides these variables, so the successor can evaluate geometry rather than relying only on visual judgment.

Measure:

- center-position error
- angle error
- length and width error
- mask overlap
- scratch-presence rate
- fidelity when the requested mask is empty

## 3. Move from full-image generation to defect insertion or transfer

Use a real normal surface as the background and synthesize or transfer only the defect.

A DT-GAN-inspired direction is especially relevant because it separates foreground defect information from background product information and supports latent-guided or reference-guided defect synthesis.

Recommended reading:

- *Defect Transfer GAN: Diverse Defect Synthesis for Data Augmentation*
- BMVC page: https://bmvc2022.mpi-inf.mpg.de/445/
- arXiv: https://arxiv.org/abs/2302.08366

A practical staged comparison would be:

1. source normal image + binary defect mask
2. source normal image + reference defect image
3. foreground/background disentanglement
4. defect style and geometry varied independently
5. cross-product transfer after Type-A experiments are stable

Do not assume that a direct reproduction of DT-GAN code is available. Treat the paper as an architectural reference unless an official implementation is verified.

## 4. Evaluate controlled diffusion or inpainting

After the mask-conditioned baseline works, test a diffusion-based insertion or inpainting approach.

The diffusion experiment should preserve the normal surface while generating only the masked defect region. Compare:

- fixed mask with different random seeds
- fixed seed with different masks
- source-image preservation outside the mask
- defect diversity within the requested region
- scratch versus spot performance
- few-shot adaptation cost and compute requirements

Because the real defect dataset is small, begin from a pretrained image or inpainting model and evaluate lightweight adaptation before considering training a diffusion model from scratch.

## 5. Compare against the existing cDCGAN evidence

Keep the current cDCGAN results as the baseline.

The successor should compare new approaches on:

- defect visibility
- class or condition fidelity
- location control
- geometry control
- within-condition diversity
- background preservation
- memorization and nearest-neighbor similarity
- failure artifacts
- training and inference cost

Use fixed inputs, masks, references, and random seeds so model comparisons are interpretable.

## 6. Add downstream evaluation

The ultimate question is whether synthetic defects improve an inspection model rather than only appearing plausible.

Evaluate:

- real-only classifier performance
- traditional augmentation
- real plus cDCGAN samples
- real plus mask-conditioned samples
- real plus defect-transfer samples
- real plus diffusion/inpainting samples

Report per-class precision, recall, F1, confusion matrices, and performance under extremely limited real-defect sample counts.

## Suggested immediate experiment

Train a mask-conditioned model on the procedural dataset.

Use the same normal source image and latent input for two conditions:

1. empty mask
2. generated scratch mask

The first success criterion is not photorealism. It is whether the model reliably places a line-like scratch inside the requested region while leaving the background unchanged outside it.
