# Recommended next steps

## 1. Spatial or mask-guided synthesis

Provide a binary or soft mask describing where the defect should appear.

Test:

- same latent vector with different masks
- same mask with different latent vectors
- empty mask versus scratch mask
- generated scratch overlap with requested mask

## 2. Explicit geometric conditioning

Condition on normalized:

- center position
- angle
- length
- width
- severity/intensity

The procedural dataset is well suited to this because its geometry is known.

## 3. Separate background and defect synthesis

Begin with a real normal surface and generate/transfer only the defect instead of synthesizing the whole image from noise.

## 4. Quantitative evaluation

Measure:

- scratch-presence rate
- label fidelity
- angle/length/width error
- spatial-distribution error
- diversity
- memorization
- downstream classifier benefit

## Suggested immediate experiment

Train on the procedural dataset using an explicit scratch mask. Compare identical latent inputs across empty-mask and scratch-mask conditions.
