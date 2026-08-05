# Troubleshooting: RIT RC cDCGAN Training

This note records the main cluster failure mode encountered during the project and the implementation changes that stabilized training.

## Scope

The issue occurred in the TensorFlow/Keras conditional DCGAN pipeline used for 128×128 grayscale surface-defect generation.

Baseline configuration at the time of failure:

- three classes: normal, scratch, and spot
- approximately 700 training exposures per class after balancing
- batch size initially 32
- 82 planned epochs
- full image arrays loaded into memory
- generated sample grid saved every epoch
- checkpoints saved at selected epochs

## Observed failure pattern

Two early RIT RC runs showed that increasing host memory only delayed the failure:

| Requested memory | Approximate failure point | Interpretation |
|---|---:|---|
| 32 GB | epoch 13 | out-of-memory failure |
| 64 GB | epoch 28 | out-of-memory failure occurred later |

Because the raw dataset tensor was comparatively small, the repeated failure at later epochs suggested progressive memory growth rather than a one-time dataset-fit problem.

## Likely contributors

The strongest implementation-level concerns were:

1. Repeatedly changing `discriminator.trainable` inside the batch loop while using compiled Keras models.
2. Repeated `train_on_batch()` execution with retracing warnings.
3. Per-epoch prediction and Matplotlib figure creation.
4. Loading the complete prepared dataset into NumPy arrays.
5. Batch size 32 increasing activation and temporary-tensor pressure.

The project did not treat generator and discriminator parameter count as the sole explanation because the timing of the failures indicated memory accumulation across epochs.

## Stabilization approach

The training pipeline was refactored while preserving the experiment's model objective:

- retained the same conditional GAN architecture and labels
- moved data handling toward `tf.data`
- replaced the older `train_on_batch()` loop with `tf.GradientTape`
- used stable compiled training steps with `@tf.function`
- avoided changing trainable state inside the inner loop
- controlled checkpoint and visualization frequency
- used a cluster-friendly batch size, with batch size 16 preferred in later runs

This separated the scientific experiment from the engineering failure: the model and research question remained the same, while the training implementation became more stable.

## Practical diagnosis checklist

When a future run fails, check these in order:

1. Job state and exit code:

   ```bash
   sacct -j <JOB_ID> --format=JobID,JobName,State,ExitCode,Elapsed,ReqMem,MaxRSS
   ```

2. Standard output and error logs:

   ```bash
   tail -n 100 /shared/rc/defgengan/logs/<OUTPUT_LOG>
   tail -n 100 /shared/rc/defgengan/logs/<ERROR_LOG>
   ```

3. Whether samples and checkpoints continued to appear before failure:

   ```bash
   find /shared/rc/defgengan/outputs/<EXPERIMENT_NAME> -maxdepth 2 -type f | sort
   ```

4. Whether memory usage rises with epoch count.

5. Whether plotting code closes figures after saving:

   ```python
   plt.close(fig)
   ```

6. Whether training functions are being retraced or models are being recompiled inside loops.

## Important interpretation

Do not assume that a larger RAM request fixes progressive memory growth. If 64 GB merely allows more epochs than 32 GB before the same failure, fix the training loop before requesting substantially more memory.

## Reproducibility rule

When changing the implementation to address a failure, preserve and record:

- dataset version
- random seed
- class counts and sampling strategy
- batch size
- epoch count
- generator and discriminator architecture
- optimizer settings
- checkpoint schedule
- selected fixed-latent evaluation samples

This makes it possible to distinguish an engineering fix from a change to the experiment itself.
