# Troubleshooting: RIT RC cDCGAN Training

This note records the main cluster failure modes encountered during the project and the implementation changes that stabilized training.

## Scope

The principal memory issue occurred in the TensorFlow/Keras conditional DCGAN pipeline used for 128×128 grayscale surface-defect generation.

Baseline configuration at the time of the early failures:

- three classes: normal, scratch, and spot;
- approximately 700 training exposures per class after balancing;
- batch size initially 32;
- 82 planned epochs;
- full image arrays loaded into memory in the early implementation;
- generated sample grids saved during training;
- checkpoints saved at selected epochs.

## Failure category 1: host-memory growth and OOM

### Verified attempts

| Job ID | Requested memory | MaxRSS | Elapsed | Final state | Observed progress |
|---|---:|---:|---:|---|---|
| `21134179` | 32G | 31.93G | 00:19:13 | `OUT_OF_MEMORY` | generated samples through approximately epoch 13 |
| `21140611` | 64G | 64.00G | 01:20:02 | `OUT_OF_MEMORY` | generated samples through approximately epoch 28 |
| `21152531` | 64G | 64.00G | 01:40:04 | `OUT_OF_MEMORY` | later memory-stability attempt; progress reached approximately epoch 15 in the retained discussion |
| `21165937` | 64G | 64.00G | 01:09:58 | `OUT_OF_MEMORY` | exact research variant `TBD` |

Increasing requested memory delayed some failures but did not reliably solve the underlying engineering problem.

### Engineering hypotheses investigated

The project investigated the following implementation-level hypotheses. They were plausible contributors, not formally proven root causes:

1. Repeatedly changing `discriminator.trainable` inside the batch loop while using compiled Keras models.
2. Repeated `train_on_batch()` execution and TensorFlow retracing.
3. Repeated prediction, model saving, and Matplotlib figure creation.
4. Keeping the complete prepared dataset in NumPy arrays.
5. Batch size 32 increasing activation and temporary-tensor pressure.
6. Graph or object accumulation across epochs.

The parameter count alone was not treated as a sufficient explanation because the failure pattern involved growth over elapsed training rather than an immediate model-allocation failure.

### Stabilization approach

The training pipeline was refactored while preserving the research objective:

- retained conditional labels and the same overall GAN objective;
- moved data handling toward `tf.data`;
- replaced the older `train_on_batch()` pattern with `tf.GradientTape`;
- used stable compiled training steps with `@tf.function` where applicable;
- avoided changing trainable state inside the inner loop;
- reduced visualization/checkpoint frequency;
- saved generator-only checkpoints during intermediate epochs where appropriate;
- used batch size 16 for later controlled experiments;
- explicitly closed Matplotlib figures;
- used garbage collection only as a supporting measure, not the main fix.

Later successful jobs used roughly 2–4G MaxRSS rather than exhausting 64G, demonstrating that implementation structure mattered more than simply increasing the memory request.

## Failure category 2: Spack/OpenSSL activation failure

### Verified failed attempt

Job `21443794` was the first Exp7 submission attempt:

```text
state: FAILED
exit code: 1:0
elapsed: 00:00:04
MaxRSS: 0.03G
stdout: empty
stderr: Spack/Python hashlib import failed because the activated environment's libcrypto did not satisfy OPENSSL_3.4.0
```

The Python training script did not start. This is an environment/setup failure, not a failed Exp7 research result.

### What worked

A fresh login shell showed `LD_LIBRARY_PATH` and `PYTHONPATH` unset. The existing Slurm file was then submitted directly without manually activating the Spack environment in the submission shell. Job `21443801` completed Exp7 successfully.

Recommended diagnostic sequence:

```bash
exit
ssh <RIT_USERNAME>@sporcsubmit.rc.rit.edu
printf 'LD_LIBRARY_PATH=%s\n' "${LD_LIBRARY_PATH-<UNSET>}"
printf 'PYTHONPATH=%s\n' "${PYTHONPATH-<UNSET>}"
env | grep '^SPACK_' || true
cd /shared/rc/defgengan/scripts
sbatch run_cdcgan_simple_scratch_exp7.slurm
```

Do not add speculative environment cleanup permanently unless a clean-shell submission still fails and current RC guidance supports the change.

The archived batch scripts activate:

```bash
spack env activate default-ml-x86_64-25052701
```

Reverify that environment before future reuse.

## Failure category 3: generic runtime failure

Job `21171574` failed with exit code `1:0` after 53 seconds during the normal-vs-scratch/Exp6 period. The audit proves the accounting outcome and retained logs, but not the exact root cause from the inventory alone. Record the cause as `TBD` unless the full log is reviewed.

## Practical diagnosis checklist

### 1. Check job state and accounting

```bash
sacct -j <JOB_ID> --format=JobID,JobName,State,ExitCode,Elapsed,ReqMem,MaxRSS
```

Interpret common states:

```text
COMPLETED       scheduler-level success
OUT_OF_MEMORY   cgroup/Slurm memory limit reached
FAILED          process exited nonzero
CANCELLED       cancelled before or during execution
TIMEOUT         requested wall time exhausted
PENDING         accepted but not yet started
```

### 2. Read both logs

```bash
tail -n 100 /shared/rc/defgengan/logs/<JOB_NAME>_<JOB_ID>.out
tail -n 100 /shared/rc/defgengan/logs/<JOB_NAME>_<JOB_ID>.err
```

TensorFlow often writes warnings and informational GPU messages to stderr. Do not classify the job as failed solely because `.err` is nonempty; use `sacct`, exit code, and the final stdout messages.

### 3. Verify artifact progression

```bash
find /shared/rc/defgengan/outputs/<EXPERIMENT_NAME> -maxdepth 3 -type f | sort
```

Empty final-model/loss directories can mean the job terminated before end-of-run saving, even when intermediate sample grids exist.

### 4. Check current and final memory

During or after a run:

```bash
sstat -j <JOB_ID>.batch --format=JobID,MaxRSS,AveRSS,MaxVMSize
sacct -j <JOB_ID> --format=JobID,State,ReqMem,MaxRSS
```

Availability of live statistics depends on cluster configuration.

### 5. Check plotting and checkpoint code

```python
plt.close(fig)
```

Avoid accumulating figures, predictions, model objects, or traced functions in loops.

### 6. Check retracing/recompilation

Verify that models or `@tf.function` training steps are not recreated repeatedly inside batch or epoch loops.

## Important interpretation

Do not assume a larger RAM request fixes progressive memory growth. When 64G merely allows more epochs before the same OOM state, fix the training/data pipeline before requesting substantially more memory.

Do not treat plausible metal texture or favorable GAN losses as proof of defect learning. Inspect fixed-latent and random-latent outputs by class.

## Reproducibility rule

When changing implementation to address a failure, preserve and record:

- dataset version;
- random seed;
- class counts and sampling strategy;
- batch size and per-class batch composition;
- epoch count and steps per epoch;
- generator/discriminator architecture;
- optimizer settings;
- checkpoint and visualization schedules;
- fixed-latent evaluation samples;
- code commit and script hash;
- Slurm job ID, resources, logs, state, and MaxRSS.

This distinguishes an engineering fix from a scientific experiment change.
