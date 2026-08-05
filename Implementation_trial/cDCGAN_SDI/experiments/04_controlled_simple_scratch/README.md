# Controlled simple-scratch cDCGAN experiment

## Goal

Test whether the existing binary cDCGAN can learn simple scratch geometry when
complex metal-surface texture is removed.

This is deliberately one controlled experiment, not an architecture comparison.

## Files

- `generate_simple_scratch_dataset.py`
- `train_cdcgan_simple_scratch_exp7.py`
- `run_cdcgan_simple_scratch_exp7.slurm`

## 1. Generate the dataset locally

From the folder containing the generator:

```bash
python generate_simple_scratch_dataset.py \
  --out simple_scratch_v1 \
  --count-per-class 700 \
  --seed 42 \
  --clean
```

Inspect:

- `simple_scratch_v1/dataset_preview.png`
- `simple_scratch_v1/metadata.csv`
- `simple_scratch_v1/dataset_summary.json`

The folder should contain:

```text
simple_scratch_v1/
├── normal/       # 700 unique images
├── scratches/    # 700 unique images
├── metadata.csv
├── dataset_preview.png
└── dataset_summary.json
```

## 2. Upload to RIT RC

```bash
scp -r simple_scratch_v1 \
  <RIT_USERNAME>@sporcsubmit.rc.rit.edu:/shared/rc/defgengan/data/

scp train_cdcgan_simple_scratch_exp7.py \
  <RIT_USERNAME>@sporcsubmit.rc.rit.edu:/shared/rc/defgengan/scripts/

scp run_cdcgan_simple_scratch_exp7.slurm \
  <RIT_USERNAME>@sporcsubmit.rc.rit.edu:/shared/rc/defgengan/scripts/
```

## 3. Submit

```bash
ssh <RIT_USERNAME>@sporcsubmit.rc.rit.edu
cd /shared/rc/defgengan/scripts
sbatch run_cdcgan_simple_scratch_exp7.slurm
squeue --me
```

## 4. Watch logs

Replace `JOBID`:

```bash
tail -f /shared/rc/defgengan/logs/simpleScratch_JOBID.out
tail -f /shared/rc/defgengan/logs/simpleScratch_JOBID.err
```

## 5. Check outputs

```bash
ls -R /shared/rc/defgengan/outputs/exp7_simple_scratch_geometry
```

Important folders:

```text
generated_samples/
├── fixed_z/      # same latent inputs at every checkpoint
└── random_z/     # fresh latent inputs for diversity inspection

checkpoints/
final_models/
losses/
```

## 6. Download results

Run from the laptop:

```bash
scp -r \
  <RIT_USERNAME>@sporcsubmit.rc.rit.edu:/shared/rc/defgengan/outputs/exp7_simple_scratch_geometry \
  .
```

## Tonight's recommended scope

1. Generate and inspect the dataset.
2. Upload the three experiment files and dataset.
3. Submit the 82-epoch job.
4. Inspect epochs 5, 10, 21, 42, and 82.
5. Compare fixed-z progression and random-z diversity.
6. Record one of three conclusions:

   - coherent and varied scratches: background complexity was a major bottleneck;
   - coherent but repetitive scratches: class learning works, diversity remains limiting;
   - no coherent scratches: debug the controlled baseline before deciding to pivot.

Do not add texture, masks, diffusion, or a latent-dimension ablation tonight.
