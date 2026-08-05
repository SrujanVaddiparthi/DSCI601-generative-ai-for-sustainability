# RIT RC Experiment Run Ledger

This ledger separates **research experiments** from **individual Slurm attempts**. A research experiment may have multiple submissions: setup failures, out-of-memory failures, cancelled duplicates, successful reruns, or continuations.

The ledger omits credentials and uses `<RIT_USERNAME>` in public commands. Unknown values are marked `TBD` rather than inferred.

## Shared project layout

```text
/shared/rc/defgengan/
├── data/
├── scripts/
├── logs/
└── outputs/
```

## Archived resource convention

The retained Slurm files use the following default configuration:

```text
partition: tier3
account: defgengan
GPU: A100 (`--gres=gpu:a100:1`)
CPUs: 8
memory: 64G
wall time: 5 hours
```

Early three-class attempts used 32G or 64G as shown below. The audit's `sacct` export truncates `ReqTRES`; the GPU request for historical attempts is therefore taken from the associated archived Slurm pattern only where the experiment/script relationship is verified. Otherwise it is `TBD`.

# Experiment-to-attempt mapping

| Research experiment | Attempt relationship | Verified Slurm attempt(s) | Status |
|---|---|---|---|
| Three-class debug/baseline | short validation run | `21134091` | completed |
| Three-class full run, pre-refactor | 32G full attempt | `21134179` | OOM |
| Three-class full run, pre-refactor retry | 64G retry | `21140611` | OOM |
| Three-class full run, later memory test | later 64G attempt | `21152531` | OOM |
| Three-class stabilized implementation | successful post-refactor run | `21166473` is the strongest audit match; exact experiment-to-job mapping remains `TBD` because the generic job name was reused | completed accounting record; mapping TBD |
| Normal-vs-scratch Exp4 | research experiment completed; output retained | job ID `TBD` | completed artifacts retained |
| Normal-vs-scratch Exp5 | research experiment completed; output retained | job ID `TBD` | completed artifacts retained |
| Normal-vs-scratch Exp6 | research experiment completed; output retained | job ID `TBD` | completed artifacts retained |
| Normal-vs-spot Exp1 | research experiment completed; output retained | job ID `TBD` | completed artifacts retained |
| Normal-vs-spot Exp2 | research experiment completed; output retained | job ID `TBD` | completed artifacts retained |
| Normal-vs-spot Exp3 | research experiment completed; output retained | job ID `TBD` | completed artifacts retained |
| Normal-vs-spot Exp4 | research experiment completed; output retained | job ID `TBD` | completed artifacts retained |
| Exp7 balanced procedural scratch | first setup attempt, then successful clean-shell retry | `21443794`, `21443801` | failed before Python; then completed |
| Exp8 scratch-heavy procedural scratch | successful training run | `21443819` | completed |

## Verified Slurm attempt ledger

One row is used for each top-level Slurm attempt. `MaxRSS` is the batch-step value from `sacct` when present.

| Job ID | Job name | Experiment / attempt type | Python script | Slurm file | Partition | Account | GPU | CPUs | Req. memory | Wall time | Final state | Exit code | Elapsed | MaxRSS | Output folder | Logs |
|---|---|---|---|---|---|---|---|---:|---:|---:|---|---|---:|---:|---|---|
| `21134091` | `cdcgan3dbg` | Three-class debug/baseline validation | `train_cdcgan_3class_cluster.py` or immediate archived predecessor; exact submitted file `TBD` | historical `run_cdcgan_3class.slurm`; exact archived revision `TBD` | `tier3` | `defgengan` | A100 request supported by contemporaneous project workflow; exact submitted directive `TBD` | 8 | 32G | 1 hour was shown for this historical job in prior job detail; archived revision `TBD` | `COMPLETED` | `0:0` | 00:10:29 | 10.51G | `/shared/rc/defgengan/outputs/exp3_3class_baseline` | `cdcgan3dbg_21134091.out`, `cdcgan3dbg_21134091.err` |
| `21134179` | `cdcgan3dbg` | Three-class full run; early 32G OOM attempt | `train_cdcgan_3class_cluster.py` | historical `run_cdcgan_3class.slurm`; exact revision `TBD` | `tier3` | `defgengan` | A100 request supported by contemporaneous workflow; exact submitted directive `TBD` | 8 | 32G | 1 hour in the historical job configuration; exact file revision `TBD` | `OUT_OF_MEMORY` | `0:125` | 00:19:13 | 31.93G | `/shared/rc/defgengan/outputs/exp3_3class_fullrun` | `cdcgan3dbg_21134179.out`, `cdcgan3dbg_21134179.err` |
| `21140611` | `cdcgan3dbg` | Three-class full run; 64G retry | `train_cdcgan_3class_cluster.py` | historical `run_cdcgan_3class.slurm`; exact revision `TBD` | `tier3` | `defgengan` | A100 request supported by contemporaneous workflow; exact submitted directive `TBD` | 8 | 64G | 2 hours in the submitted historical revision discussed during the run | `OUT_OF_MEMORY` | `0:125` | 01:20:02 | 64.00G | `/shared/rc/defgengan/outputs/exp3_3class_fullrun` | `cdcgan3dbg_21140611.out`, `cdcgan3dbg_21140611.err` |
| `21140613` | `cdcgan3dbg` | Accidental duplicate of the 64G retry; cancelled while pending | same as `21140611` | same submitted file as `21140611` | `tier3` | `defgengan` | same request as duplicate submission | 0 allocated | 64G | 2 hours | `CANCELLED` | `0:0` | 00:00:00 | n/a | none expected; job did not run | no retained job log found in audit inventory |
| `21152531` | `cdcgan3dbg` | Three-class later 64G memory-stability attempt | exact submitted script revision `TBD`; conversation identifies batch-size-16 memory-stability work | `run_cdcgan_3class.slurm`, historical revision | `tier3` | `defgengan` | A100 request supported by workflow | 8 | 64G | `TBD` | `OUT_OF_MEMORY` | `0:125` | 01:40:04 | 64.00G | `/shared/rc/defgengan/outputs/exp3_3class_fullrun` | `cdcgan3dbg_21152531.out`, `cdcgan3dbg_21152531.err` |
| `21165937` | `cdcgan3dbg` | Additional three-class OOM attempt; exact research variant `TBD` | `TBD` | `TBD` | `tier3` | `defgengan` | `TBD` | 8 | 64G | `TBD` | `OUT_OF_MEMORY` | `0:125` | 01:09:58 | 64.00G | exact output mapping `TBD` | `cdcgan3dbg_21165937.out`, `cdcgan3dbg_21165937.err` |
| `21166425` | `cdcgan3dbg` | Successful stabilized-era run; exact experiment mapping `TBD` | `TBD` | `run_cdcgan_3class.slurm` or contemporaneous revision; exact mapping `TBD` | `tier3` | `defgengan` | `TBD` | 8 | 64G | likely 5 hours from archived revision; exact submitted file `TBD` | `COMPLETED` | `0:0` | 00:05:30 | 4.03G | exact output mapping `TBD` | `cdcgan3dbg_21166425.out`, `cdcgan3dbg_21166425.err` |
| `21166473` | `cdcgan3dbg` | Successful stabilized three-class candidate; exact mapping not proven by generic job name | `train_cdcgan_3class_cluster_v2.py` is the strongest chronological match; submitted file `TBD` | `run_cdcgan_3class.slurm` | `tier3` | `defgengan` | A100 in archived Slurm file | 8 | 64G | 5 hours in archived file | `COMPLETED` | `0:0` | 00:03:12 | 2.18G | `/shared/rc/defgengan/outputs/exp3_3class_fullrun_v2` is the strongest matching retained root; mapping `TBD` | `cdcgan3dbg_21166473.out`, `cdcgan3dbg_21166473.err` |
| `21166619` | `cdcgan3dbg` | Successful run; exact experiment mapping `TBD` | `TBD` | `TBD` | `tier3` | `defgengan` | `TBD` | 8 | 64G | `TBD` | `COMPLETED` | `0:0` | 00:06:34 | 4.04G | `TBD` | `cdcgan3dbg_21166619.out`, `cdcgan3dbg_21166619.err` |
| `21166673` | `cdcgan3dbg` | Successful run in normal-vs-scratch period; exact Exp4 mapping `TBD` | `train_cdcgan_normalVscratch_cluster_v2.py` is a chronological candidate; submitted file `TBD` | historical normal-vs-scratch Slurm file; exact revision `TBD` | `tier3` | `defgengan` | `TBD` | 8 | 64G | `TBD` | `COMPLETED` | `0:0` | 00:10:21 | 4.20G | possible `/shared/rc/defgengan/outputs/exp4_normal_vs_scratch`; mapping `TBD` | `cdcgan3dbg_21166673.out`, `cdcgan3dbg_21166673.err` |
| `21167939` | `cdcgan3dbg` | Successful normal-vs-scratch-era run; exact experiment mapping `TBD` | `TBD` | `TBD` | `tier3` | `defgengan` | `TBD` | 8 | 64G | `TBD` | `COMPLETED` | `0:0` | 00:05:47 | 4.13G | `TBD` | `cdcgan3dbg_21167939.out`, `cdcgan3dbg_21167939.err` |
| `21167977` | `cdcgan3dbg` | Successful normal-vs-scratch-era run; possible Exp5, mapping `TBD` | `train_cdcgan_normalVscratch_cluster_v3_stratified.py` is a chronological candidate; submitted file `TBD` | historical normal-vs-scratch Slurm file; exact revision `TBD` | `tier3` | `defgengan` | `TBD` | 8 | 64G | `TBD` | `COMPLETED` | `0:0` | 00:09:18 | 2.35G | possible `/shared/rc/defgengan/outputs/exp5_normal_vs_scratch_stratified`; mapping `TBD` | `cdcgan3dbg_21167977.out`, `cdcgan3dbg_21167977.err` |
| `21171574` | `cdcgan3dbg` | Failed setup/runtime attempt during Exp6 period; exact failure category `TBD` | `TBD` | `run_cdcgan_normalVscratch.slurm` or contemporaneous revision; exact mapping `TBD` | `tier3` | `defgengan` | `TBD` | 8 | 64G | `TBD` | `FAILED` | `1:0` | 00:00:53 | 1.56G | `TBD` | `cdcgan3dbg_21171574.out`, `cdcgan3dbg_21171574.err` |
| `21171584` | `cdcgan3dbg` | Successful run during Exp6 period; exact mapping `TBD` | `train_cdcgan_normalVscratch_cluster_v4_exp6.py` is the strongest chronological match; submitted file `TBD` | `run_cdcgan_normalVscratch.slurm` | `tier3` | `defgengan` | A100 in archived Slurm file | 8 | 64G | 5 hours in archived file | `COMPLETED` | `0:0` | 00:09:58 | 2.47G | possible `/shared/rc/defgengan/outputs/exp6_normal_vs_scratch_final`; mapping `TBD` | `cdcgan3dbg_21171584.out`, `cdcgan3dbg_21171584.err` |
| `21171641` | `cdcgan3dbg` | Successful normal-vs-spot-era run; possible Exp1, mapping `TBD` | `train_cdcgan_normalVspot_cluster_v1_baseline.py` is a chronological candidate | historical normal-vs-spot Slurm file; exact revision `TBD` | `tier3` | `defgengan` | `TBD` | 8 | 64G | `TBD` | `COMPLETED` | `0:0` | 00:09:38 | 2.80G | possible `exp_spot_1_baseline_normal_vs_spot`; mapping `TBD` | `cdcgan3dbg_21171641.out`, `cdcgan3dbg_21171641.err` |
| `21172528` | `cdcgan3dbg` | Successful normal-vs-spot-era run; possible Exp2, mapping `TBD` | `train_cdcgan_normalVspot_cluster_v2_fixed_enhancement.py` is a chronological candidate | historical normal-vs-spot Slurm file; exact revision `TBD` | `tier3` | `defgengan` | `TBD` | 8 | 64G | `TBD` | `COMPLETED` | `0:0` | 00:11:34 | 4.15G | possible `exp_spot_2_fixed_enhancement_normal_vs_spot`; mapping `TBD` | `cdcgan3dbg_21172528.out`, `cdcgan3dbg_21172528.err` |
| `21172534` | `cdcgan3dbg` | Successful normal-vs-spot-era run; possible Exp3, mapping `TBD` | `train_cdcgan_normalVspot_cluster_v3_stratified.py` is a chronological candidate | historical normal-vs-spot Slurm file; exact revision `TBD` | `tier3` | `defgengan` | `TBD` | 8 | 64G | `TBD` | `COMPLETED` | `0:0` | 00:10:56 | 2.41G | possible `exp_spot_3_stratified_randomized`; mapping `TBD` | `cdcgan3dbg_21172534.out`, `cdcgan3dbg_21172534.err` |
| `21172541` | `cdcgan3dbg` | Successful normal-vs-spot-era run; possible Exp4, mapping `TBD` | `train_cdcgan_normalVspot_cluster_v4_augmented.py` is a chronological candidate | `run_cdcgan_normalVspot.slurm` | `tier3` | `defgengan` | A100 in archived Slurm file | 8 | 64G | 5 hours in archived file | `COMPLETED` | `0:0` | 00:11:28 | 2.39G | possible `exp_spot_4_stratified_randomized_augmented`; mapping `TBD` | `cdcgan3dbg_21172541.out`, `cdcgan3dbg_21172541.err` |
| `21443794` | `simpleScratch` | Exp7 environment/setup failure before Python execution | `train_cdcgan_simple_scratch_exp7.py` was intended but did not start | `run_cdcgan_simple_scratch_exp7.slurm` | `tier3` | `defgengan` | A100 | 8 | 64G | 5 hours | `FAILED` | `1:0` | 00:00:04 | 0.03G | intended `/shared/rc/defgengan/outputs/exp7_simple_scratch_geometry`; no training result from this attempt | `simpleScratch_21443794.out`, `simpleScratch_21443794.err` |
| `21443801` | `simpleScratch` | Exp7 successful clean-shell retry | `train_cdcgan_simple_scratch_exp7.py` | `run_cdcgan_simple_scratch_exp7.slurm` | `tier3` | `defgengan` | A100 | 8 | 64G | 5 hours | `COMPLETED` | `0:0` | 00:06:09 | 4.14G | `/shared/rc/defgengan/outputs/exp7_simple_scratch_geometry` | `simpleScratch_21443801.out`, `simpleScratch_21443801.err` |
| `21443819` | `scratchHeavy` | Exp8 successful scratch-heavy run | `train_cdcgan_simple_scratch_exp8.py` | `run_cdcgan_simple_scratch_exp8.slurm` | `tier3` | `defgengan` | A100 | 8 | 64G | 5 hours | `COMPLETED` | `0:0` | 00:05:05 | 4.09G | `/shared/rc/defgengan/outputs/exp8_simple_scratch_100N_700S` | `scratchHeavy_21443819.out`, `scratchHeavy_21443819.err` |

## Research experiment inventory

These experiment definitions and output roots are verified from retained scripts/directories even when the exact job mapping is not.

| Experiment | Script | Slurm file | Epochs | Verified output root | Job ID |
|---|---|---|---:|---|---|
| Three-class stabilized | `train_cdcgan_3class_cluster_v2.py` | `run_cdcgan_3class.slurm` | 82 | `/shared/rc/defgengan/outputs/exp3_3class_fullrun_v2` | `TBD` |
| Normal-vs-scratch Exp4 | `train_cdcgan_normalVscratch_cluster_v2.py` | historical revision of `run_cdcgan_normalVscratch.slurm` | 200 | `/shared/rc/defgengan/outputs/exp4_normal_vs_scratch` | `TBD` |
| Normal-vs-scratch Exp5 | `train_cdcgan_normalVscratch_cluster_v3_stratified.py` | historical revision of `run_cdcgan_normalVscratch.slurm` | 200 | `/shared/rc/defgengan/outputs/exp5_normal_vs_scratch_stratified` | `TBD` |
| Normal-vs-scratch Exp6 | `train_cdcgan_normalVscratch_cluster_v4_exp6.py` | `run_cdcgan_normalVscratch.slurm` | 200 | `/shared/rc/defgengan/outputs/exp6_normal_vs_scratch_final` | `TBD` |
| Normal-vs-spot Exp1 | `train_cdcgan_normalVspot_cluster_v1_baseline.py` | historical revision of `run_cdcgan_normalVspot.slurm` | 200 | `/shared/rc/defgengan/outputs/exp_spot_1_baseline_normal_vs_spot` | `TBD` |
| Normal-vs-spot Exp2 | `train_cdcgan_normalVspot_cluster_v2_fixed_enhancement.py` | historical revision of `run_cdcgan_normalVspot.slurm` | 200 | `/shared/rc/defgengan/outputs/exp_spot_2_fixed_enhancement_normal_vs_spot` | `TBD` |
| Normal-vs-spot Exp3 | `train_cdcgan_normalVspot_cluster_v3_stratified.py` | historical revision of `run_cdcgan_normalVspot.slurm` | 200 | `/shared/rc/defgengan/outputs/exp_spot_3_stratified_randomized` | `TBD` |
| Normal-vs-spot Exp4 | `train_cdcgan_normalVspot_cluster_v4_augmented.py` | `run_cdcgan_normalVspot.slurm` | 200 | `/shared/rc/defgengan/outputs/exp_spot_4_stratified_randomized_augmented` | `TBD` |
| Exp7 balanced procedural scratch | `train_cdcgan_simple_scratch_exp7.py` | `run_cdcgan_simple_scratch_exp7.slurm` | 82 | `/shared/rc/defgengan/outputs/exp7_simple_scratch_geometry` | `21443794` failed setup; `21443801` completed |
| Exp8 scratch-heavy procedural scratch | `train_cdcgan_simple_scratch_exp8.py` | `run_cdcgan_simple_scratch_exp8.slurm` | 82 | `/shared/rc/defgengan/outputs/exp8_simple_scratch_100N_700S` | `21443819` |

## Interpretation rules

- A completed `sacct` row proves that the Slurm job exited successfully; it does not by itself prove scientific quality.
- A retained output directory proves directory presence at audit time; it does not prove that every file is complete.
- Generic job name `cdcgan3dbg` was reused, so chronological script/job matching is not sufficient to convert `TBD` mappings into verified facts.
- `OUT_OF_MEMORY` records are engineering outcomes, not research conclusions.
- Job `21443794` is not an Exp7 scientific failure. It failed during environment activation before Python execution; Exp7 later completed under job `21443801`.
