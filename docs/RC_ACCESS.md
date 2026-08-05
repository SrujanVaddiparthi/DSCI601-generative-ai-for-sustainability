# RIT Research Computing workflow

This document records the RIT Research Computing (RC) workflow used by this project. It is a handover record, not a substitute for current RC policy or onboarding documentation. Reverify hostnames, software environments, allocation status, and access requirements before reuse.

## Login

```bash
ssh <RIT_USERNAME>@sporcsubmit.rc.rit.edu
```

Use only an individually assigned RIT account. Never share passwords, multifactor-authentication access, SSH private keys, tokens, or active sessions.

## Project context

Slurm account/project:

```text
defgengan
```

Shared project storage:

```text
/shared/rc/defgengan
```

Verified shared layout:

```text
/shared/rc/defgengan/
├── data/
├── scripts/
├── logs/
└── outputs/
```

The shared root and its four main subdirectories were group-accessible through the project group recorded by RC. A successor must still be added to the appropriate RC project and allocations before access should be expected.

## Shared project storage versus personal home storage

Use `/shared/rc/defgengan` for project data, scripts, logs, outputs, and handover artifacts that future authorized project members need.

```text
/shared/rc/defgengan   shared project storage
/home/<RIT_USERNAME>   personal account storage
```

Files under `/home/<RIT_USERNAME>` do not automatically transfer to another researcher. The audit found a personal copy at:

```text
/home/<RIT_USERNAME>/exp3_3class_fullrun
```

The same experiment family is also represented under shared project storage. Treat the home-directory copy as personal/legacy unless separately verified and intentionally archived.

## Archived Slurm defaults

The archived project Slurm scripts used:

```text
partition: tier3
account: defgengan
GPU request: gpu:a100:1
CPUs per task: 8
memory: 64G
wall time: 0-05:00:00
```

Representative directives:

```bash
#SBATCH --account=defgengan
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=0-05:00:00
#SBATCH --output=/shared/rc/defgengan/logs/%x_%j.out
#SBATCH --error=/shared/rc/defgengan/logs/%x_%j.err
```

These values describe the archived scripts, not a permanent entitlement. Confirm current limits and permitted resources before submitting new work.

## Software environment

Archived scripts activated:

```bash
spack env activate default-ml-x86_64-25052701
```

This environment was used by the archived project scripts and by successful jobs, but it must be reverified before reuse. The evidence confirms TensorFlow imports in the project scripts; it does not prove a single authoritative Python, TensorFlow, CUDA, or cuDNN version for all successful batch jobs. Record those values as `TBD` unless independently captured from a new job.

### Submission process that worked

A clean login shell followed by direct `sbatch` submission worked for the successful Exp7 retry. Do not manually activate the Spack environment before `sbatch` unless current RC documentation specifically requires it; the batch script already performs activation.

```bash
ssh <RIT_USERNAME>@sporcsubmit.rc.rit.edu
cd /shared/rc/defgengan/scripts
sbatch run_file.slurm
```

Then monitor with:

```bash
squeue --me
sacct -j <JOB_ID> --format=JobID,JobName,State,ExitCode,Elapsed,ReqMem,MaxRSS
```

Do not train on the submit node.

## Common commands

```bash
my-accounts
sbatch run_file.slurm
squeue --me
squeue -j <JOB_ID> --start
scontrol show job <JOB_ID>
sacct -j <JOB_ID> --format=JobID,JobName,State,ExitCode,Elapsed,Start,End,ReqMem,MaxRSS
tail -f /shared/rc/defgengan/logs/<LOG_FILE>.out
tail -n 100 /shared/rc/defgengan/logs/<LOG_FILE>.err
scancel <JOB_ID>
```

Keep multiline shell commands syntactically valid. A backslash must be the final character on the line; otherwise, place the command on one line.

## Known retained output locations

```text
/shared/rc/defgengan/outputs/exp3_3class_baseline
/shared/rc/defgengan/outputs/exp3_3class_fullrun
/shared/rc/defgengan/outputs/exp3_3class_fullrun_v2
/shared/rc/defgengan/outputs/exp4_normal_vs_scratch
/shared/rc/defgengan/outputs/exp5_normal_vs_scratch_stratified
/shared/rc/defgengan/outputs/exp6_normal_vs_scratch_final
/shared/rc/defgengan/outputs/exp_spot_1_baseline_normal_vs_spot
/shared/rc/defgengan/outputs/exp_spot_2_fixed_enhancement_normal_vs_spot
/shared/rc/defgengan/outputs/exp_spot_3_stratified_randomized
/shared/rc/defgengan/outputs/exp_spot_4_stratified_randomized_augmented
/shared/rc/defgengan/outputs/exp7_simple_scratch_geometry
/shared/rc/defgengan/outputs/exp8_simple_scratch_100N_700S
```

Directory presence proves that these output roots existed at audit time. It does not by itself prove that every subartifact is complete or that every directory corresponds to exactly one Slurm attempt.

## Collaborator onboarding

Professor Abu Islam or another authorized project manager should onboard each collaborator through their own RC identity:

1. The collaborator obtains or confirms their individual RIT RC access.
2. An authorized project manager adds that person to the `defgengan` ColdFront project.
3. The person is added to the relevant project-directory/storage and SPORC/Slurm allocations, as required by current RC procedures.
4. The collaborator verifies access to `/shared/rc/defgengan` and the `defgengan` Slurm account.
5. The collaborator submits jobs using their own username and authentication.

Do not transfer access by sharing the former maintainer's credentials or keys.

Exact ColdFront ownership, manager permissions, allocation expiration dates, renewal authority, and the current onboarding interface were not proven by the audit and must be confirmed with RIT Research Computing.

## Retaining or extending access

Personal RC access and project-allocation availability are separate questions. Before eligibility or an allocation expires:

1. Confirm the user's continuing RIT affiliation and RC eligibility.
2. Confirm the `defgengan` project and its storage/cluster allocations remain active.
3. Ask the project PI or authorized manager to request the appropriate extension or membership update.
4. Contact RIT Research Computing when the ColdFront interface, ownership, renewal rights, or eligibility rules are unclear.

Do not assume that project membership alone extends an individual's account eligibility.

## Questions that require RIT Research Computing

Contact RC for authoritative answers about:

- current login host or VPN/network requirements;
- whether `default-ml-x86_64-25052701` still exists and is supported;
- current partition, QoS, GPU, memory, and wall-time limits;
- ColdFront project ownership and manager permissions;
- collaborator onboarding steps and allocation membership;
- project/allocation expiration and renewal;
- post-graduation or post-employment eligibility;
- retention and transfer policy for personal home-directory data;
- recovery of older accounting records not present in the audit.
