# RIT Research Computing workflow

This records the workflow used in the project. Verify current RIT RC onboarding before handing access to a new researcher.

## Login

```bash
ssh RIT_USERNAME@sporcsubmit.rc.rit.edu
```

## Project context

Account/project:

```text
defgengan
```

Shared storage:

```text
/shared/rc/defgengan
```

Environment used successfully:

```bash
spack env activate default-ml-x86_64-25052701
```

Confirm that this environment still exists and is accessible.

## Commands

```bash
my-accounts
sbatch run_file.slurm
squeue --me
sacct -j JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed,Start,End
tail -f log_file.out
scancel JOB_ID
```

Do not train on the submit node. Do not share passwords, Duo access, SSH keys, or personal credentials.

## Known output locations

Exp7:

```text
/shared/rc/defgengan/outputs/exp7_simple_scratch_geometry
```

Exp8:

```text
/shared/rc/defgengan/outputs/exp8_simple_scratch_100N_700S
```
