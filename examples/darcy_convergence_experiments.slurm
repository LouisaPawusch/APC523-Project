#!/bin/bash
#SBATCH --job-name=runscript  # create a short name for your job
#SBATCH --output=slurm-%A.out # stdout file, these will be indexed by the job array ID
#SBATCH --error=slurm-%A.err  # stderr file, these will be indexed by the job array ID
#SBATCH --nodes=1             # node count
#SBATCH --ntasks=1            # number of tasks per node (48 cpus on node 1 (with 4 gpus), 24 on node 2 (with 2 gpus))
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is  default)
#SBATCH --time=01:00:00          # total run time limit (HH:MM:SS)

module purge 

## ADD ACTIVATION OF ENVIRONMENT HERE (e.g. source ../.venv/bin/activate)
source ../.venv/bin/activate
python3 darcy_convergence_experiments.py
