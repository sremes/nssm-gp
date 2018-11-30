#!/bin/bash
#SBATCH --time=1-00          # 1 day
#SBATCH --mem-per-cpu=500    # .5G of memory
#SBATCH --cpus-per-task=8    # 8 cores for a task
#SBATCH --ntasks=1           # 1 tasks at a time
#SBATCH --nodes=1            # run all on same node

srun -J skin-sm python -u run-experiment.py --ard --kernel sm --data skin
