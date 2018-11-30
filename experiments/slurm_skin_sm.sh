#!/bin/bash
#SBATCH --time=0-12:0:0      # 1 day
#SBATCH --mem-per-cpu=500    # .5G of memory
#SBATCH --cpus-per-task=8    # 8 cores for a task
#SBATCH --ntasks=1           # 1 tasks at a time
#SBATCH --nodes=1            # run all on same node
#SBATCH --array=1-12         # run for different q
#SBATCH --output=logs/skin_sm_q_%a.log # name logs

n=$SLURM_ARRAY_TASK_ID                  # define n  
line=`sed "${n}q;d" sm_params.txt`      # get n:th line (1-indexed) of the file 

export OMP_NUM_THREADS=8

srun -J skin-sm-$SLURM_ARRAY_TASK_ID python -u run-experiment.py $line
