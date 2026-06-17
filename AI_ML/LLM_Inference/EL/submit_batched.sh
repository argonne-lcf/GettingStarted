#!/bin/bash -l
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -q debug
#PBS -A datascience
#PBS -l filesystems=home:flare

cd $PBS_O_WORKDIR

rm -r logs/* script_logs/* ckpt_* ./.actor_ckpt ./vllm_cache

module load mpifileutils

source /path/to/your/venv

python3 EL_batched_inference.py --num-prompts 32