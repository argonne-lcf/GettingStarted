#!/bin/bash -l
#PBS -l select=2
#PBS -l walltime=00:30:00
#PBS -q debug
#PBS -A datascience
#PBS -l filesystems=home:flare

cd $PBS_O_WORKDIR

rm -r logs/* script_logs/* ckpt_* ./.actor_ckpt ./vllm_cache

module load mpifileutils

source /path/to/your/venv

ACTORS_PER_PROCESS=96
python3 EL_request_driven.py --num-prompts 32 --ngpus-per-model 1 --actors-per-process ${ACTORS_PER_PROCESS} --send-timeout 10.0 --send-retries 5
rm -r logs_${ACTORS_PER_PROCESS} script_logs_${ACTORS_PER_PROCESS}
mv logs logs_${ACTORS_PER_PROCESS}
mv script_logs script_logs_${ACTORS_PER_PROCESS}
