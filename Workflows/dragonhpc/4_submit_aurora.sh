#!/bin/bash -l
#PBS -A alcf_training
#PBS -l select=2
#PBS -N dragon_test
#PBS -l walltime=0:10:00
#PBS -l filesystems=home:flare
#PBS -k doe
#PBS -l place=scatter
#PBS -q debug

cd $PBS_O_WORKDIR

module load frameworks
# Replace this path with the path to your environment
source /grand/alcf_training/workflows/_env/bin/activate

dragon 0_dragon_pool.py
sleep 1
dragon 1_dragon_process_group.py
sleep 1
dragon 2_dragon_mpi_process_group.py
sleep 1
dragon 3_dragon_dictionary.py
