#!/bin/bash -l
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q debug
#PBS -A Aurora_deployment

cd ${PBS_O_WORKDIR}

source $HOME/_env/bin/activate
python my_parsl_workflow.py
