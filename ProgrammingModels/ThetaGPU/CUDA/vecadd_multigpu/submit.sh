#!/bin/sh
#COBALT -n 1 -t 15 -q single-gpu -A Catalyst

# run using all available GPUs
mpirun -n 1 ./vecadd


# run on only first two GPUs
#export CUDA_VISIBLE_DEVICES=0,1
#mpirun -n 1 ./vecadd

