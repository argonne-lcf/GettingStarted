#!/bin/sh
#COBALT -n 1 -t 15 -q full-node -A Catalyst

mpirun -hostfile ${COBALT_NODEFILE} -n 16 -N 16 ./vecadd

