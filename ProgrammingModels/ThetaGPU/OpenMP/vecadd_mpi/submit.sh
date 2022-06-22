#!/bin/sh
#COBALT -n 1 -t 15 -q full-node -A Catalyst

mpirun -n 8 ./vecadd

