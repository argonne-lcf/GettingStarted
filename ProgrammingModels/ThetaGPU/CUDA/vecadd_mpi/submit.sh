#!/bin/sh
#COBALT -n 1 -t 15 -q single-gpu -A Catalyst

mpirun -n 16 ./vecadd

