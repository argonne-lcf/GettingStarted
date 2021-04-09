#!/bin/sh
#COBALT -n 1 -t 15 -q debug -A Catalyst

mpirun -np 1 ./vecadd

