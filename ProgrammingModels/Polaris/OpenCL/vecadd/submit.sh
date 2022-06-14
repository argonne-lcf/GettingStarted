#!/bin/sh
#COBALT -n 1 -t 15 -q single-gpu -A Catalyst

mpiexec -np 1 ./vecadd

