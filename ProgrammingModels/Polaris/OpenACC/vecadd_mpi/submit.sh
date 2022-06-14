#!/bin/sh
#COBALT -n 1 -t 15 -q full-node -A Catalyst

mpiexec -n 4 ./vecadd

