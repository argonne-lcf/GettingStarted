#!/bin/csh
#COBALT -n 2 -t 10 -q debug-cache-quad -A <project_name> -O LAMMPS

aprun -n 128 -N 64 -d 1 --cc depth -e OMP_NUM_THREADS=1 -j 1 ./lmp_theta -in lmp.in

