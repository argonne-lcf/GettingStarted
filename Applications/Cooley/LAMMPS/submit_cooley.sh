#!/bin/sh

qsub -n 1 -t 10 -A Catalyst ./run_lammps.sh


