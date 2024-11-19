#!/bin/bash -l
#PBS -l select=2:system=crux
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q debug 
#PBS -A Catalyst
#PBS -l filesystems=home:grand:eagle

#cd ${PBS_O_WORKDIR}

# MPI example w/ 32 MPI ranks per node and 8 OpenMP threads per rank. Each MPI rank spans 2 physical cores and utilizes all 4 hyperthreads on each core.
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=32
NCORES_PER_RANK=2
NTHREADS=8
STRIDE=64

#Generate cpu bind list during job (or create it beforehand and paste here)
CPU_LIST="`python3 ../../../HelperScripts/Crux/cpu_binding.py 0 ${NRANKS_PER_NODE} ${NCORES_PER_RANK} ${NTHREADS} ${STRIDE}`"
echo "CPU_LIST= ${CPU_LIST}"

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

MPI_ARGS="-n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --cpu-bind list:${CPU_LIST} "
OMP_ARGS="--env OMP_NUM_THREADS=${NTHREADS} --env OMP_PROC_BIND=true "

mpiexec ${MPI_ARGS} ${OMP_ARGS} ./hello_affinity
