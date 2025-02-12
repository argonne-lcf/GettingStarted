#!/bin/bash -l
#PBS -l select=8:system=sophia
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q by-node
#PBS -A Catalyst
#PBS -l filesystems=home:grand:eagle

cd ${PBS_O_WORKDIR}

# Example using all 8 GPUs and 128 cores (1 thread per core)
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=2
NDEPTH=64
NTHREADS=64

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

MPI_ARG="-n ${NTOTRANKS} --npernode ${NRANKS_PER_NODE} "
MPI_ARG=" -x OMP_NUM_THREADS=${NTHREADS} -x OMP_PROC_BIND=spread -x OMP_PLACES=cores "

mpiexec ${MPI_ARG} ./hello_affinity
