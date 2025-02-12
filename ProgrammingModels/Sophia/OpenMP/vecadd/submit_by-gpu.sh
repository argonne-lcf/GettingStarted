#!/bin/bash -l
#PBS -l select=1:system=sophia
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q by-gpu 
#PBS -A Catalyst
#PBS -l filesystems=home:grand:eagle

#cd ${PBS_O_WORKDIR}

# Example using 1 GPU and 16 cores (1 thread per core)
NNODES=`wc -l < $PBS_NODEFILE`

# 1 MPI rank per GPU
NRANKS_PER_NODE=1
NTHREADS=4

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

MPI_ARG="-n ${NTOTRANKS} "
MPI_ARG+="--bind-to core --map-by ppr:${NRANKS_PER_NODE}:numa:PE=${NTHREADS}  "
MPI_ARG+=" -x OMP_NUM_THREADS=${NTHREADS} -x OMP_PROC_BIND=spread -x OMP_PLACES=cores "

AFFINITY=""
AFFINITY="../../../../HelperScripts/Sophia/set_affinity_gpu.sh "

COMMAND="mpiexec ${MPI_ARG} ${AFFINITY} ./vecadd"
echo "COMMAND= ${COMMAND}"
${COMMAND}
