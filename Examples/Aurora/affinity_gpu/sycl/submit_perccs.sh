#!/bin/bash -l
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=0:15:00
#PBS -q debug
#PBS -A Catalyst
#PBS -l filesystems=home:flare

#cd ${PBS_O_WORKDIR}

# MPI example w/ 1 MPI rank per node w/ access to all GPU tiles
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=8
NDEPTH=1
NTHREADS=1

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

MPI_ARG="-n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth "

AFFINITY=""
AFFINITY="../../../../HelperScripts/Aurora/set_affinity_gpu_4ccs.sh"

COMMAND="mpiexec ${MPI_ARG} ${AFFINITY} ./hello_affinity"
echo "COMMAND= ${COMMAND}"
${COMMAND}
