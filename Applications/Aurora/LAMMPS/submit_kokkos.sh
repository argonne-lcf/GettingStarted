#!/bin/bash -l
#PBS -l select=1:system=aurora
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q debug
#PBS -A Catalyst
#PBS -l filesystems=home:flare

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=12
NDEPTH=4
NTHREADS=1

export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_NUM_THREADS=${NTHREADS}
export OMP_PLACES=cores

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))

INPUT=in.lj
EXE=/home/knight/lammps/lammps-git/src/lmp_aurora_kokkos

EXE_ARG="-in ${INPUT} "

EXE_ARG+=" -k on g 1 -sf kk -pk kokkos neigh half newton on gpu/aware on "

MPI_ARG=" -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth "

AFFINITY=""
AFFINITY=../../../HelperScripts/Aurora/set_affinity_gpu.sh

NUMA=""
#NUMA=" numactl --preferred=0 "

COMMAND="mpiexec ${MPI_ARG} ${AFFINITY} ${NUMA} ${EXE} ${EXE_ARG}"
echo "COMMAND= ${COMMAND}"
${COMMAND}
