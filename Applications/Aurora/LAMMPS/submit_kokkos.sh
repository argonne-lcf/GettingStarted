#!/bin/bash -l
#PBS -l select=1
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
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))

cd /path_to_work/

EXE=/path_to_lammps/build/lmp

INPUT=in.lj
EXE_ARG="-in ${INPUT} "

EXE_ARG+=" -k on g 1 -sf kk -pk kokkos neigh half newton on gpu/aware on "

MPI_ARG=" -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} "
MPI_ARG+=" --cpu-bind list:1:2:3:4:5:6:53:54:55:56:57:58 "
MPI_ARG+=" --env OMP_NUM_THREADS=${NTHREADS} --env OMP_PROC_BIND=spread --env OMP_PLACES=cores "

AFFINITY="gpu_tile_compact.sh"

COMMAND="mpiexec ${MPI_ARG} ${AFFINITY} ${EXE} ${EXE_ARG}"
echo "COMMAND= ${COMMAND}"
${COMMAND}
