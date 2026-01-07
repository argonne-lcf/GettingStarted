#!/bin/bash -l
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=0:15:00
#PBS -q prod
#PBS -A Catalyst
#PBS -l filesystems=home:flare

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=12
NTHREADS=1

export MPICH_GPU_SUPPORT_ENABLED=0
export OMP_NUM_THREADS=${NTHREADS}
export OMP_PLACES=cores
export OMP_PROC_BIND=true

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))

cd /path_to_work/

EXE=/path_to_lammps/build/lmp

INPUT=in.lj

EXE_ARG="-in ${INPUT}  "
EXE_ARG+=" -pk gpu 1 -pk omp ${NTHREADS} -sf hybrid gpu omp "

MPI_ARG="-n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} "

if (( NRANKS_PER_NODE <= 12 )); then
MPI_ARG+="--cpu-bind list:1:2:3:4:5:6:53:54:55:56:57:58"
AFFINITY="gpu_tile_compact.sh"
elif (( NRANKS_PER_NODE <= 24)); then
MPI_ARG+="--cpu-bind list:1:2:3:4:5:6:7:8:9:10:11:12:53:54:55:56:57:58:59:60:61:62:63:64"
AFFINITY="/path/GettingStarted/HelperScripts/Aurora/set_affinity_gpu_2ccs.sh"
else
MPI_ARG+="--cpu-bind list:1:2:3:4:5:6:7:8:9:10:11:12:13:14:15:16:17:18:19:20:21:22:23:24:53:54:55:56:57:58:59:60:61:62:63:64:65:66:67:68:69:70:71:72:73:74:75:76"
AFFINITY="/path/GettingStarted/HelperScripts/Aurora/set_affinity_gpu_4ccs.sh"
fi

COMMAND="mpiexec ${MPI_ARG} ${AFFINITY} ${EXE} ${EXE_ARG}"
echo "COMMAND= ${COMMAND}"
${COMMAND}

