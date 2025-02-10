#!/bin/bash -l
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=0:15:00
#PBS -q EarlyAppAccess
#PBS -A Aurora_deployment

#cd ${PBS_O_WORKDIR}

# MPI example w/ 1 MPI rank per node w/ access to all GPU tiles
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=2
NDEPTH=1
NTHREADS=1

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

MPI_ARG="-n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth "
MPI_ARG+="--env OMP_NUM_THREADS=${NTHREADS} --env OMP_PLACES=cores "

rm -f test.log
rm -rf callstacks_at_hang_point

# Launch log monitor
export APP_EXE=./hello_hang
./hang_monitor.sh test.log &
LOGMON_PID=$!

# Run application
mpiexec ${MPI_ARG} ./hello_hang

# Kill log monitor
./hang_monitor_stop.sh $LOGMON_PID
