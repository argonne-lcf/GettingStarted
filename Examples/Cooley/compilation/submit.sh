#!/bin/bash
#COBALT -n 1
#COBALT -t 30
#COBALT -A SDL_Workshop
#COBALT -q training

NODES=`cat $COBALT_NODEFILE | wc -l`
PROCS=$((NODES * 12))

mpirun -f $COBALT_NODEFILE -n $PROCS ./hellompi
status=$?

echo "mpirun status is $status"
exit $status


