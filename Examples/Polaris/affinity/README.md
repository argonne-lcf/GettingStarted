# Compilation w/ Cray compiler wrappers
While not necessary to compile this non-GPU code on a compute node, doing so in an interactive job will allow one to quickly explore `mpiexec` settings within a single job.
```
$ qsub -I -l select=2,walltime=0:30:00

$ module load craype-accel-nvidia80
$ make clean
$ make

$ ./submit.sh
```
## Example submission script
The following submission script will launch 16 MPI ranks on each node allocated. The MPI ranks are bound to CPUS with a depth (stride) of 4.
```
#!/bin/sh

# MPI example w/ 16 MPI ranks per node spread evenly across cores
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=16
NDEPTH=4
NTHREADS=1

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth ./hello_affinity
```

## Example output:
This example launches 16 MPI ranks on each node with each rank bound to sets of four cores and output is written to the stdout file generated.
```
$ qsub -l select=2,walltime=0:10:00 ./submit.sh 

NUM_OF_NODES= 2 TOTAL_NUM_RANKS= 32 RANKS_PER_NODE= 16 THREADS_PER_RANK= 1
To affinity and beyond!! nname= x3007c0s13b0n0  rnk= 0  list_cores= (0-3)

To affinity and beyond!! nname= x3007c0s13b0n0  rnk= 1  list_cores= (4-7)
...
To affinity and beyond!! nname= x3007c0s13b0n0  rnk= 15  list_cores= (60-63)

To affinity and beyond!! nname= x3007c0s13b1n0  rnk= 16  list_cores= (0-3)
...
To affinity and beyond!! nname= x3007c0s13b1n0  rnk= 31  list_cores= (60-63)

```
