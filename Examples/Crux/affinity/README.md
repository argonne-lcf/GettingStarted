# Compilation w/ Cray compiler wrappers

In this environment, the Cray compiler wrappers `cc, CC, ftn` point to the base CPU compilers for the Programming Environment loaded (e.g. gnu and cray). It is recommended to use the Cray compiler wrappers for compiling and linking applications.

Users are able to build applications on the Crux login nodes, but may find it convenient to build and test applications on the Crux compute nodes in short interactive jobs. This also has the benefit of allowing one to quickly test submission scripts and explore `mpiexec` settings within a single job.
```
$ qsub -I -l select=2,walltime=0:30:00 -l filesystems=home:grand:eagle -A <PROJECT>

$ make clean
$ make

$ ./submit.sh
```

## Example submission script

The following submission script will launch 64 MPI ranks on each node allocated. The MPI ranks are bound to CPUS with a depth (stride) of 1.
```
#!/bin/sh -l
#PBS -l select=2:system=crux
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q debug 
#PBS -A <PROJECT>
#PBS -l filesystems=home:grand:eagle

cd ${PBS_O_WORKDIR}

# MPI example w/ 64 MPI ranks per node (1 rank per core)
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=64
NDEPTH=1
NTHREADS=1

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

echo "Affinitying using cpu-bind depth"
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth ./hello_affinity
```

## Example output:
This example launches 64 MPI ranks on each node with each rank bound to a single core and output is written to the stdout file generated.
```
$ qsub -l select=2,walltime=0:10:00 -l filesystems=home:grand:eagle -A <PROJECT> ./submit.sh 

NUM_OF_NODES= 2 TOTAL_NUM_RANKS= 128 RANKS_PER_NODE= 64 THREADS_PER_RANK= 1
Affinitying using cpu-bind depth
To affinity and beyond!! nname= x1000c0s3b0n0  rnk= 0  list_cores= (0)
...
To affinity and beyond!! nname= x1000c0s3b0n0  rnk= 63  list_cores= (63)

To affinity and beyond!! nname= x1000c0s3b0n1  rnk= 64  list_cores= (0)
...
To affinity and beyond!! nname= x1000c0s3b0n1  rnk= 127  list_cores= (63)
```