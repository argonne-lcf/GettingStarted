# Submit interactive job to single GPU

```
qsub -I -l select=1,walltime=1:00:00,filesystems=home:eagle -A Catalyst -q by-gpu
```

## Compilation

```
[knight@sophia-gpu-05 affinity]$ module load compilers/openmpi
[knight@sophia-gpu-05 affinity]$ make
mpicxx -g -fopenmp -O3 -std=c++0x  -c main.cpp
mpicxx -o hello_affinity -g -fopenmp -O3 -std=c++0x  main.o
```

## Run single GPU example

```
[knight@sophia-gpu-01 affinity]$ cat submit_1gpu.sh 
#!/bin/bash -l
#PBS -l select=1:system=sophia
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q by-gpu
#PBS -A Catalyst
#PBS -l filesystems=home:grand:eagle

cd ${PBS_O_WORKDIR}

# Example using 1 GPU and 16 cores (1 thread per core)
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=1
NDEPTH=16
NTHREADS=16

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

MPI_ARG="-n ${NTOTRANKS} --npernode ${NRANKS_PER_NODE} "
MPI_ARG=" -x OMP_NUM_THREADS=${NTHREADS} -x OMP_PROC_BIND=spread -x OMP_PLACES=cores "

mpiexec ${MPI_ARG} ./hello_affinity
```

```
[knight@sophia-gpu-01 affinity]$ ./submit_1gpu.sh 
NUM_OF_NODES= 1 TOTAL_NUM_RANKS= 1 RANKS_PER_NODE= 1 THREADS_PER_RANK= 16
To affinity and beyond!! nname= sophia-gpu-01  rnk= 0  tid= 0: list_cores= (48,176)
To affinity and beyond!! nname= sophia-gpu-01  rnk= 0  tid= 1: list_cores= (49,177)
To affinity and beyond!! nname= sophia-gpu-01  rnk= 0  tid= 2: list_cores= (50,178)
To affinity and beyond!! nname= sophia-gpu-01  rnk= 0  tid= 3: list_cores= (51,179)
To affinity and beyond!! nname= sophia-gpu-01  rnk= 0  tid= 4: list_cores= (52,180)
To affinity and beyond!! nname= sophia-gpu-01  rnk= 0  tid= 5: list_cores= (53,181)
To affinity and beyond!! nname= sophia-gpu-01  rnk= 0  tid= 6: list_cores= (54,182)
To affinity and beyond!! nname= sophia-gpu-01  rnk= 0  tid= 7: list_cores= (55,183)
To affinity and beyond!! nname= sophia-gpu-01  rnk= 0  tid= 8: list_cores= (56,184)
To affinity and beyond!! nname= sophia-gpu-01  rnk= 0  tid= 9: list_cores= (57,185)
To affinity and beyond!! nname= sophia-gpu-01  rnk= 0  tid= 10: list_cores= (58,186)
To affinity and beyond!! nname= sophia-gpu-01  rnk= 0  tid= 11: list_cores= (59,187)
To affinity and beyond!! nname= sophia-gpu-01  rnk= 0  tid= 12: list_cores= (60,188)
To affinity and beyond!! nname= sophia-gpu-01  rnk= 0  tid= 13: list_cores= (61,189)
To affinity and beyond!! nname= sophia-gpu-01  rnk= 0  tid= 14: list_cores= (62,190)
To affinity and beyond!! nname= sophia-gpu-01  rnk= 0  tid= 15: list_cores= (63,191)
```
