# Compilation w/ Cray compiler wrappers

On Crux, the Cray compiler wrappers `cc, CC, ftn` point to the base CPU compilers for the Programming Environment loaded (e.g. gnu and cray). It is recommended to use the Cray compiler wrappers for compiling and linking applications.

Users are able to build applications on the Crux login nodes, but may find it convenient to build and test applications on the Crux compute nodes in short interactive jobs. This also has the benefit of allowing one to quickly test submission scripts and explore `mpiexec` settings within a single job.
```
$ qsub -I -l select=2,walltime=0:30:00 -l filesystems=home:grand:eagle -A <PROJECT>

$ make clean
$ make

$ ./submit.sh
```

## Example submission script

The following submission script will launch 8 MPI ranks on each node allocated. Each MPI rank spawns 8 OpenMP threads that are each bound to a single physical CPU core. No hyperthreads are used in this example.

```
#!/bin/bash -l
#PBS -l select=2:system=crux
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q debug 
#PBS -A Catalyst
#PBS -l filesystems=home:grand:eagle

cd ${PBS_O_WORKDIR}

# MPI example w/ 16 MPI ranks per node spread evenly across cores
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=8
NDEPTH=8
NTHREADS=8

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

# Launch 8 MPI ranks per node, each with 8 OpenMP threads bound to single core (no hyperthreads used) 
MPI_ARGS="-n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth "
OMP_ARGS="--env OMP_NUM_THREADS=${NTHREADS} --env OMP_PROC_BIND=true --env OMP_PLACES=cores "

mpiexec ${MPI_ARGS} ${OMP_ARGS} ./hello_affinity

```

Example output from this 2-node run is shown below. Output from the job is written to the stdout file generated.

```
$ qsub -l select=2,walltime=0:10:00 -l filesystems=home:grand:eagle -A <PROJECT> ./submit.sh 

NUM_OF_NODES= 2 TOTAL_NUM_RANKS= 16 RANKS_PER_NODE= 8 THREADS_PER_RANK= 8
To affinity and beyond!! nname= x1000c0s0b0n0  rnk= 0  tid= 0: list_cores= (0)
To affinity and beyond!! nname= x1000c0s0b0n0  rnk= 0  tid= 1: list_cores= (1)
To affinity and beyond!! nname= x1000c0s0b0n0  rnk= 0  tid= 2: list_cores= (2)
To affinity and beyond!! nname= x1000c0s0b0n0  rnk= 0  tid= 3: list_cores= (3)
To affinity and beyond!! nname= x1000c0s0b0n0  rnk= 0  tid= 4: list_cores= (4)
To affinity and beyond!! nname= x1000c0s0b0n0  rnk= 0  tid= 5: list_cores= (5)
To affinity and beyond!! nname= x1000c0s0b0n0  rnk= 0  tid= 6: list_cores= (6)
To affinity and beyond!! nname= x1000c0s0b0n0  rnk= 0  tid= 7: list_cores= (7)
...
To affinity and beyond!! nname= x1000c0s0b0n0  rnk= 7  tid= 0: list_cores= (56)
To affinity and beyond!! nname= x1000c0s0b0n0  rnk= 7  tid= 1: list_cores= (57)
To affinity and beyond!! nname= x1000c0s0b0n0  rnk= 7  tid= 2: list_cores= (58)
To affinity and beyond!! nname= x1000c0s0b0n0  rnk= 7  tid= 3: list_cores= (59)
To affinity and beyond!! nname= x1000c0s0b0n0  rnk= 7  tid= 4: list_cores= (60)
To affinity and beyond!! nname= x1000c0s0b0n0  rnk= 7  tid= 5: list_cores= (61)
To affinity and beyond!! nname= x1000c0s0b0n0  rnk= 7  tid= 6: list_cores= (62)
To affinity and beyond!! nname= x1000c0s0b0n0  rnk= 7  tid= 7: list_cores= (63)

To affinity and beyond!! nname= x1000c0s0b1n0  rnk= 8  tid= 0: list_cores= (0)
To affinity and beyond!! nname= x1000c0s0b1n0  rnk= 8  tid= 1: list_cores= (1)
To affinity and beyond!! nname= x1000c0s0b1n0  rnk= 8  tid= 2: list_cores= (2)
To affinity and beyond!! nname= x1000c0s0b1n0  rnk= 8  tid= 3: list_cores= (3)
To affinity and beyond!! nname= x1000c0s0b1n0  rnk= 8  tid= 4: list_cores= (4)
To affinity and beyond!! nname= x1000c0s0b1n0  rnk= 8  tid= 5: list_cores= (5)
To affinity and beyond!! nname= x1000c0s0b1n0  rnk= 8  tid= 6: list_cores= (6)
To affinity and beyond!! nname= x1000c0s0b1n0  rnk= 8  tid= 7: list_cores= (7)
...
To affinity and beyond!! nname= x1000c0s0b1n0  rnk= 15  tid= 0: list_cores= (56)
To affinity and beyond!! nname= x1000c0s0b1n0  rnk= 15  tid= 1: list_cores= (57)
To affinity and beyond!! nname= x1000c0s0b1n0  rnk= 15  tid= 2: list_cores= (58)
To affinity and beyond!! nname= x1000c0s0b1n0  rnk= 15  tid= 3: list_cores= (59)
To affinity and beyond!! nname= x1000c0s0b1n0  rnk= 15  tid= 4: list_cores= (60)
To affinity and beyond!! nname= x1000c0s0b1n0  rnk= 15  tid= 5: list_cores= (61)
To affinity and beyond!! nname= x1000c0s0b1n0  rnk= 15  tid= 6: list_cores= (62)
To affinity and beyond!! nname= x1000c0s0b1n0  rnk= 15  tid= 7: list_cores= (63)
```

## Example submission script w/ hyperthreads

Depending on the application needs, one may benefit from the hyperthreads available on each CPU core of which four can be utilized. Providing an explicit CPU list, while tedious, provides the most flexibility to get all of the assignments correct. The following submission script uses a helper script `HelperScripts/Crux/cpu_binding.py` to create the necessary string to pass to `mpiexec`. This example launches 32 MPI ranks on each node allocated. Each MPI rank spawns 8 OpenMP threads that spread across two adjacent physical cores (i.e. 4 OpenMP threads per core). As documented in `/proc/cpuinfo` on a compute node, physical core 0 corresponds to processors 0, 64, 128, 192 and physical core 1 corresponds to processors 1, 64, 129, 193, and so on.

```
#!/bin/bash -l
#PBS -l select=2:system=crux
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q debug 
#PBS -A Catalyst
#PBS -l filesystems=home:grand:eagle

#cd ${PBS_O_WORKDIR}

# MPI example w/ 32 MPI ranks per node and 8 OpenMP threads per rank. Each MPI rank spans 2 physical cores and utilizes all 4 hyperthreads on each core.
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=32
NCORES_PER_RANK=2
NTHREADS=8
STRIDE=64

#Generate cpu bind list during job (or create it beforehand and paste here)
CPU_LIST="`python3 ../../../HelperScripts/Crux/cpu_binding.py 0 ${NRANKS_PER_NODE} ${NCORES_PER_RANK} ${NTHREADS} ${STRIDE}`"
echo "CPU_LIST= ${CPU_LIST}"

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

MPI_ARGS="-n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --cpu-bind list:${CPU_LIST} "
OMP_ARGS="--env OMP_NUM_THREADS=${NTHREADS} --env OMP_PROC_BIND=true "

mpiexec ${MPI_ARGS} ${OMP_ARGS} ./hello_affinity
```

Example output from this 2-node run is shown below.

```
$ qsub -l select=2,walltime=0:10:00 -l filesystems=home:grand:eagle -A <PROJECT> ./submit_ht.sh 

CPU_LIST= 0,1,64,65,128,129,192,193:2,3,66,67,130,131,194,195:4,5,68,69,132,133,196,197:6,7,70,71,134,135,198,199:8,9,72,73,136,137,200,201:10,11,74,75,138,139,202,203:12,13,76,77,140,141,204,205:14,15,78,79,142,143,206,207:16,17,80,81,144,145,208,209:18,19,82,83,146,147,210,211:20,21,84,85,148,149,212,213:22,23,86,87,150,151,214,215:24,25,88,89,152,153,216,217:26,27,90,91,154,155,218,219:28,29,92,93,156,157,220,221:30,31,94,95,158,159,222,223:32,33,96,97,160,161,224,225:34,35,98,99,162,163,226,227:36,37,100,101,164,165,228,229:38,39,102,103,166,167,230,231:40,41,104,105,168,169,232,233:42,43,106,107,170,171,234,235:44,45,108,109,172,173,236,237:46,47,110,111,174,175,238,239:48,49,112,113,176,177,240,241:50,51,114,115,178,179,242,243:52,53,116,117,180,181,244,245:54,55,118,119,182,183,246,247:56,57,120,121,184,185,248,249:58,59,122,123,186,187,250,251:60,61,124,125,188,189,252,253:62,63,126,127,190,191,254,255
NUM_OF_NODES= 2 TOTAL_NUM_RANKS= 64 RANKS_PER_NODE= 32 THREADS_PER_RANK= 8
To affinity and beyond!! nname= x1000c0s0b0n0  rnk= 0  tid= 0: list_cores= (0)
To affinity and beyond!! nname= x1000c0s0b0n0  rnk= 0  tid= 1: list_cores= (128)
To affinity and beyond!! nname= x1000c0s0b0n0  rnk= 0  tid= 2: list_cores= (1)
To affinity and beyond!! nname= x1000c0s0b0n0  rnk= 0  tid= 3: list_cores= (129)
To affinity and beyond!! nname= x1000c0s0b0n0  rnk= 0  tid= 4: list_cores= (64)
To affinity and beyond!! nname= x1000c0s0b0n0  rnk= 0  tid= 5: list_cores= (192)
To affinity and beyond!! nname= x1000c0s0b0n0  rnk= 0  tid= 6: list_cores= (65)
To affinity and beyond!! nname= x1000c0s0b0n0  rnk= 0  tid= 7: list_cores= (193)
...
To affinity and beyond!! nname= x1000c0s0b0n0  rnk= 31  tid= 0: list_cores= (62)
To affinity and beyond!! nname= x1000c0s0b0n0  rnk= 31  tid= 1: list_cores= (190)
To affinity and beyond!! nname= x1000c0s0b0n0  rnk= 31  tid= 2: list_cores= (63)
To affinity and beyond!! nname= x1000c0s0b0n0  rnk= 31  tid= 3: list_cores= (191)
To affinity and beyond!! nname= x1000c0s0b0n0  rnk= 31  tid= 4: list_cores= (126)
To affinity and beyond!! nname= x1000c0s0b0n0  rnk= 31  tid= 5: list_cores= (254)
To affinity and beyond!! nname= x1000c0s0b0n0  rnk= 31  tid= 6: list_cores= (127)
To affinity and beyond!! nname= x1000c0s0b0n0  rnk= 31  tid= 7: list_cores= (255)

To affinity and beyond!! nname= x1000c0s0b1n0  rnk= 32  tid= 0: list_cores= (0)
To affinity and beyond!! nname= x1000c0s0b1n0  rnk= 32  tid= 1: list_cores= (128)
To affinity and beyond!! nname= x1000c0s0b1n0  rnk= 32  tid= 2: list_cores= (1)
To affinity and beyond!! nname= x1000c0s0b1n0  rnk= 32  tid= 3: list_cores= (129)
To affinity and beyond!! nname= x1000c0s0b1n0  rnk= 32  tid= 4: list_cores= (64)
To affinity and beyond!! nname= x1000c0s0b1n0  rnk= 32  tid= 5: list_cores= (192)
To affinity and beyond!! nname= x1000c0s0b1n0  rnk= 32  tid= 6: list_cores= (65)
To affinity and beyond!! nname= x1000c0s0b1n0  rnk= 32  tid= 7: list_cores= (193)
...
To affinity and beyond!! nname= x1000c0s0b1n0  rnk= 63  tid= 0: list_cores= (62)
To affinity and beyond!! nname= x1000c0s0b1n0  rnk= 63  tid= 1: list_cores= (190)
To affinity and beyond!! nname= x1000c0s0b1n0  rnk= 63  tid= 2: list_cores= (63)
To affinity and beyond!! nname= x1000c0s0b1n0  rnk= 63  tid= 3: list_cores= (191)
To affinity and beyond!! nname= x1000c0s0b1n0  rnk= 63  tid= 4: list_cores= (126)
To affinity and beyond!! nname= x1000c0s0b1n0  rnk= 63  tid= 5: list_cores= (254)
To affinity and beyond!! nname= x1000c0s0b1n0  rnk= 63  tid= 6: list_cores= (127)
To affinity and beyond!! nname= x1000c0s0b1n0  rnk= 63  tid= 7: list_cores= (255)
```
