# Compilation w/ Cray compiler wrappers
Users are able to build applications on the Polaris login nodes, but may find it convenient to build and test applications on the Polaris compute nodes in short interactive jobs. This also has the benefit of allowing one to quickly submission scripts.
```
$ qsub -I -l select=1,walltime=0:30:00

$ make -f Makefile.nhpc clean
$ make -f Makefile.nhpc

./submit.sh
```
## Example submission script
The following submission script will launch 8 MPI ranks on each node allocated. The MPI ranks are bound to CPUS with a depth (stride) of 8.
```
#!/bin/sh
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q workq 

cd ${PBS_O_WORKDIR}

# MPI example w/ 8 MPI ranks per node spread evenly across cores
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=8
NDEPTH=8
NTHREADS=1

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth ./hello_affinity
```

## Example output:
In the default environment, each MPI rank detects all four GPU available on each Polaris node and targets the first visible device. All MPI ranks end up targeting the same GPU in this case, which may not be ideal.
```
./submit.sh 
NUM_OF_NODES= 1 TOTAL_NUM_RANKS= 8 RANKS_PER_NODE= 8 THREADS_PER_RANK= 1
rnk= 0 :  # of devices detected= 4
To affinity and beyond!! nname= x3212c0s25b1n0  rnk= 0  list_cores= (0-7)  num_devices= 4
    [0,0] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-87969d3e-0fb9-9e82-85f7-976f56cd8309
    [0,1] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-df43b0c3-c6be-995a-5102-df8f9a25f7dd
    [0,2] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-d3475abc-8c3d-29da-5ec6-47b48cfcd954
    [0,3] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-46279b47-ba62-bd70-8f44-1adf69bba7c9

To affinity and beyond!! nname= x3212c0s25b1n0  rnk= 1  list_cores= (8-15)  num_devices= 4
    [1,0] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-87969d3e-0fb9-9e82-85f7-976f56cd8309
    [1,1] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-df43b0c3-c6be-995a-5102-df8f9a25f7dd
    [1,2] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-d3475abc-8c3d-29da-5ec6-47b48cfcd954
    [1,3] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-46279b47-ba62-bd70-8f44-1adf69bba7c9

To affinity and beyond!! nname= x3212c0s25b1n0  rnk= 2  list_cores= (16-23)  num_devices= 4
    [2,0] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-87969d3e-0fb9-9e82-85f7-976f56cd8309
    [2,1] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-df43b0c3-c6be-995a-5102-df8f9a25f7dd
    [2,2] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-d3475abc-8c3d-29da-5ec6-47b48cfcd954
    [2,3] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-46279b47-ba62-bd70-8f44-1adf69bba7c9

To affinity and beyond!! nname= x3212c0s25b1n0  rnk= 3  list_cores= (24-31)  num_devices= 4
    [3,0] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-87969d3e-0fb9-9e82-85f7-976f56cd8309
    [3,1] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-df43b0c3-c6be-995a-5102-df8f9a25f7dd
    [3,2] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-d3475abc-8c3d-29da-5ec6-47b48cfcd954
    [3,3] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-46279b47-ba62-bd70-8f44-1adf69bba7c9

To affinity and beyond!! nname= x3212c0s25b1n0  rnk= 4  list_cores= (32-39)  num_devices= 4
    [4,0] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-87969d3e-0fb9-9e82-85f7-976f56cd8309
    [4,1] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-df43b0c3-c6be-995a-5102-df8f9a25f7dd
    [4,2] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-d3475abc-8c3d-29da-5ec6-47b48cfcd954
    [4,3] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-46279b47-ba62-bd70-8f44-1adf69bba7c9

To affinity and beyond!! nname= x3212c0s25b1n0  rnk= 5  list_cores= (40-47)  num_devices= 4
    [5,0] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-87969d3e-0fb9-9e82-85f7-976f56cd8309
    [5,1] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-df43b0c3-c6be-995a-5102-df8f9a25f7dd
    [5,2] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-d3475abc-8c3d-29da-5ec6-47b48cfcd954
    [5,3] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-46279b47-ba62-bd70-8f44-1adf69bba7c9

To affinity and beyond!! nname= x3212c0s25b1n0  rnk= 6  list_cores= (48-55)  num_devices= 4
    [6,0] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-87969d3e-0fb9-9e82-85f7-976f56cd8309
    [6,1] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-df43b0c3-c6be-995a-5102-df8f9a25f7dd
    [6,2] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-d3475abc-8c3d-29da-5ec6-47b48cfcd954
    [6,3] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-46279b47-ba62-bd70-8f44-1adf69bba7c9

To affinity and beyond!! nname= x3212c0s25b1n0  rnk= 7  list_cores= (56-63)  num_devices= 4
    [7,0] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-87969d3e-0fb9-9e82-85f7-976f56cd8309
    [7,1] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-df43b0c3-c6be-995a-5102-df8f9a25f7dd
    [7,2] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-d3475abc-8c3d-29da-5ec6-47b48cfcd954
    [7,3] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-46279b47-ba62-bd70-8f44-1adf69bba7c9
```
## GPU Affinity helper script
By setting the environment variable `CUDA_VISIBLE_DEVICES` separately for each MPI rank, one can explicitly set which GPUs (and in which order) will be visible. The helper script `set_affinity_gpu_polaris.sh` if provided as an example.
```
$ cat set_affinity_gpu_polaris.sh 

#!/bin/bash
num_gpus=4
gpu=$((${PMI_LOCAL_RANK} % ${num_gpus}))
export CUDA_VISIBLE_DEVICES=$gpu
echo “RANK= ${PMI_RANK} LOCAL_RANK= ${PMI_LOCAL_RANK} gpu= ${gpu}”
exec "$@"
```
## Example output
After careful inspection of the reported uuids one can confirm that the GPUs were assigned to MPI ranks round-robin.
```
NUM_OF_NODES= 1 TOTAL_NUM_RANKS= 8 RANKS_PER_NODE= 8 THREADS_PER_RANK= 1
“RANK= 0 LOCAL_RANK= 0 gpu= 0”
“RANK= 1 LOCAL_RANK= 1 gpu= 1”
“RANK= 2 LOCAL_RANK= 2 gpu= 2”
“RANK= 3 LOCAL_RANK= 3 gpu= 3”
“RANK= 4 LOCAL_RANK= 4 gpu= 0”
“RANK= 5 LOCAL_RANK= 5 gpu= 1”
“RANK= 6 LOCAL_RANK= 6 gpu= 2”
“RANK= 7 LOCAL_RANK= 7 gpu= 3”
rnk= 0 :  # of devices detected= 1
To affinity and beyond!! nname= x3212c0s25b1n0  rnk= 0  list_cores= (0-7)  num_devices= 1
    [0,0] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-87969d3e-0fb9-9e82-85f7-976f56cd8309

To affinity and beyond!! nname= x3212c0s25b1n0  rnk= 1  list_cores= (8-15)  num_devices= 1
    [1,0] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-df43b0c3-c6be-995a-5102-df8f9a25f7dd

To affinity and beyond!! nname= x3212c0s25b1n0  rnk= 2  list_cores= (16-23)  num_devices= 1
    [2,0] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-d3475abc-8c3d-29da-5ec6-47b48cfcd954

To affinity and beyond!! nname= x3212c0s25b1n0  rnk= 3  list_cores= (24-31)  num_devices= 1
    [3,0] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-46279b47-ba62-bd70-8f44-1adf69bba7c9

To affinity and beyond!! nname= x3212c0s25b1n0  rnk= 4  list_cores= (32-39)  num_devices= 1
    [4,0] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-87969d3e-0fb9-9e82-85f7-976f56cd8309

To affinity and beyond!! nname= x3212c0s25b1n0  rnk= 5  list_cores= (40-47)  num_devices= 1
    [5,0] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-df43b0c3-c6be-995a-5102-df8f9a25f7dd

To affinity and beyond!! nname= x3212c0s25b1n0  rnk= 6  list_cores= (48-55)  num_devices= 1
    [6,0] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-d3475abc-8c3d-29da-5ec6-47b48cfcd954

To affinity and beyond!! nname= x3212c0s25b1n0  rnk= 7  list_cores= (56-63)  num_devices= 1
    [7,0] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-46279b47-ba62-bd70-8f44-1adf69bba7c9
```
