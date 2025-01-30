#!/bin/bash -l
#PBS -l select=1:system=aurora
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q debug
#PBS -A Catalyst
#PBS -l filesystems=home:flare

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=96
NTHREADS=1

export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_NUM_THREADS=${NTHREADS}
export OMP_PLACES=cores

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))

INPUT=in.lj

EXE=/home/knight/lammps/lammps-git/src/lmp_aurora

EXE_ARG="-in ${INPUT} "

#EXE_ARG="-pk omp ${NTHREADS} -sf omp "
#EXE_ARG="-pk omp ${NTHREADS} -pk intel 0 -sf intel"

EXE_ARG+="-pk gpu 1 -pk omp ${NTHREADS} -sf hybrid gpu omp "
#EXE_ARG+="-pk gpu 1 -pk omp ${NTHREADS} -pk intel ${NTHREADS} -sf hybrid gpu intel "

MPI_ARG=" -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} "
MPI_ARG+=" --cpu-bind list:1,105:2,106:3,107:4,108:5,109:6,110:7,111:8,112:9,113:10,114:11,115:12,116:13,117:14,118:15,119:16,120:17,121:18,122:19,123:20,124:21,125:22,126:23,127:24,128:25,129:26,130:27,131:28,132:29,133:30,134:31,135:32,136:33,137:34,138:35,139:36,140:37,141:38,142:39,143:40,144:41,145:42,146:43,147:44,148:45,149:46,150:47,151:48,152:53,157:54,158:55,159:56,160:57,161:58,162:59,163:60,164:61,165:62,166:63,167:64,168:65,169:66,170:67,171:68,172:69,173:70,174:71,175:72,176:73,177:74,178:75,179:76,180:77,181:78,182:79,183:80,184:81,185:82,186:83,187:84,188:85,189:86,190:87,191:88,192:89,193:90,194:91,195:92,196:93,197:94,198:95,199:96,200:97,201:98,202:99,203:100,204 "

AFFINITY=""
#AFFINITY=../../../HelperScripts/Aurora/set_affinity_gpu.sh
AFFINITY=../../../HelperScripts/Aurora/set_affinity_gpu_4ccs.sh 

NUMA=""
#NUMA=" numactl --preferred=0 "

COMMAND="mpiexec ${MPI_ARG} ${AFFINITY} ${NUMA} ${EXE} ${EXE_ARG}"
echo "COMMAND= ${COMMAND}"
${COMMAND}
