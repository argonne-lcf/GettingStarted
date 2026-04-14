#!/bin/bash -l
##PBS -S /bin/bash
##PBS -N adios2_example
##PBS -l select=2
##PBS -l walltime=0:10:00
##PBS -l filesystems=home:flare
##PBS -A <project_name>
##PBS -q debug-scaling
##PBS -k doe
##PBS -j oe

#cd $PBS_O_WORKDIR
export TZ='/usr/share/zoneinfo/US/Central'

echo Jobid: $PBS_JOBID
echo Running on host `hostname`
echo Running on nodes `cat $PBS_NODEFILE`

# Load modules
module load frameworks
module load adios2/2.11.0-sycl
module list

# ADIOS2 vars
export PYTHONPATH=$PYTHONPATH:/opt/aurora/26.26.0/spack/unified/1.1.1/install/linux-x86_64/adios2-2.11.0-vwdc5l7/lib/python3.12/site-packages/
#export FABRIC_PROVIDER=cxi
#export FABRIC_IFACE=cxi
export SstVerbose=1
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# Clean up old files if they exist
if ls *.sst 1> /dev/null 2>&1
then
    echo Cleaning up old .sst files \n
    rm *.sst
fi
if ls *.bp 1> /dev/null 2>&1
then
    echo Cleaning up old .bp files
    rm -r ./*.bp
fi

# Compute number of ranks
RANKS_PER_NODE=12
TOT_NODES=$(cat $PBS_NODEFILE | wc -l)
if [ $((TOT_NODES % 2)) -ne 0 ]; then
    echo "Error: Need even number of nodes, got $TOT_NODES"
    exit 1
fi
COMPONENT_NODES=$((TOT_NODES / 2))
RANKS=$(( COMPONENT_NODES * RANKS_PER_NODE ))
head -n $COMPONENT_NODES $PBS_NODEFILE > sim_hostfile
tail -n $COMPONENT_NODES $PBS_NODEFILE > trainer_hostfile

# Setup run
CPU_BIND="list:1:8:16:24:32:40:53:60:68:76:84:92"
SIM_EXE=./sim
TRAIN_EXE=./trainer.py
NUM_PTS=10000
MODE=sync
TRANSPORT=RDMA
IO_MODE=posix

# Sequential launch
echo -e "\nRunning sequential launch"
echo "================================================"
mpiexec -np $RANKS --ppn $RANKS_PER_NODE \
  --hostfile ./sim_hostfile --cpu-bind $CPU_BIND \
  $SIM_EXE $NUM_PTS $MODE $TRANSPORT $IO_MODE &
mpiexec -np $RANKS --ppn $RANKS_PER_NODE \
  --hostfile ./trainer_hostfile --cpu-bind $CPU_BIND \
  python $TRAIN_EXE --data_plane $TRANSPORT --io_mode $IO_MODE
wait
echo "================================================"

# MPMD launch
echo -e "\nRunning MPMD launch"
echo "================================================"
mpiexec -np $RANKS --ppn $RANKS_PER_NODE \
  --cpu-bind $CPU_BIND \
  $SIM_EXE $NUM_PTS $MODE $TRANSPORT $IO_MODE \
  : -np $RANKS --ppn $RANKS_PER_NODE python $TRAIN_EXE --data_plane $TRANSPORT --io_mode $IO_MODE
echo "================================================"

# Clean up
rm *.sst 
rm -r *.bp
rm sim_hostfile trainer_hostfile  
