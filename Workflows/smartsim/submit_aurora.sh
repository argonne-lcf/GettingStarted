#!/bin/bash -l
#PBS -S /bin/bash
#PBS -N smartsim_example
#PBS -l select=3
#PBS -l walltime=0:30:00
#PBS -l filesystems=home:flare
#PBS -A <project-name>
#PBS -q debug-scaling
#PBS -k doe
#PBS -j oe

cd $PBS_O_WORKDIR/producer-consumer
export TZ='/usr/share/zoneinfo/US/Central'

echo Jobid: $PBS_JOBID
echo Running on host `hostname`
echo Running on nodes `cat $PBS_NODEFILE`

# Load modules and venv
module load frameworks
source ../_env/bin/activate
module list

# Cleanup
function clean_up {
    if [ -d Example ]; then
        rm -r Example
    fi
}
clean_up

# SmartSim env variables
export SR_LOG_FILE=stdout
export SR_LOG_LEVEL=QUIET
export SR_SOCKET_TIMEOUT=10000

# Setup run
SIM_EXE=./sim
TRAIN_EXE=./trainer.py
NUM_PTS=10000
DEPLOYMENT="clustered"
DB_NODES=1

if [ "$DEPLOYMENT" == "colocated" ]; then
    DB_NODES=1
fi

# Run workflow
echo -e "\nRunning SmartSim Workflow"
echo "================================================"
python driver.py --deployment $DEPLOYMENT --num_pts $NUM_PTS --db_nodes $DB_NODES
echo "================================================"

