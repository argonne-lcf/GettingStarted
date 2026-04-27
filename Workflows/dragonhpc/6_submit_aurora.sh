#!/bin/bash -l
#PBS -S /bin/bash
#PBS -N dragon_example
#PBS -l select=2
#PBS -l walltime=0:30:00
#PBS -l filesystems=home:flare
#PBS -A <project_name>
#PBS -q debug-scaling
#PBS -k doe
#PBS -j oe

cd $PBS_O_WORKDIR/5_producer_consumer
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
    if [ -f sim.out ]; then
        rm sim.out
    fi
    if [ -f trainer.out ]; then
        rm trainer.out
    fi
    if ls ddict_orc* 1> /dev/null 2>&1; then
        rm ddict_orc*
    fi
}
clean_up

# Setup run
NUM_PTS=10000
DEPLOYMENT="mixed"
DDICT_NODES=1

if [ "$DEPLOYMENT" == "colocated" ] || [ "$DEPLOYMENT" == "mixed" ]; then
    DDICT_NODES=1
fi

# Run workflow
echo -e "\nRunning DragonHPC Producer-Consumer Workflow"
echo "================================================"
dragon driver.py --deployment $DEPLOYMENT --num_pts $NUM_PTS --ddict_nodes $DDICT_NODES
echo "================================================"

