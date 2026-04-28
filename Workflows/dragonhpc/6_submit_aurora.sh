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
function clean_up() {
    local mode="${1:-}"
    case "$mode" in
        all)
            [ -f sim.out ]     && rm sim.out
            [ -f trainer.out ] && rm trainer.out
            ls ddict_orc* 1> /dev/null 2>&1 && rm ddict_orc*
            ;;
        ddict)
            ls ddict_orc* 1> /dev/null 2>&1 && rm ddict_orc*
            ;;
        *)
            echo "clean_up: unknown mode '${mode}'; expected 'all' or 'ddict'" >&2
            return 1
            ;;
    esac
}
clean_up all

# Build
./config.sh

# Setup run
NUM_PTS=100
DEPLOYMENT="mixed"
DDICT_NODES=1

if [ "$DEPLOYMENT" == "colocated" ] || [ "$DEPLOYMENT" == "mixed" ]; then
    DDICT_NODES=1
fi

# Dragon cleanup
dragon-cleanup

# Run workflow
echo -e "\nRunning DragonHPC Producer-Consumer Workflow"
echo "================================================"
dragon driver.py --deployment $DEPLOYMENT --num_pts $NUM_PTS --ddict_nodes $DDICT_NODES
echo "================================================"

clean_up ddict
