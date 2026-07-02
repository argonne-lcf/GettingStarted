#!/bin/bash -l
#PBS -N el-llm
#PBS -l select=2
#PBS -l walltime=00:30:00
#PBS -q debug-scaling
#PBS -A <project_name>
#PBS -l filesystems=home:flare
#PBS -j oe
cd $PBS_O_WORKDIR

set -e

# Load the modules
module load frameworks
module load mpifileutils
module load xpu-smi
module list

# Make sure HF Token is set
if [[ -z "${HF_TOKEN}" ]]; then
    echo "Error: HF_TOKEN is not set" >&2
    exit 1
fi

# Set job variables
NODES=$(cat ${PBS_NODEFILE} | wc -l)
MODEL=meta-llama/Llama-3.3-70B-Instruct
MODEL_FLARE_PATH=/flare/datasets/model-weights/hub
TMP_PATH=/tmp/hf_home
TP_SIZE=4
BATCH_SIZE=16
ENGINES_PER_NODE=2
ACTORS_PER_POOL=$(( 1 * ENGINES_PER_NODE ))

# Compile bcast
mpicc -O2 -o ./bcast /flare/datasets/softwares/bcast/bcast.c

# Build the virtual environment with EL and move to other nodes
python -m venv /tmp/_env --system-site-packages
source /tmp/_env/bin/activate
cd /tmp
git clone https://github.com/argonne-lcf/ensemble_launcher.git
cd ensemble_launcher
git checkout multi_node_vllm_rb # NB: to remove, things will be merged into main
pip install .
cd $PBS_O_WORKDIR
mpiexec -np "${NODES}" -ppn 1 --cpu-bind numa ./bcast --no-root-write \
  /tmp/_env /tmp

# Move model weights to /tmp on the nodes
MODEL_DIR="models--${MODEL//\//--}"
MODEL_FLARE_PATH=/flare/datasets/model-weights/hub/$MODEL_DIR
if [[ ! -e "$MODEL_FLARE_PATH" ]]; then
    echo "Did not find model $MODEL_FLARE_PATH"
    exit 1
fi
MODEL_TMP_PATH=/tmp/hf_home/hub/
mpiexec -np "${NODES}" -ppn 1 --cpu-bind numa ./bcast \
  $MODEL_FLARE_PATH $MODEL_TMP_PATH
export HF_HOME=/tmp/hf_home

# Pre-build vLLM model-info caches and move to other nodes
export VLLM_CACHE_ROOT=/tmp/hf_home/hub/.vllm_cache
echo "Building vLLM model-info cache in ${VLLM_CACHE_ROOT} ..."
python /flare/datasets/softwares/vllm/vllm_build_model_cache.py
echo "Cache build complete."
mpiexec -np "${NODES}" -ppn 1 --cpu-bind numa ./bcast --no-root-write \
  $VLLM_CACHE_ROOT /tmp/hf_home/hub

# Other env variables
export TMPDIR=/tmp
export OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

# Launch workflow
echo -e "\n\nLaunching $MODEL on $NODES nodes with $ENGINES_PER_NODE engines per node..."
python3 ./EL_request_driven.py \
  --model_name $MODEL \
  --cache_dir $HF_HOME \
  --tp_size $TP_SIZE \
  --batch_size $BATCH_SIZE \
  --engines_per_node $ENGINES_PER_NODE \
  --actors_per_pool $ACTORS_PER_POOL \
  --prompt_file /flare/datasets/prompts/prompts.jsonl
