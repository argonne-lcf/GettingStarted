#!/bin/bash -l
#PBS -N mpi-vllm
#PBS -l select=2
#PBS -l walltime=00:30:00
#PBS -q debug-scaling
#PBS -A datascience
#PBS -l filesystems=home:flare
#PBS -j oe
cd $PBS_O_WORKDIR

# Load modules
module load frameworks
module load xpu-smi
module list

STAGE_DSYNC=/flare/datascience/balin/vllm/dragon_Mar26/pbs_utils/stage_dsync

# Move model weights to /tmp on the nodes
MODEL_FLARE_PATH=/flare/datascience/balin/vllm/dragon_Mar26/.cache
MODEL_TMP_PATH=/tmp/hf_home/hub/.cache/
$STAGE_DSYNC -r 8 $MODEL_FLARE_PATH $MODEL_TMP_PATH
MODEL_DIR=$MODEL_TMP_PATH/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659

# Pre-build vLLM model-info caches
export VLLM_CACHE_ROOT=/flare/datascience/balin/vllm/dragon_Jun26/.vllm_cache
echo "Building vLLM model-info caches in ${VLLM_CACHE_ROOT} ..."
python ./vllm_build_model_cache.py
echo "Cache build complete."

# Move model-info cache to /tmp on the nodes
MODELINFO_FLARE_PATH=$VLLM_CACHE_ROOT
MODELINFO_TMP_PATH=/tmp/hf_home/hub/.vllm_cache
$STAGE_DSYNC -r 8 $MODELINFO_FLARE_PATH $MODELINFO_TMP_PATH
export VLLM_CACHE_ROOT=$MODELINFO_TMP_PATH

# Move prompts to /tmp on the nodes
PROMPTS_FLARE_PATH=/flare/datascience/balin/vllm/prompts.jsonl
PROMPTS_TMP_PATH=/tmp/hf_home/prompts.jsonl
$STAGE_DSYNC -r 8 $PROMPTS_FLARE_PATH $PROMPTS_TMP_PATH

# Other env variables
export TMPDIR=/tmp
export OPENBLAS_NUM_THREADS=1
HF_TOKEN=""
export VLLM_LOGGING_LEVEL=INFO # DEBUG, INFO, WARNING, ERROR

# Launch workflow
BASE_PATH=/flare/datascience/balin/vllm/mpi
EXE=${BASE_PATH}/mpi_llm_inference.py
TP_SIZE=1
BATCH_SIZE=16
NODES=$(cat ${PBS_NODEFILE} | wc -l)
ENGINES_PER_NODE=12
RANKS=$(( NODES * ENGINES_PER_NODE ))
CPU_BIND="list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100"
echo -e "\n\nLaunching workflow on $NODES nodes with $ENGINES_PER_NODE engines per node..."

# Run
mpiexec -n $RANKS --ppn $ENGINES_PER_NODE --cpu-bind $CPU_BIND \
  python $EXE \
  --hf_token $HF_TOKEN  \
  --model_name $MODEL_DIR \
  --tp_size $TP_SIZE \
  --batch_size $BATCH_SIZE \
  --prompt_file $PROMPTS_TMP_PATH \
  2>&1 | tee vllm_log.txt

