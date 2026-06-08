#!/bin/bash -l
#PBS -N dragon-llm
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -q debug
#PBS -A <project>
#PBS -l filesystems=home:flare
#PBS -j oe
cd $PBS_O_WORKDIR

# Load frameworks module
module load frameworks
module list
source _dragon_venv/bin/activate

# Env variables
export TMPDIR=/tmp
export OPENBLAS_NUM_THREADS=1
HF_TOKEN="your_token"
PROMPTS="./prompts.jsonl"
MODEL="/path/to/.cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"

# Launch workflow
NODES=$(cat ${PBS_NODEFILE} | wc -l)
echo -e "\n\nLaunching workflow on $NODES nodes ..."
dragon ./dragon_llm_inference.py \
  --hf_token $HF_TOKEN  \
  --model_name $MODEL \
  --tp_size 1 \
  --batch_size 32 \
  --prompt_file $PROMPTS

