# Scaling LLM Inference on ALCF Systems

Intro ...

## DragonHPC

### Set up

On Aurora

```
module load frameworks
python -m venv _dragon_venv --system-site-packages
source _dragon_venv/bin/activate

# Install dragonhpc
python3 -m pip install dragonhpc
dragon-config add --ofi-runtime-lib=="/opt/cray/libfabric/1.22.0/lib64/"
```

### Run

```
qsub sub_dragon_aurora.sh
```


...

## EnsembleLauncher (EL)

...

## Tips for scaling

Mention things like moving env, model weights and cache to /tmp on each of the nodes and provide utilities for these things.

