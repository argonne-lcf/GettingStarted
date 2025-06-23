# Pytorch on ALCF 

This repo consists of simple Pytorch Scripts that can run on CPU, CUDA GPUs and INTEL XPUs

This repo is inspired from https://github.com/FilippoSimini/pytorch_on_polaris.git. 

## Usage

python cpu_train.py

python gpu_train.py

mpiexec -n 4 -ppn 4 python ddp_train.py

mpiexec -n 12 -ppn 12 python xpu_train.py


 
