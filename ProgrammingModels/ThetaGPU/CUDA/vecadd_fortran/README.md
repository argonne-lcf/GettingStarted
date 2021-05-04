```
# Compilation w/ NVHPC SDK w/ ALCF provided OpenMPI
```
qsub -I -n 1 -t 60 -q single-gpu -A Catalyst

module use /lus/theta-fs0/software/environment/thetagpu/lmod/tmp
module switch openmpi/openmpi-4.0.5 openmpi-4.1.0_nvhpc-21.3 

make -f Makefile.nvhpc clean
make -f Makefile.nvhpc

./submit.sh
```
## Example output:
```
{1, 2, 3, 4, 5} + {10, 20, 30, 40, 50} = { 11 22 33 44 55}
```
