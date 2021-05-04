# Compilation w/ GCC
```
module load cobalt/cobalt-gpu
qsub -I -n 1 -t 60 -q single-gpu -A Catalyst

make clean
make

./submit.sh
```
## Example output:
```
# of devices= 1
  [0] Platform[ Nvidia ] Type[ GPU ] Device[ A100-SXM4-40GB MIG 4c.7g.40gb ]
Running on GPU 0!
Using single-precision

  Name= A100-SXM4-40GB MIG 4c.7g.40gb
  Locally unique identifier= 
  Clock Frequency(KHz)= 1410000
  Compute Mode= 0
  Major compute capability= 8
  Minor compute capability= 0
  Number of multiprocessors on device= 56
  Warp size in threads= 32
  Single precision performance ratio= 2

Result is CORRECT!! :)
```
# Compilation w/ NVHPC SDK w/ ALCF provided OpenMPI
```
qsub -I -n 1 -t 60 -q single-gpu -A Catalyst

module load nvhpc-nompi/21.3

make -f Makefile.nvhpc clean
make -f Makefile.nvhpc

./submit.sh
```
## Example output:
```
# of devices= 1
  [0] Platform[ Nvidia ] Type[ GPU ] Device[ A100-SXM4-40GB MIG 4c.7g.40gb ]
Running on GPU 0!
Using single-precision

  Name= A100-SXM4-40GB MIG 4c.7g.40gb
  Locally unique identifier= 
  Clock Frequency(KHz)= 1410000
  Compute Mode= 0
  Major compute capability= 8
  Minor compute capability= 0
  Number of multiprocessors on device= 56
  Warp size in threads= 32
  Single precision performance ratio= 2

Result is CORRECT!! :)
```
