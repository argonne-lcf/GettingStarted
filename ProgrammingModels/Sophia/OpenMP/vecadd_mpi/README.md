[knight@sophia-gpu-07 ~]$ module use /soft/compilers/nvhpc/nvhpc_2024/modulefiles
[knight@sophia-gpu-07 ~]$ module load nvhpc

[knight@sophia-gpu-07 vecadd]$ which nvc++
/soft/compilers/nvhpc/nvhpc_2024/Linux_x86_64/24.11/compilers/bin/nvc++

[knight@sophia-gpu-07 vecadd]$ 
[knight@sophia-gpu-07 vecadd]$ make clean
rm -f *.o *~
[knight@sophia-gpu-07 vecadd]$ make
nvc++ -g -O3 -D_SINGLE_PRECISION -mp=gpu -gpu=cc80 -c main.cpp
nvc++ -o vecadd -g -O3 -D_SINGLE_PRECISION -mp=gpu -gpu=cc80 main.o 

[knight@sophia-gpu-07 vecadd_mpi]$ ./submit_by-gpu.sh 
NUM_OF_NODES= 1 TOTAL_NUM_RANKS= 1 RANKS_PER_NODE= 1 THREADS_PER_RANK= 4
COMMAND= mpiexec -n 1 --bind-to core --map-by ppr:1:numa:PE=4   -x OMP_NUM_THREADS=4 -x OMP_PROC_BIND=spread -x OMP_PLACES=cores  ../../../../HelperScripts/Sophia/set_affinity_gpu.sh  ./vecadd
# of devices= 1
Rank 0 on host -1 running on GPU 0!
Using single-precision


Result is CORRECT!! :)
