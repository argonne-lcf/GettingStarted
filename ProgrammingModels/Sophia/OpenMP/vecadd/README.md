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

[knight@sophia-gpu-07 vecadd]$ ./vecadd
# of devices= 1
Using single-precision


Result is CORRECT!! :)

