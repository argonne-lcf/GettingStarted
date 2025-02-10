#include <stdio.h>
#include <iostream>
#include <fstream>
#include <mpi.h>

void foo()
{
  int a = 1;
  int b;
  MPI_Allreduce(&a, &b, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
}

void bar()
{
  int a = 1;
  MPI_Bcast(&a, 1, MPI_INT, 1, MPI_COMM_WORLD);
}

int main(int argc, char *argv[])
{
  int me, num_ranks;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  if(me == 0) {
    std::cout << "num_ranks= " << num_ranks << std::endl;
    if(num_ranks == 1) std::cout << "Need to run on at least 2 MPI ranks..." << std::endl;
  }
  
  std::ofstream outfile;
  if(me == 0) outfile.open("test.log");
  
  for(int i=0; i<10; ++i) {
    foo();
    if(me == 0) {
      outfile << "To hang, or not to hang, that is the question!" << std::endl;
      std::cout << "To hang, or not to hang, that is the question!" << std::endl;
    }
  }

  if(me == 0) std::cout << "Hanging... Hanging... Hung..." << std::endl;
  
  if(me == 0) bar();
  else foo();

  std::cout << "Should not reach here... me= " << me << std::endl;
  
  outfile.close();
  
  MPI_Finalize();
}
