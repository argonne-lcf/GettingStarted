import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.threads = True
mpi4py.rc.thread_level = 'multiple'
from mpi4py import MPI


def main():
    # Initialize MPI
    if not MPI.Is_initialized():
        MPI.Init_thread()

    # Get the number of processes and the rank of the process
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    hostname = MPI.Get_processor_name()

    print(
        f'Rank {rank} / {size} says hello from node {hostname}',
        flush=True
    )
    comm.Barrier()

    # Print some info about the MPI environment
    if rank == 0:
        print('\n'.join([
            'Some info:',
            '----------',
            f'MPI number of threads: {MPI.Query_thread()}',
            f'MPI Vendor: {MPI.get_vendor()}',
            f'MPI Library version: {MPI.Get_library_version()}',
        ]))

    # Perform test allreduce operations
    min_loc = comm.allreduce((2.0 * rank, rank), op=MPI.MINLOC)
    max_loc = comm.allreduce((2.0 * rank, rank), op=MPI.MAXLOC)

    if rank == 0:
        print(f'Min val: {min_loc[0]} at rank: {min_loc[1]}')
        print(f'Max val: {max_loc[0]}, at rank: {max_loc[1]}', flush=True)


if __name__ == '__main__':
    main()
