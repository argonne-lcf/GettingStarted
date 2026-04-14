#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <filesystem>
#include <unistd.h>

#include "adios2.h"
#include <mpi.h>



int check_run(MPI_Comm comm, adios2::IO bpIO)
{
    int exit_val = 1;
    int exists;
    std::string fname = "check-run.bp";
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Check if check-run file exists
    if (rank == 0) {
        if (std::filesystem::exists(fname)) {
            printf("[Sim] Found check-run file!\n");
            fflush(stdout);
            exists = 1;
        } else {
            exists = 0;
        }
    }
    MPI_Bcast(&exists, 1, MPI_INT, 0, comm);

    // Read check-run file if exists
    if (exists) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        adios2::Engine reader = bpIO.Open(fname, adios2::Mode::Read);
        reader.BeginStep();
        adios2::Variable<int> var = bpIO.InquireVariable<int>("check-run");
        if (rank == 0 and var) {
            reader.Get(var, &exit_val);
        }
        reader.EndStep();
        reader.Close();
        MPI_Bcast(&exit_val, 1, MPI_INT, 0, comm);
    }

    if (exit_val == 0 && rank == 0) {
        printf("[Sim] ML training says time to quit ...\n");
    }
    fflush(stdout);

    return exit_val;
}


int main(int argc, char *argv[])
{
    int global_rank, rank;
    int global_size, size;
    int provide;

    // MPI_THREAD_MULTIPLE is only required if you enable the SST MPI_DP
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provide);
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &global_size);
    
    // Split communicator based on color (similar to trainer.py)
    int color = 5678;
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, global_rank, &comm);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    if (rank == 0) {
        char hostname[256];
        gethostname(hostname, sizeof(hostname));
        std::cout << "[Sim] Running on host " << hostname << std::endl;
        std::cout << "[Sim] Running with " << size << " MPI ranks \n" << std::endl;
    }

    // Read input
    if (argc != 5) {
        std::cerr << "[Sim] Usage: " << argv[0] << " <num_points> <sync mode> <data plane> <IO mode>" << std::endl;
        std::cerr << "[Sim] Expected 3 argument, got " << (argc - 1) << std::endl;
        MPI_Finalize();
        return -1;
    } 
    long long int N = std::stoll(argv[1]);
    std::string mode = argv[2];
    std::string data_plane = argv[3];
    std::string io_mode = argv[4];

    try
    {
        adios2::ADIOS adios(comm);
        adios2::IO bpIO = adios.DeclareIO("graphStream");
        adios2::IO sstIO = adios.DeclareIO("solutionStream");
        sstIO.SetEngine("Sst");
        adios2::Params params;

        if (mode == "sync") {
            // sync setup
            params["RendezvousReaderCount"] = "1"; // proceed only when 1 reader is present, blocking
            params["QueueFullPolicy"] = "Block"; // block when queue is full
            params["QueueLimit"] = "1"; // number of steps writes allows to be queued before taking action
        } else if (mode == "async") {
            // async setup
            params["RendezvousReaderCount"] = "1"; // proceed even if no reader is present, non-blocking
            params["QueueFullPolicy"] = "Discard"; // discard snapshot when queue is full
            params["QueueLimit"] = "3"; // number of steps writer allows to be queued before taking action (0 means no limit)
            params["ReserveQueueLimit"] = "0"; // number of steps writer allows to be queued before taking action when no reader is connected (0 means no limit)
        }
        params["DataTransport"] = data_plane;
        //params["DataInterface"] = "cxi0";
        params["OpenTimeoutSecs"] = "600";
        sstIO.SetParameters(params);

        // Define graph data
        N = N + rank; // emulate imperfect load balance
        //long long int num_edges = 2 * N + rank; // emulate imperfect load balance
        std::vector<double> pos_node(N, 0.0);
        //std::vector<long long int> edge_index(num_edges * 2, 0);

        for (long long int n=0; n<N; n++) {
            pos_node[n] = static_cast<double>(n);
        }
        //for (long long int n=0; n<num_edges; n++) {
        //    edge_index[n + 0*num_edges] = n;
        //    edge_index[n + 1*num_edges] = n+1;
        //}

        // Get global size of data
        long long int global_N, global_num_edges;
        MPI_Allreduce(&N, &global_N, 1, MPI_LONG_LONG, MPI_SUM, comm);
        //MPI_Allreduce(&num_edges, &global_num_edges, 1, MPI_LONG_LONG, MPI_SUM, comm);

        // Gather size of data
        std::vector<long long int> gathered_N(size);
        //std::vector<long long int> gathered_num_edges(size);
        MPI_Allgather(&N, 1, MPI_LONG_LONG, gathered_N.data(), 1, MPI_LONG_LONG, comm);
        //MPI_Allgather(&num_edges, 1, MPI_LONG_LONG, gathered_num_edges.data(), 1, MPI_LONG_LONG, comm);

        // Get global offset
        long long int offset_N = 0;
        //long long int offset_num_edges = 0;
        for (long long int i=0; i<rank; i++) {
            offset_N += gathered_N[i];
            //offset_num_edges += gathered_num_edges[i];
        }

        // Define ADIOS2 variables
        unsigned long _size = size;
        unsigned long _rank = rank;
        unsigned long long _N = N;
        unsigned long long _global_N = global_N;
        unsigned long long _offset_N = offset_N;
        //unsigned long long _num_edges = num_edges;
        //unsigned long long _global_num_edges = global_num_edges;
        //unsigned long long _offset_num_edges = offset_num_edges;
        auto posVar = bpIO.DefineVariable<double>("pos_node", 
                                                  {_global_N}, // global dim
                                                  {_offset_N}, // starting offset in global dim
                                                  {_N}); // local size
        //auto edgeVar = bpIO.DefineVariable<long long int>("edge_index", 
        //                                            {2 * _global_num_edges}, 
        //                                            {2 * _offset_num_edges}, 
        //                                            {2 * _num_edges});
        auto NVar = bpIO.DefineVariable<long long int>("N", {_size}, {_rank}, {1});
        //auto numedgesVar = bpIO.DefineVariable<long long int>("num_edges", {_size}, {_rank}, {1});

        // Write graph data to file
        std::string path;
        if (io_mode == "daos") {
            path = "/tmp/datascience/balin/graph.bp";
        } else if (io_mode == "posix") {
            path = "./graph.bp";
        }
        double tic = MPI_Wtime();
        adios2::Engine graphWriter = bpIO.Open(path, adios2::Mode::Write);
        graphWriter.BeginStep();

        graphWriter.Put<long long int>(NVar, N);
        //graphWriter.Put<long long int>(numedgesVar, num_edges);
        graphWriter.Put<double>(posVar, pos_node.data());
        //graphWriter.Put<long long int>(edgeVar, edge_index.data());

        graphWriter.EndStep();
        graphWriter.Close();
        double time = MPI_Wtime() - tic;
        MPI_Barrier(comm);
        if (rank == 0) std::cout << "[Sim] Done writing graph data in " << time << std::endl;

        // Setup iteration loop and open stream
        int iters = 500;
        std::vector<double> U(N, 0.0);
        auto UVar = sstIO.DefineVariable<double>("U", {_global_N}, {_offset_N}, {_N});
        if (rank == 0) {
            std::cout << "[Sim] Opening stream ... " << std::endl;
        }
        adios2::Engine solWriter = sstIO.Open("solutionStream", adios2::Mode::Write);

        // Loop
        for (int iter=0; iter<iters; iter++) {
            // Check if should exit iteration loop
            int exit_val = check_run(comm, bpIO);
            if (exit_val == 0) {
                break;
            }
            
            // Update solution vector and sleep to emulate compute time
            std::this_thread::sleep_for(std::chrono::milliseconds(2000));
            double frac = (iter != 0) ? (1.0 / iter) : 0.0;
            for (int n=0; n<N; n++) {
                U[n] = static_cast<double>(n+frac);
            }

            if (rank == 0) {
                std::cout << "[Sim] Sending data for step " << iter << std::endl;
            }
            double tic = MPI_Wtime();
            solWriter.BeginStep();
            solWriter.Put<double>(UVar, U.data());
            solWriter.EndStep();
            MPI_Barrier(comm);
            double toc = MPI_Wtime();
            if (rank == 0) {
                std::cout << "[Sim] Done writing solution data for step " << iter << " in " << toc - tic << " seconds" << std::endl;
            }
        }
        solWriter.Close();
    }
    catch (std::invalid_argument &e)
    {
        std::cout << "[Sim] Invalid argument exception, STOPPING PROGRAM from rank " << rank << "\n";
        std::cout << e.what() << "\n";
    }
    catch (std::ios_base::failure &e)
    {
        std::cout << "IO System base failure exception, STOPPING PROGRAM from rank " << rank
                  << "\n";
        std::cout << e.what() << "\n";
    }
    catch (std::exception &e)
    {
        std::cout << "Exception, STOPPING PROGRAM from rank " << rank << "\n";
        std::cout << e.what() << "\n";
    }

    MPI_Finalize();

    return 0;
}

