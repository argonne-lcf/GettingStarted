#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <filesystem>
#include <unistd.h>

#include "client.h"
#include <mpi.h>



int check_run(MPI_Comm comm, SmartRedis::Client *client)
{
    int exit_val = 1;
    std::string run_key = "check-run";
    int *check_run = new int[1]();
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Check if check-run tensor exists in DB from head rank
    if (rank == 0) {
        if (client->tensor_exists(run_key)) {
            client->unpack_tensor(run_key, check_run, {1},
                SRTensorTypeInt32, SRMemLayoutContiguous);
            exit_val = check_run[0];
        }
    }
    MPI_Bcast(&exit_val, 1, MPI_INT, 0, comm);

    if (exit_val==0 && rank==0) {
        std::cout << "[Sim] ML training says time to quit" << std::endl;
    }
    return exit_val;
}


int main(int argc, char *argv[])
{
    int rank;
    int size;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm comm = MPI_COMM_WORLD;
    int local_rank = std::stoi(std::getenv("PALS_LOCAL_RANKID"));
    int local_size = std::stoi(std::getenv("PALS_LOCAL_SIZE"));
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    if (rank == 0) {
        std::cout << "[Sim] Running with " << size << " MPI ranks and head node " << hostname << std::endl;
    }

    // Read input
    if (argc != 3) {
        std::cerr << "[Sim] Usage: " << argv[0] << " <num_points> <db_nodes>" << std::endl;
        std::cerr << "[Sim] Expected 2 argument, got " << (argc - 1) << std::endl;
        MPI_Finalize();
        return -1;
    } 
    unsigned long long int N = std::stoll(argv[1]);
    int db_nodes = std::stoi(argv[2]);

    // Initialize SmartRedis client
    if (rank == 0) {
        std::cout << "[Sim] Initializing SmartRedis client ...\n" << std::endl;
    }
    bool cluster_mode;
    if (db_nodes > 1)
        cluster_mode = true;
    else
        cluster_mode = false;
    std::string logger_name("Client");
    SmartRedis::Client client(cluster_mode, logger_name);
    MPI_Barrier(comm);
    if (rank == 0) {
        std::cout << "[Sim] All done\n" << std::endl;
    }

    // Setup iteration loop
    int iters = 500;
    std::vector<double> U(N, 0.0);
    std::string key = "y." + std::to_string(rank);
    double stream_time = 0.0;
    int count = 0;

    // Loop
    for (int iter=0; iter<iters; iter++) {
        // Check if should exit iteration loop
        int exit_val = check_run(comm, &client);
        if (exit_val == 0) {
            break;
        }
        
        // Update solution vector and sleep to emulate compute time
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        double frac = (iter != 0) ? (1.0 / iter) : 0.0;
        for (int n=0; n<N; n++) {
            U[n] = static_cast<double>(n+frac);
        }

        // Send data to DB
        if (rank == 0) {
            std::cout << "[Sim] Sending data for step " << iter << std::endl;
        }
        MPI_Barrier(comm);
        double tic = MPI_Wtime();
        client.put_tensor(key, U.data(), {N}, SRTensorTypeDouble, SRMemLayoutContiguous);
        double toc = MPI_Wtime();
        if (iter > 0) {
            stream_time += toc - tic;
            count++;
        }
        MPI_Barrier(comm);
        if (rank == 0) {
            std::cout << "[Sim] Done writing solution data for step " << iter << " in " << toc - tic << " seconds" << std::endl;
        }
    }

    // Compute average put_tensor time across all ranks
    stream_time /= count;
    double global_avg_stream_time;
    MPI_Allreduce(&stream_time, &global_avg_stream_time, 1, MPI_DOUBLE, MPI_SUM, comm);
    global_avg_stream_time /= size;

    // Print put_tensor performance summary
    if (rank == 0) {
        std::cout << "\n=== Performance Summary ===" << std::endl;
        double data_size_gb = N * 8.0 / 1e9;
        double recv_bw = N * 8.0 / global_avg_stream_time / 1e9;
        std::cout << "Data size per message: " << data_size_gb << " GB" << std::endl;
        std::cout << "Total iterations: " << count+1 << std::endl;
        std::cout << "Average put_tensor time: " << global_avg_stream_time << " seconds" << std::endl;
        std::cout << "Average put_tensor bandwidth: " << recv_bw << " GB/s" << std::endl;
    }

    MPI_Finalize();

    return 0;
}

