#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <filesystem>
#include <unistd.h>

#include <dragon/dictionary.hpp>
#include <dragon/serializable.hpp>
#include "cpp_serializers.hpp"
#include <mpi.h>

// Default DDict operation timeout
static timespec_t TIMEOUT = {600, 0};

// DDict type used by this proxy simulation:
//   keys  : strings (e.g. "check-run", "y.<rank>")
//   values: 1D vectors of doubles (check-run is a 1 entry whose value
//           is interpreted as a boolean run flag by the reader)
using SimDDict = dragon::DDict<dragon::SerializableString,
                               custom::SerializableDoubleVector>;

// Logger
static FILE *g_log_fp = nullptr;
static int   g_log_rank = 0;
static int   g_log_size = 0;
static bool  g_debug_enabled = false;

static void log_init(MPI_Comm comm, const char *path)
{
    MPI_Comm_rank(comm, &g_log_rank);
    MPI_Comm_size(comm, &g_log_size);

    if (g_log_rank == 0) {
        FILE *f = std::fopen(path, "w");
        if (f) std::fclose(f);
    }
    MPI_Barrier(comm);
    g_log_fp = std::fopen(path, "a");
}

static void log_close()
{
    if (g_log_fp) {
        std::fflush(g_log_fp);
        std::fclose(g_log_fp);
        g_log_fp = nullptr;
    }
}

static void log_line(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void log_line(const char *fmt, ...)
{
    if (!g_log_fp) return;

    char ts[32];
    std::time_t t = std::time(nullptr);
    std::tm tm_val;
    localtime_r(&t, &tm_val);
    std::strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", &tm_val);

    std::fprintf(g_log_fp, "%s [rank %3d/%d] INFO ",
                 ts, g_log_rank, g_log_size);

    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(g_log_fp, fmt, ap);
    va_end(ap);

    std::fputc('\n', g_log_fp);
    std::fflush(g_log_fp);
}

#define log_debug(...) do { if (g_debug_enabled) log_line(__VA_ARGS__); } while (0)


int check_run(MPI_Comm comm, SimDDict *dd)
{
    int exit_val = 1;
    dragon::SerializableString run_key("check-run");
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Only head rank queries the DDict
    if (rank == 0) {
        if (dd->contains(run_key)) {
            custom::SerializableDoubleVector run_val = (*dd)[run_key];
            const std::vector<double> &run_val_vec = run_val.getVal();
            if (!run_val_vec.empty()) {
                exit_val = static_cast<int>(run_val_vec[0]);
            }
        }
    }
    MPI_Bcast(&exit_val, 1, MPI_INT, 0, comm);

    if (exit_val == 0 && rank == 0) {
        log_line("[Sim] ML training says time to quit");
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

    // Initialize the shared log file (must come after MPI_Init and before
    // any log_line calls so ranks don't race on the truncation).
    log_init(comm, "sim.out");

    if (rank == 0) {
        log_line("[Sim] Running with %d MPI ranks and head node %s",
                 size, hostname);
    }

    // Read input
    if (argc < 4 || argc > 5) {
        if (rank == 0) {
            log_line("[Sim] Usage: %s <deployment> <num_points> <serialized_ddict> [verbosity]",
                     argv[0]);
            log_line("[Sim] verbosity: 'info' (default) or 'debug'");
            log_line("[Sim] Expected 3-4 arguments, got %d", argc - 1);
        }
        log_close();
        MPI_Finalize();
        return -1;
    }
    std::string deployment = argv[1];
    unsigned long long int N = std::stoll(argv[2]);
    const char *ddict_ser = argv[3];
    if (argc == 5) {
        if (std::strcmp(argv[4], "debug") == 0) {
            g_debug_enabled = true;
        } else if (std::strcmp(argv[3], "info") != 0 && rank == 0) {
            log_line("[Sim] Unknown verbosity '%s'; defaulting to 'info'",
                     argv[3]);
        }
    }
    if (rank == 0 && g_debug_enabled) {
        log_line("[Sim] Debug logging enabled");
    }

    // Attach to the Distributed Dictionary created on the Python side
    if (rank == 0) {
        log_line("[Sim] Attaching to Dragon DDict ...");
    }
    SimDDict dd(ddict_ser, &TIMEOUT);
    MPI_Barrier(comm);
    if (rank == 0) {
        log_line("[Sim] All done");
    }

    // For colocated deployments, use the local manager to access local data only
    if (deployment == "colocated") {
        dd = dd.manager(dd.local_manager());
    }

    // Setup iteration loop
    int iters = 100;
    const int NFIELDS = 3;
    std::vector<double> U(N * NFIELDS, 0.0);
    dragon::SerializableString U_key("y." + std::to_string(rank));
    double stream_time = 0.0;
    int count = 0;

    // Loop
    for (int iter=0; iter<iters; iter++) {
        // Check if should exit iteration loop
        int exit_val = check_run(comm, &dd);
        if (exit_val == 0) {
            break;
        }
        MPI_Barrier(comm);
        if (rank == 0) {
            log_debug("[DEBUG] Passed check_run on iter %d", iter);
        }

        // Update solution array and sleep to emulate compute time.
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        const double rank_offset = static_cast<double>(rank)
                                 * static_cast<double>(N)
                                 * static_cast<double>(NFIELDS);
        for (int c=0; c<NFIELDS; c++) {
            for (unsigned long long int n=0; n<N; n++) {
                U[c * N + n] = rank_offset + static_cast<double>(c);
            }
        }
        MPI_Barrier(comm);
        if (rank == 0) {
            log_debug("[DEBUG] Updated solution array on iter %d", iter);
        }

        // Send data to the DDict
        if (rank == 0) {
            log_line("[Sim] Sending data for step %d", iter);
        }
        MPI_Barrier(comm);
        double tic_serialize = MPI_Wtime();
        custom::SerializableDoubleVector U_value(U);
        double toc_serialize = MPI_Wtime();
        if (rank == 0) {
            log_debug("[DEBUG] Serialized data for step %d in %.6f seconds",
                      iter, toc_serialize - tic_serialize);
        }
        double tic_put = MPI_Wtime();
        dd[U_key] = U_value;
        double toc_put = MPI_Wtime();
        if (rank == 0) {
            log_debug("[DEBUG] Put data for step %d in %.6f seconds",
                      iter, toc_put - tic_put);
        }
        if (iter > 0) {
            stream_time += toc_put - tic_put;
            count++;
        }
        MPI_Barrier(comm);
        if (rank == 0) {
            log_line("[Sim] Done writing solution data for step %d in %.6f seconds",
                     iter, toc_put - tic_serialize);
        }

        // debug: print available keys in the DDict
        if (rank == 0 && g_debug_enabled) {
            std::string keys;
            for (const auto &key : dd.keys()) {
                keys += key.getVal() + " ";
            }
            log_debug("[DEBUG] Available keys in DDict: %s", keys.c_str());
        }
        MPI_Barrier(comm);
    }

    // Compute average put time across all ranks
    stream_time /= count;
    double global_avg_stream_time;
    MPI_Allreduce(&stream_time, &global_avg_stream_time, 1, MPI_DOUBLE, MPI_SUM, comm);
    global_avg_stream_time /= size;

    // Print performance summary
    if (rank == 0) {
        double data_size_gb = static_cast<double>(N) * NFIELDS * 8.0 / 1e9;
        double put_bw = data_size_gb / global_avg_stream_time;
        log_line("=== Performance Summary ===");
        log_line("Array shape per message: %llu x %d", N, NFIELDS);
        log_line("Data size per message: %g GB", data_size_gb);
        log_line("Total iterations: %d", count + 1);
        log_line("Average DDict put time: %g seconds", global_avg_stream_time);
        log_line("Average DDict put bandwidth: %g GB/s", put_bw);
    }

    log_close();
    MPI_Finalize();

    return 0;
}
