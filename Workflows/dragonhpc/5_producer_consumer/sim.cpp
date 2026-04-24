#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <filesystem>
#include <unistd.h>

#include <dragon/dictionary.hpp>
#include <dragon/serializable.hpp>
#include <mpi.h>

// Default DDict operation timeout (60s)
static timespec_t TIMEOUT = {60, 0};

// DDict type used by this proxy simulation:
//   keys  : strings (e.g. "check-run", "y.<rank>")
//   values: 2D vectors of doubles (check-run is a 1x1 entry whose value
//           is interpreted as a boolean run flag by the reader)
using SimDDict = dragon::DDict<dragon::SerializableString,
                               dragon::SerializableDouble2DVector>;

// ---------------------------------------------------------------------------
// Minimal rank-aware file logger
//
// All ranks append to a single shared file (sim.out). Rank 0 truncates the
// file at startup and every rank appends afterwards. POSIX guarantees that
// writes <= PIPE_BUF to an O_APPEND file are atomic, so line-sized records
// from multiple ranks stay intact.
// ---------------------------------------------------------------------------
static FILE *g_log_fp = nullptr;
static int   g_log_rank = 0;
static int   g_log_size = 0;

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


int check_run(MPI_Comm comm, SimDDict *dd)
{
    int exit_val = 1;
    dragon::SerializableString run_key("check-run");
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Only head rank queries the DDict
    if (rank == 0) {
        if (dd->contains(run_key)) {
            dragon::SerializableDouble2DVector run_val = (*dd)[run_key];
            const std::vector<std::vector<double>> &run_val_vec = run_val.getVal();
            if (!run_val_vec.empty() && !run_val_vec[0].empty()) {
                exit_val = static_cast<int>(run_val_vec[0][0]);
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
    if (argc != 3) {
        if (rank == 0) {
            log_line("[Sim] Usage: %s <num_points> <serialized_ddict>", argv[0]);
            log_line("[Sim] Expected 2 arguments, got %d", argc - 1);
        }
        log_close();
        MPI_Finalize();
        return -1;
    }
    unsigned long long int N = std::stoll(argv[1]);
    const char *ddict_ser = argv[2];

    // Attach to the Distributed Dictionary created on the Python side
    if (rank == 0) {
        log_line("[Sim] Attaching to Dragon DDict ...");
    }
    SimDDict dd(ddict_ser, &TIMEOUT);
    MPI_Barrier(comm);
    if (rank == 0) {
        log_line("[Sim] All done");
    }

    // Setup iteration loop
    int iters = 500;
    const int NCOLS = 3;
    std::vector<std::vector<double>> U(N, std::vector<double>(NCOLS, 0.0));
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

        // Update solution array and sleep to emulate compute time
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        double frac = (iter != 0) ? (1.0 / iter) : 0.0;
        for (unsigned long long int n=0; n<N; n++) {
            for (int c=0; c<NCOLS; c++) {
                U[n][c] = static_cast<double>(n + c) + frac;
            }
        }

        // Send data to the DDict
        if (rank == 0) {
            log_line("[Sim] Sending data for step %d", iter);
        }
        MPI_Barrier(comm);
        double tic = MPI_Wtime();
        dragon::SerializableDouble2DVector U_value(U);
        dd[U_key] = U_value;
        double toc = MPI_Wtime();
        if (iter > 0) {
            stream_time += toc - tic;
            count++;
        }
        MPI_Barrier(comm);
        if (rank == 0) {
            log_line("[Sim] Done writing solution data for step %d in %.6f seconds",
                     iter, toc - tic);
        }
    }

    // Compute average put time across all ranks
    stream_time /= count;
    double global_avg_stream_time;
    MPI_Allreduce(&stream_time, &global_avg_stream_time, 1, MPI_DOUBLE, MPI_SUM, comm);
    global_avg_stream_time /= size;

    // Print performance summary
    if (rank == 0) {
        double nbytes = static_cast<double>(N) * NCOLS * 8.0;
        double data_size_gb = nbytes / 1e9;
        double recv_bw = nbytes / global_avg_stream_time / 1e9;
        log_line("=== Performance Summary ===");
        log_line("Array shape per message: %llu x %d", N, NCOLS);
        log_line("Data size per message: %g GB", data_size_gb);
        log_line("Total iterations: %d", count + 1);
        log_line("Average DDict put time: %g seconds", global_avg_stream_time);
        log_line("Average DDict put bandwidth: %g GB/s", recv_bw);
    }

    log_close();
    MPI_Finalize();

    return 0;
}
