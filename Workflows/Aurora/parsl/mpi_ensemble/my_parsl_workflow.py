import os
import parsl
from parsl import bash_app

from config import mpi_ensemble_config

working_directory = os.getcwd()

# This app will run the hello_affinity application with mpiexec
# Using the set_affinity_gpu_aurora.sh script will bind each mpi rank to a gpu tile
@bash_app
def mpi_hello_affinity(parsl_resource_specification, depth=8, stdout='mpi_hello.stdout', stderr='mpi_hello.stderr'):
    APP_DIR = "$HOME/GettingStarted"
    # PARSL_MPI_PREFIX will resolve to `mpiexec -n num_ranks -ppn ranks_per_node -hosts NODE001,NODE002`
    return f"$PARSL_MPI_PREFIX --cpu-bind depth --depth={depth} {APP_DIR}/HelperScripts/Aurora/set_affinity_gpu_aurora.sh {APP_DIR}/Examples/Aurora/affinity_gpu/sycl/hello_affinity"


with parsl.load(mpi_ensemble_config):

    task_futures = []
    
    # Create 2-node tasks
    # We set 12 ranks per node to match the number of gpu tiles on an aurora node
    resource_specification = {'num_nodes': 2, # Number of nodes required for the application instance
                              'ranks_per_node': 12, # Number of ranks / application elements to be launched per node
                              'num_ranks': 24, # Number of ranks in total
                             }

    print(f"Creating mpi tasks with {resource_specification['num_nodes']} nodes per task, {resource_specification['num_ranks']} ranks per task, and {resource_specification['ranks_per_node']} ranks per node")
    task_futures += [mpi_hello_affinity(
                            parsl_resource_specification=resource_specification,
                            stdout=f"{working_directory}/mpi_output/{i}/hello.stdout",
                            stderr=f"{working_directory}/mpi_output/{i}/hello.stderr")
                        for i in range(10)]
    
    # This loop will block until all task results are returned
    print(f"{len(task_futures)} tasks created, wating for completion")
    for tf in task_futures:
        tf.result()
        
    print("Tasks done!")
