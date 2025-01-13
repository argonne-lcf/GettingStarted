import parsl
import os
from parsl.config import Config
# PBSPro is the right provider for ALCF:
from parsl.providers import PBSProProvider
# The MPIExecutor is for running MPI applications:
from parsl.executors import MPIExecutor
# Use the Simple launcher
from parsl.launchers import SimpleLauncher

# These options will run work in 1 node batch jobs run one at a time
nodes_per_job = 10
max_num_jobs = 1
tile_names = [f'{gid}.{tid}' for gid in range(6) for tid in range(2)]

# We will save outputs in the current working directory
working_directory = os.getcwd()

mpi_ensemble_config = Config(
    executors=[
        MPIExecutor(
            # This allows for a minimum run size of 1 node for task runs
            max_workers_per_block=nodes_per_job, 
            provider=PBSProProvider(
                account="Aurora_deployment",
                worker_init=f"""source $HOME/_env/bin/activate; \
                                cd {working_directory}""",
                walltime="0:30:00",
                queue="workq",
                scheduler_options="#PBS -l filesystems=home:flare",
                launcher=SimpleLauncher(),
                select_options="",
                nodes_per_block=nodes_per_job,
                max_blocks=1,
                cpus_per_node=208,
            ),
        ),
    ],
    retries=1,
)
