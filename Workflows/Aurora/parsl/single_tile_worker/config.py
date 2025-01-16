import os
from parsl.config import Config

# PBSPro is the right provider for ALCF:
from parsl.providers import PBSProProvider
# The high throughput executor is for scaling large single core/tile/gpu tasks on HPC system:
from parsl.executors import HighThroughputExecutor
# Use the MPI launcher to launch worker processes:
from parsl.launchers import MpiExecLauncher

# These options will run work in 1 node batch jobs run one at a time
nodes_per_job = 1
max_num_jobs = 1
tile_names = [f'{gid}.{tid}' for gid in range(6) for tid in range(2)]

# The config will launch workers from this directory
execute_dir = os.getcwd()

aurora_single_tile_config = Config(
    executors=[
        HighThroughputExecutor(
            # Ensures one worker per GPU tile on each node
            available_accelerators=tile_names,
            max_workers_per_node=12,
            # Distributes threads to workers/tiles in a way optimized for Aurora 
            cpu_affinity="list:0-7,104-111:8-15,112-119:16-23,120-127:24-31,128-135:32-39,136-143:40-47,144-151:52-59,156-163:60-67,164-171:68-75,172-179:76-83,180-187:84-91,188-195:92-99,196-203",
            # Increase if you have many more tasks than workers
            prefetch_capacity=0,
            # Options that specify properties of PBS Jobs
            provider=PBSProProvider(
                # Project name
                account="Aurora_deployment",
                # Submission queue
                queue="debug",
                # Commands run before workers launched
                # Make sure to activate your environment where Parsl is installed
                worker_init=f'''source $HOME/_env/bin/activate; cd {execute_dir}''',
                # Wall time for batch jobs
                walltime="0:30:00",
                # Change if data/modules located on other filesystem
                scheduler_options="#PBS -l filesystems=home:flare",
                # Ensures 1 manger per node; the manager will distribute work to its 12 workers, one per tile
                launcher=MpiExecLauncher(bind_cmd="--cpu-bind", overrides="--ppn 1"),
                # options added to #PBS -l select aside from ncpus
                select_options="",
                # Number of nodes per PBS job
                nodes_per_block=nodes_per_job,
                # Minimum number of concurrent PBS jobs running workflow
                min_blocks=0,
                # Maximum number of concurrent PBS jobs running workflow
                max_blocks=max_num_jobs,
                # Hardware threads per node
                cpus_per_node=208,
            ),
        ),
    ],
    # How many times to retry failed tasks
    # this is necessary if you have tasks that are interrupted by a PBS job ending
    # so that they will restart in the next job
    retries=1,
)
