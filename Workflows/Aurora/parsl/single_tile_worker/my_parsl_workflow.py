import os
import parsl
from parsl import bash_app, python_app

from config import aurora_single_tile_config

# Bash apps are for executing compiled applications or other shell commands
@bash_app
def hello_affinity(stdout='hello.stdout', stderr='hello.stderr'):
    return f'echo tile_id=$ZE_AFFINITY_MASK && $HOME/GettingStarted/Examples/Aurora/affinity_gpu/openmp/hello_affinity'

# Python apps are for executing native python functions
@python_app
def hello_world(message, sleep_time=1):
    import time
    time.sleep(sleep_time)
    return f"Hello {message}"

working_directory = os.getcwd()

with parsl.load(aurora_single_tile_config):

    # Create 100 hello_world tasks
    hello_world_futures = [hello_world(f"Aurora {i}") for i in range(100)]

    # Create 100 hello_affinity tasks
    hello_affinity_futures = [hello_affinity_futures.append(
                                stdout=f"{working_directory}/output/hello_{i}.stdout",
                                stderr=f"{working_directory}/output/hello_{i}.stderr")
                              for i in range(100)]
    
    # This line will block until all hello_world results are returned
    hello_world_results = [tf.result() for tf in hello_world_futures]
    print(f"python apps apps return the function result, e.g. {hello_world_results[0]=}")

    # This line will block until all hello_affinity results are returned
    hello_affinity_results = [tf.result() for tf in hello_affinity_futures]
    print(f"bash apps apps return the return code of the executable, e.g. {hello_affinity_results[0]=}")

    print("Tasks done!")