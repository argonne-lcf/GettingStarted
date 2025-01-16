import os
import parsl
from parsl import bash_app, python_app

from config import aurora_single_tile_config

# Bash apps are for executing compiled applications or other shell commands
@bash_app
def hello_affinity(stdout='hello.stdout', stderr='hello.stderr'):
    return f'$HOME/GettingStarted/Examples/Aurora/affinity_gpu/sycl/hello_affinity'

# Python apps are for executing native python functions
@python_app
def hello_world(message, sleep_time=1):
    import time
    time.sleep(sleep_time)
    return f"Hello {message}"

working_directory = os.getcwd()

print("Starting my_parsl_workflow")

with parsl.load(aurora_single_tile_config):

    # Create 12 hello_world tasks
    hello_world_futures = [hello_world(f"Aurora {i}") for i in range(12)]
    print(f"Created {len(hello_world_futures)} hello_world tasks")
    
    # Create 12 hello_affinity tasks
    hello_affinity_futures = [hello_affinity(stdout=f"{working_directory}/output/hello_{i}.stdout",
                                             stderr=f"{working_directory}/output/hello_{i}.stderr")
                              for i in range(12)]
    print(f"Created {len(hello_world_futures)} hello_affinity tasks")

    # This line will block until all hello_world results are returned
    hello_world_results = [tf.result() for tf in hello_world_futures]
    print("hello_world tasks complete")
    print(f"python apps like hello_world return the function result, e.g. {hello_world_results[0]=}")

    # This line will block until all hello_affinity results are returned
    hello_affinity_results = [tf.result() for tf in hello_affinity_futures]
    print("hello_affinity tasks complete")
    print(f"bash apps like hello_affinity return the return code of the executable, e.g. {hello_affinity_results[0]=}")

    print(f"Read results of hello_affinity from stdout file:")
    for i,tf in enumerate(hello_affinity_futures):
        with open(f"{working_directory}/output/hello_{i}.stdout", "r") as f:
            outputs = f.readlines()
            print(outputs)
    
    print("Tasks done!")
