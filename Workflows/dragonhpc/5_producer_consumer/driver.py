import os
import sys
from typing import List, Optional
import argparse
import socket

import dragon
import multiprocessing as mp
from dragon.data.ddict.ddict import DDict
from dragon.native.process_group import ProcessGroup
from dragon.native.process import ProcessTemplate, MSG_PIPE, MSG_DEVNULL
from dragon.infrastructure.connection import Connection
from dragon.infrastructure.policy import Policy
from dragon.native.machine import System, Node
from dragon.infrastructure.facts import PMIBackend

## Get some information on the system
HOSTNAME = socket.getfqdn()
if "aurora" in HOSTNAME:
    PMI_BACKEND = PMIBackend.PMIX
elif "polaris" in HOSTNAME:
    PMI_BACKEND = PMIBackend.CRAY
else:
    raise ValueError(f"Unknown system: {HOSTNAME}")

## Read output from ProcessGroup
def read_output(stdout_conn: Connection) -> str:
    """Read stdout from the Dragon connection.

    :param stdout_conn: Dragon connection to rank 0's stdout
    :type stdout_conn: Connection
    :return: string with the output from stdout
    :rtype: str
    """
    output = ""
    try:
        # this is brute force
        while True:
            output += stdout_conn.recv()
    except EOFError:
        pass
    finally:
        stdout_conn.close()
    return output

## Read error from ProcessGroup
def read_error(stderr_conn: Connection) -> str:
    """Read stdout from the Dragon connection.

    :param stderr_conn: Dragon connection to rank 0's stderr
    :type stderr_conn: Connection
    :return: string with the output from stderr
    :rtype: str
    """
    output = ""
    try:
        # this is brute force
        while True:
            output += stderr_conn.recv()
    except EOFError:
        pass
    finally:
        stderr_conn.close()
    return output

## Launch a process group
def launch_ProcessGroup(
    num_procs_pn: int, 
    nodelist: List[str],
    exe: str, 
    args_list: List[str], 
    run_dir: str, 
    global_policy: Optional[Policy] = None,
    cpu_bind: Optional[List[int]] = None
) -> None:
    """
    Launch a ProcessGroup
    """
    grp = ProcessGroup(
        restart=False, 
        pmi=PMI_BACKEND,
        ignore_error_on_exit=True, 
        policy=global_policy
    )
    for node_num in range(len(nodelist)):   
        node_name = Node(nodelist[node_num]).hostname
        if cpu_bind is not None and len(cpu_bind)>0:
            for proc in range(num_procs_pn):
                local_policy = Policy(placement=Policy.Placement.HOST_NAME, 
                    host_name=node_name,
                    cpu_affinity=[cpu_bind[proc]]
                )
                grp.add_process(nproc=1, 
                                template=ProcessTemplate(target=exe, 
                                                         args=list(args_list), 
                                                         cwd=run_dir,
                                                         policy=local_policy, 
                                                         stdout=MSG_DEVNULL))
        else:
            local_policy = Policy(placement=Policy.Placement.HOST_NAME, host_name=node_name)
            grp.add_process(nproc=num_procs_pn, 
                            template=ProcessTemplate(target=exe, 
                                                     args=args_list, 
                                                     cwd=run_dir,
                                                     policy=local_policy, 
                                                     stdout=MSG_DEVNULL))
    grp.init()
    grp.start()
    grp.join()
    grp.stop()

## Mixed launch
def launch_workflow(args: argparse.Namespace, dd_serialized: str, ddict_nodelist: List[str], sim_nodelist: List[str], ml_nodelist: List[str]) -> None:
    """
    Launch the workflow with the mixed deployment (components are colocated on same nodes,
    but data can still transfer across nodes to fill in DDict)

    :param args: command line arguments
    :type args: argparse.Namespace
    :param dd_serialized: serialized Dragon Distributed Dictionary
    :type dd_serialized: str
    :param ddict_nodelist: node list for the DDict
    :type ddict_nodelist: List[str]
    :param sim_nodelist: node list for the simulation
    :type sim_nodelist: List[str]
    :param ml_nodelist: node list for the training
    :type ml_nodelist: List[str]
    """
    # Set global policy
    global_policy = Policy(distribution=Policy.Distribution.BLOCK)

    # Set up and launch the simulation component
    print('Launching the simulation ...', flush=True)
    sim_exe = "./sim"
    sim_args_list = [f"{args.num_pts}", f"{dd_serialized}"]
    sim_run_dir = os.getcwd()
    sim_cpu_bind = [1,8,16,24,32,40,53,60,68,76,84,92]
    sim_launch_proc = mp.Process(
        target=launch_ProcessGroup, 
        args=(
            args.procs_per_node, 
            sim_nodelist, 
            sim_exe, 
            sim_args_list, 
            sim_run_dir,
            global_policy, 
            sim_cpu_bind,
        )
    )
    sim_launch_proc.start()
    print('Done\n', flush=True)

    # Setup and launch the distributed training component
    print('Launching the training ...', flush=True)
    ml_exe = sys.executable # gets the python executable
    ml_args_list = ["./trainer.py", f"--ddict_ser={dd_serialized}"]
    #dd_serialized_nice = dd_serialized.replace('=',r'\=')
    ml_run_dir = os.getcwd()
    ml_cpu_bind = [4,12,20,28,36,44,56,64,72,80,88,96]
    ml_launch_proc = mp.Process(
        target=launch_ProcessGroup, 
        args=(
            args.procs_per_node, 
            ml_nodelist,
            ml_exe, 
            ml_args_list, 
            ml_run_dir,
            global_policy, 
            ml_cpu_bind,
        )
    )
    ml_launch_proc.start()
    print('Done\n', flush=True)

    # Join both simulation and training
    print('Waiting for simulation and training to complete ...', flush=True)
    ml_launch_proc.join()
    sim_launch_proc.join()
    print('Done\n', flush=True)

## Main function
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--deployment", type=str, default="mixed", choices=["colocated", "clustered", "mixed"], help="Deployment type")
    parser.add_argument("--num_pts", type=int, default=10000, help="Number of points in array")
    parser.add_argument("--ddict_nodes", type=int, default=1, help="Number of nodes for the DDict")
    parser.add_argument("--ddict_mem_size_per_node", type=float, default=0.1, help="Memory size per node for the DDict (in GB)")
    parser.add_argument("--managers_per_node", type=int, default=4, help="Number of managers per node for the DDict")
    parser.add_argument("--procs_per_node", type=int, default=12, help="Number of processes per node for the simulation and training")
    args = parser.parse_args()

    # Set the start method for multiprocessing to 'dragon'
    mp.set_start_method("dragon")
    
    # Get information on this allocation
    alloc = System()
    num_tot_nodes = alloc.nnodes
    dragon_nodelist = alloc.nodes
    print(f"\nRunning on {len(dragon_nodelist)} total nodes")
    print([Node(dragon_nodelist[i]).hostname for i in range(len(dragon_nodelist))], "\n")

    # Split nodes between components according to the deployment type
    if args.deployment == "colocated" or args.deployment == "mixed":
        ddict_nodes = sim_nodes = ml_nodes = num_tot_nodes
        ddict_nodelist = sim_nodelist = ml_nodelist = dragon_nodelist
    elif args.deployment == "clustered":
        assert (num_tot_nodes - args.ddict_nodes) % 2 == 0, \
            "Number of nodes for the DDict must be even for clustered deployment"
        ddict_nodes = args.ddict_nodes
        sim_nodes = (num_tot_nodes - args.ddict_nodes) // 2
        ml_nodes = (num_tot_nodes - args.ddict_nodes) // 2
        ddict_nodelist = [dragon_nodelist[i] for i in range(args.ddict_nodes)]
        sim_nodelist = [dragon_nodelist[i] for i in range(args.ddict_nodes, args.ddict_nodes+sim_nodes)]
        ml_nodelist = [dragon_nodelist[i] for i in range(args.ddict_nodes+sim_nodes, args.ddict_nodes+sim_nodes+ml_nodes)]

    # Start the Dragon Distributed Dictionary (DDict)
    total_mem_size = args.ddict_mem_size_per_node * ddict_nodes * (1024*1024*1024)
    print(f"Total memory size for the DDict: {total_mem_size} GB", flush=True)
    dd_policy = Policy(cpu_affinity=[50,51,100,101])
    dd = DDict(
        managers_per_node=args.managers_per_node, 
        n_nodes=ddict_nodes, 
        total_mem=total_mem_size, 
        policy=dd_policy, 
        timeout=3600
    )
    print(f"Launched the Dragon Dictionary on {ddict_nodes} nodes \n", flush=True)

    # Serialize the DDict
    dd_serialized = dd.serialize()

    # Launch the workflow
    print(f"Running with the {args.deployment} deployment \n")
    launch_workflow(args, dd_serialized, ddict_nodelist, sim_nodelist, ml_nodelist)

    # Close the DDict and quit
    dd.destroy()
    print("\nClosed the Dragon Dictionary", flush=True)
    print("\nQuitting ...", flush=True)


## Run main
if __name__ == "__main__":
    main()