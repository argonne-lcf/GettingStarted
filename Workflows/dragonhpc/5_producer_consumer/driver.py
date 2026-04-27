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
    cpu_bind: Optional[List[int]] = None,
    ddicts: Optional[List[str]] = None
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

## Colocated launch
def launch_colocated(args: argparse.Namespace, dragon_nodelist: List[str]) -> None:
    """
    Launch the workflow with the colocated deployment (components are launched on same set of nodes,
    and data is kept local to each node, no inter-node transfers)

    :param args: command line arguments
    :type args: argparse.Namespace
    :param dragon_nodelist: node list provided by Dragon
    :type dragon_nodelist: List[str]
    """
    # Print nodelist
    print(f"\nRunning on {len(dragon_nodelist)} total nodes")
    print([Node(dragon_nodelist[i]).hostname for i in range(len(dragon_nodelist))], "\n")

    global_policy = Policy(distribution=Policy.Distribution.BLOCK)
    sim_nodelist = dragon_nodelist
    ml_nodelist = dragon_nodelist

    # Launch a DDict on each node
    num_dd_nodes = 1
    node_mem_size = args.dict_mem_size_per_node * (1024*1024*1024)
    ddicts = {}
    ddicts_serialized = []
    for node_num in range(len(dragon_nodelist)):
        try:
            node_name = Node(dragon_nodelist[node_num]).hostname
            dd_policy = Policy(placement=Policy.Placement.HOST_NAME, host_name=node_name)
            dd = DDict(cfg.dict.managers_per_node, num_dd_nodes, node_mem_size, policy=dd_policy)
            dd['node'] = node_name
            ddicts[node_name] = dd
            ddicts_serialized.append(dd.serialize())
        except Exception as e:
            print(e, flush=True)
    print('Launched the dictionaries on all the nodes \n', flush=True)

    # Set up and launch the simulation component
    print('Launching the simulation ...', flush=True)
    sim_args_list = []
    if (cfg.sim.executable.split("/")[-1].split('.')[-1]=='py'):
        sim_exe = sys.executable
        sim_args_list.append(cfg.sim.executable)
    sim_args_list.extend(cfg.sim.arguments.split(' '))
    sim_args_list.append(f'--dictionary={ddicts_serialized[0]}')
    sim_run_dir = os.getcwd()
    sim_launch_proc = mp.Process(target=launch_ProcessGroup, args=(cfg.sim.procs, cfg.sim.procs_pn, sim_nodelist,
                                                                   sim_exe, sim_args_list, sim_run_dir,
                                                                   global_policy, list(cfg.sim.cpu_bind),
                                                                   ddicts_serialized))
    sim_launch_proc.start()
    print('Done\n', flush=True) 

    # Setup and launch the distributed training component
    print('Launching the training ...', flush=True)
    ml_args_list = []
    ml_exe = sys.executable
    ml_args_list.append(cfg.train.executable)
    if (cfg.train.config_path): ml_args_list.append(f'--config-path={cfg.train.config_path}')
    if (cfg.train.config_name): ml_args_list.append(f'--config-name={cfg.train.config_name}')
    ml_args_list.extend([f'ppn={cfg.train.procs_pn}',
                         f'online.simprocs={cfg.sim.procs}',
                         f'online.backend=dragon',
                         f'online.launch=colocated'],
                         )
    ddicts_serialized_nice = [dd_tmp.replace('=',r'\=') for dd_tmp in ddicts_serialized]
    ml_args_list.append(f'online.dragon.dictionary={ddicts_serialized_nice[0]}')
    ml_run_dir = os.getcwd()
    ml_launch_proc = mp.Process(target=launch_ProcessGroup, args=(cfg.train.procs, cfg.train.procs_pn, ml_nodelist,
                                                                  ml_exe, ml_args_list, ml_run_dir,
                                                                  global_policy, list(cfg.train.cpu_bind),
                                                                  ddicts_serialized_nice))
    ml_launch_proc.start()
    print('Done\n', flush=True)

    # Join both simulation and training
    ml_launch_proc.join()
    sim_launch_proc.join()
    print('Joined simulation and training \n', flush=True)

    # Destroy all the DDicts
    for node_num in range(len(dragon_nodelist)):
        node_name = Node(dragon_nodelist[node_num]).hostname
        dd = ddicts[node_name]
        dd.destroy()
    print('Destroyed all dictionaries \n', flush=True)
    print('Exiting launcher ...', flush=True) 

## Clustered launch
def launch_clustered(args: argparse.Namespace, dd_serialized: str, dragon_nodelist: List[str]) -> None:
    """
    Launch the workflow with the clustered deployment (components are launched on separate set of nodes,
    so data is always transferred across nodes to fill in DDict)

    :param args: command line arguments
    :type args: argparse.Namespace
    :param dd_serialized: serialized Dragon Distributed Dictionary
    :type dd_serialized: str
    :param dragon_nodelist: node list provided by Dragon
    :type dragon_nodelist: List[str]
    """
    # Print nodelist
    print(f"\nRunning on {len(dragon_nodelist)} total nodes")
    dd_nodelist = [dragon_nodelist[i] for i in range(cfg.dict.num_nodes)]
    sim_nodelist = [dragon_nodelist[i] for i in range(cfg.dict.num_nodes, cfg.dict.num_nodes+cfg.sim.num_nodes)]
    ml_nodelist = [dragon_nodelist[i] for i in range(cfg.dict.num_nodes+cfg.sim.num_nodes, 
                                                     cfg.dict.num_nodes+cfg.sim.num_nodes+cfg.train.num_nodes)]
    print(f"Database running on {cfg.dict.num_nodes} nodes:")
    print([Node(dd_nodelist[i]).hostname for i in range(cfg.dict.num_nodes)])
    print(f"Simulatiom running on {cfg.sim.num_nodes} nodes:")
    print([Node(sim_nodelist[i]).hostname for i in range(cfg.sim.num_nodes)])
    print(f"ML running on {cfg.train.num_nodes} nodes:")
    print([Node(ml_nodelist[i]).hostname for i in range(cfg.train.num_nodes)])
    sys.stdout.flush()

    global_policy = Policy(distribution=Policy.Distribution.BLOCK)

    # Set up and launch the simulation component
    print('Launching the simulation ...', flush=True)
    sim_args_list = []
    if (cfg.sim.executable.split("/")[-1].split('.')[-1]=='py'):
        sim_exe = sys.executable
        sim_args_list.append(cfg.sim.executable)
    sim_args_list.extend(cfg.sim.arguments.split(' '))
    sim_args_list.append(f'--dictionary={dd_serialized}')
    sim_run_dir = os.getcwd()
    sim_launch_proc = mp.Process(target=launch_ProcessGroup, args=(cfg.sim.procs, cfg.sim.procs_pn, sim_nodelist,
                                                                   sim_exe, sim_args_list, sim_run_dir,
                                                                   global_policy, list(cfg.sim.cpu_bind)))
    sim_launch_proc.start()
    print('Done\n', flush=True)

    # Setup and launch the distributed training component
    print('Launching the training ...', flush=True)
    ml_args_list = []
    ml_exe = sys.executable
    ml_args_list.append(cfg.train.executable)
    if (cfg.train.config_path): ml_args_list.append(f'--config-path={cfg.train.config_path}')
    if (cfg.train.config_name): ml_args_list.append(f'--config-name={cfg.train.config_name}')
    ml_args_list.extend([f'ppn={cfg.train.procs_pn}',
                         f'online.simprocs={cfg.sim.procs}',
                         f'online.backend=dragon',
                         f'online.launch=clustered'],
                         )
    dd_serialized_nice = dd_serialized.replace('=',r'\=')
    ml_args_list.append(f'online.dragon.dictionary={dd_serialized_nice}')
    ml_run_dir = os.getcwd()
    ml_launch_proc = mp.Process(target=launch_ProcessGroup, args=(cfg.train.procs, cfg.train.procs_pn, ml_nodelist,
                                                                  ml_exe, ml_args_list, ml_run_dir,
                                                                  global_policy, list(cfg.train.cpu_bind)))
    ml_launch_proc.start()
    print('Done\n', flush=True)

    # Join both simulation and training
    ml_launch_proc.join()
    sim_launch_proc.join()
    print('Exiting launcher ...', flush=True)

## Mixed launch
def launch_mixed(args: argparse.Namespace, dd_serialized: str, dragon_nodelist: List[str]) -> None:
    """
    Launch the workflow with the mixed deployment (components are colocated on same nodes,
    but data can still transfer across nodes to fill in DDict)

    :param args: command line arguments
    :type args: argparse.Namespace
    :param dd_serialized: serialized Dragon Distributed Dictionary
    :type dd_serialized: str
    :param dragon_nodelist: node list provided by Dragon
    :type dragon_nodelist: List[str]
    """
    # Set global policy
    global_policy = Policy(distribution=Policy.Distribution.BLOCK)
    sim_nodelist = dragon_nodelist
    ml_nodelist = dragon_nodelist

    # Set up and launch the simulation component
    print('Launching the simulation ...', flush=True)
    sim_exe = "./sim"
    sim_args_list = [f"{args.num_pts}", f"{dd_serialized}"]
    sim_run_dir = os.getcwd()
    sim_launch_proc = mp.Process(
        target=launch_ProcessGroup, 
        args=(
            args.procs_per_node, 
            sim_nodelist, 
            sim_exe, 
            sim_args_list, 
            sim_run_dir,
            global_policy, 
        )
    )
    sim_launch_proc.start()
    print('Done\n', flush=True)

    # Setup and launch the distributed training component
    print('Launching the training ...', flush=True)
    ml_exe = sys.executable # gets the python executable
    ml_args_list = ["./trainer.py"]
    dd_serialized_nice = dd_serialized.replace('=',r'\=')
    ml_args_list.append(f"--ddict_ser={dd_serialized}")
    ml_run_dir = os.getcwd()
    ml_launch_proc = mp.Process(
        target=launch_ProcessGroup, 
        args=(
            args.procs_per_node, 
            ml_nodelist,
            ml_exe, 
            ml_args_list, 
            ml_run_dir,
            global_policy, 
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
    parser.add_argument("--ddict_mem_size_per_node", type=int, default=8, help="Memory size per node for the DDict (in GB)")
    parser.add_argument("--managers_per_node", type=int, default=1, help="Number of managers per node for the DDict")
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

    # Start the Dragon Distributed Dictionary (DDict)
    if args.deployment == "colocated" or args.deployment == "mixed":
        ddict_nodes = num_tot_nodes
    else:
        ddict_nodes = args.ddict_nodes
    total_mem_size = args.ddict_mem_size_per_node * ddict_nodes * (1024*1024*1024)
    #dd_policy = Policy(cpu_affinity=list(cfg.dict.cpu_bind)) if cfg.dict.cpu_bind else None
    dd = DDict(
        managers_per_node=args.managers_per_node, 
        n_nodes=ddict_nodes, 
        total_mem=total_mem_size, 
        #policy=dd_policy, 
        timeout=3600
    )
    print(f"Launched the Dragon Dictionary on {ddict_nodes} nodes \n", flush=True)

    # Serialize the DDict
    dd_serialized = dd.serialize()

    # Launch the workflow
    print(f"Running with the {args.deployment} deployment \n")
    if (args.deployment == "colocated"):
        launch_colocated(args, dd_serialized, dragon_nodelist)
    elif (args.deployment == "clustered"):
        launch_clustered(args, dd_serialized, dragon_nodelist)
    elif (args.deployment == "mixed"):
        launch_mixed(args, dd_serialized, dragon_nodelist)

    # Close the DDict and quit
    dd.destroy()
    print("\nClosed the Dragon Dictionary", flush=True)
    print("\nQuitting ...", flush=True)


## Run main
if __name__ == "__main__":
    main()