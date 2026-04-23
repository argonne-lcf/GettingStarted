# general imports
import os
from omegaconf import DictConfig, OmegaConf
import argparse

# smartsim and smartredis imports
from smartsim import Experiment
from smartsim.settings import PalsMpiexecSettings


## Define function to parse node list
def parseNodeList(fname):
    with open(fname) as file:
        nodelist = file.readlines()
        nodelist = [line.rstrip() for line in nodelist]
        nodelist = [line.split('.')[0] for line in nodelist]
    nNodes = len(nodelist)
    return nodelist, nNodes


## Colocated DB launch
def launch_coDB(args, nodelist, nNodes):
    # Print nodelist
    if (nodelist is not None):
        print(f"\nRunning on {nNodes} total nodes")
        print(nodelist, "\n")
        hosts = ','.join(nodelist)

    db_nodes = 1
    nprocs = args.ppn * nNodes

    # Initialize the SmartSim Experiment
    PORT = 6780
    exp = Experiment(args.name, launcher='pals')

    # Set the run settings, including the client executable and how to run it
    # join current path to 'sim'
    client_exe = os.path.join(os.path.dirname(__file__), 'sim')
    nrs_settings = PalsMpiexecSettings(
                        client_exe,
                        exe_args=None,
                        run_args=None,
                        #env_vars={'MPICH_OFI_CXI_PID_BASE':str(0)}                   
    )
    nrs_settings.set_tasks(nprocs)
    nrs_settings.set_tasks_per_node(args.ppn)
    nrs_settings.set_hostlist(hosts)
    nrs_settings.set_cpu_binding_type("list:1:8:16:24:32:40:53:60:68:76:84:92")
    nrs_settings.add_exe_args(f"{args.num_pts} {db_nodes}")

    # Create the co-located database model
    colo_model = exp.create_model("sim", nrs_settings)
    kwargs = {
        'maxclients': 100000,
        'threads_per_queue': 4, # set to 4 for improved performance
        'inter_op_parallelism': 1,
        'intra_op_parallelism': 1,
        'cluster-node-timeout': 30000,
        }
    db_bind = [50,51,102,103]
    colo_model.colocate_db_uds(
            db_cpus=len(db_bind),
            custom_pinning=db_bind,
            debug=False,
            **kwargs
            )

    # Start the co-located model
    print("Launching simulation and SmartSim co-located DB ... ")
    exp.generate(colo_model, overwrite=True)
    exp.start(colo_model, block=False, summary=False)
    print("Done\n")

    # Setup and launch the training script
    ml_exe = os.path.join(os.path.dirname(__file__), 'trainer.py')
    ml_exe = ml_exe + f' --db_nodes=1'
    SSDB = colo_model.run_settings.env_vars['SSDB']
    ml_settings = PalsMpiexecSettings(
                    'python',
                    exe_args=ml_exe,
                    run_args=None,
                    env_vars={'SSDB':SSDB}
    )
    ml_settings.set_tasks(nprocs)
    ml_settings.set_tasks_per_node(args.ppn)
    ml_settings.set_hostlist(hosts)
    ml_settings.set_cpu_binding_type("list:4:12:20:28:36:44:56:64:72:80:88:96")
    
    print("Launching training script ... ")
    ml_model = exp.create_model("train", ml_settings)
    exp.generate(ml_model, overwrite=True)
    exp.start(ml_model, block=True, summary=False)
    print("Done\n")


## Clustered DB launch
def launch_clDB(args, nodelist, nNodes):
    # Check number of db nodes
    if args.db_nodes == 2:
        print(
            "Error: Orchestrator does not support clusters of size 2. "
            "Increase number of db nodes to at least 3."
            , flush=True
        )
        return

    # Split nodes between the components
    dbNodes_list = None
    if (nodelist is not None):
        dbNodes_list = nodelist[0: args.db_nodes]
        dbNodes = ','.join(dbNodes_list)
        remaining_nodes = nodelist[args.db_nodes:]
        num_remaining_nodes = len(remaining_nodes)
        if num_remaining_nodes%2 == 0:
            simNodes = ','.join(remaining_nodes[0: num_remaining_nodes//2])
            num_sim_nodes = num_remaining_nodes//2
            mlNodes = ','.join(remaining_nodes[num_remaining_nodes//2:])
            num_ml_nodes = num_remaining_nodes - num_sim_nodes
        else:
            print("Error: Need even number of nodes for simulation and ML, got {remaining_nodes}", flush=True)
            return
        print(f"Database running on {args.db_nodes} nodes:")
        print(dbNodes)
        print(f"Simulatiom running on {num_sim_nodes} nodes:")
        print(simNodes)
        print(f"ML running on {num_ml_nodes} nodes:")
        print(mlNodes, flush=True)

        nprocs = args.ppn * num_sim_nodes

    # Set up database and start it
    PORT = 6780
    exp = Experiment(args.name, launcher='pals')
    runArgs = {"np": 1, "ppn": 1, "cpu-bind": "numa"}
    kwargs = {
        'maxclients': 100000,
        'threads_per_queue': 4, # set to 4 for improved performance
        'inter_op_parallelism': 1,
        'intra_op_parallelism': 4,
        'cluster-node-timeout': 30000,
        }
    run_command = 'mpiexec'
    network = ["hsn0", "hsn1", "hsn2", "hsn3", "hsn4", "hsn5", "hsn6", "hsn7"]
    db = exp.create_database(port=PORT, 
                             batch=False,
                             db_nodes=args.db_nodes,
                             run_command=run_command,
                             interface=network, 
                             hosts=dbNodes_list,
                             run_args=runArgs,
                             single_cmd=True,
                             **kwargs
                            )
    exp.generate(db)
    print("\nStarting database ...")
    exp.start(db)
    print("Done\n")

    # Set the run settings, including the client executable and how to run it
    client_exe = os.path.join(os.path.dirname(__file__), 'sim')
    nrs_settings = PalsMpiexecSettings(client_exe,
                                        exe_args=None,
                                        run_args=None,
                                        env_vars=None)
    nrs_settings.set_tasks(nprocs)
    nrs_settings.set_tasks_per_node(args.ppn)
    nrs_settings.set_hostlist(simNodes)
    nrs_settings.set_cpu_binding_type("list:1:8:16:24:32:40:53:60:68:76:84:92")
    nrs_settings.add_exe_args(f"{args.num_pts} {args.db_nodes}")

    # Start the client model
    print("Launching the simulation ...")
    sim_model = exp.create_model("sim", nrs_settings)
    exp.generate(sim_model, overwrite=True)
    exp.start(sim_model, summary=False, block=False)
    print("Done\n")
    
    # Setup and launch the training script
    ml_exe = os.path.join(os.path.dirname(__file__), 'trainer.py')
    ml_exe = ml_exe + f' --db_nodes={args.db_nodes}'
    ml_settings = PalsMpiexecSettings(
                    'python',
                    exe_args=ml_exe,
                    run_args=None,
                    env_vars=None)
    ml_settings.set_tasks(nprocs)
    ml_settings.set_tasks_per_node(args.ppn)
    ml_settings.set_hostlist(mlNodes)
    ml_settings.set_cpu_binding_type("list:1:8:16:24:32:40:53:60:68:76:84:92")
        
    print("Launching training script ... ")
    ml_model = exp.create_model("train", ml_settings)
    exp.generate(ml_model, overwrite=True)
    exp.start(ml_model, block=True, summary=False)
    print("Done\n")
    
    # Stop database
    print("Stopping the Orchestrator ...")
    exp.stop(db)
    print("Done\n")


## Main function
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="Example", help="Name of the experiment")
    parser.add_argument("--num_pts", type=int, default=10000, help="Number of points in array")
    parser.add_argument("--deployment", type=str, choices=["clustered", "colocated"], default="colocated", help="Deployment type")
    parser.add_argument("--db_nodes", type=int, default=1, help="Number of database nodes")
    parser.add_argument("--ppn", type=int, default=12, help="Number of processes per node")
    args = parser.parse_args()

    # Get nodes of this allocation (job)
    hostfile = os.getenv('PBS_NODEFILE')
    nodelist, nNodes = parseNodeList(hostfile)

    # Call appropriate launcher
    print(f"\nRunning {args.deployment} DB\n")
    if (args.deployment == "colocated"):
        launch_coDB(args, nodelist, nNodes)
    elif (args.deployment == "clustered"):
        launch_clDB(args, nodelist, nNodes)
    else:
        print("\nERROR: Launcher is either colocated or clustered\n")

    # Quit
    print("Quitting")


## Run main
if __name__ == "__main__":
    main()