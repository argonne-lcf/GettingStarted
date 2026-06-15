import time
from argparse import ArgumentParser
import json
import os
import dataclasses
import socket
from subprocess import check_output, DEVNULL
from typing import Optional
import random

import torch
import vllm
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs

import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI


def find_free_port(
    port_range: tuple[int, int], host: str = "127.0.0.1"
) -> Optional[int]:
    """
    Attempt to find a free port within the given range by binding to it.
    Checks ports in a random order to reduce collisions between concurrent startups.
    """
    # Create a list of all ports in the range and shuffle them
    ports_to_check = list(range(port_range[0], port_range[1]))
    random.shuffle(ports_to_check)

    for port in ports_to_check:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue


def read_scatter_prompts(
    comm: MPI.Comm, fname: str
) -> tuple[int, int, list]:
    """Read file containing LLM prompts, chunk them based on the comm size
    (also the number of inference engines) and scatter the chunks to the ranks
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    num_tot_prompts = None

    # Read prompt file from rank 0
    if rank == 0:
        try:
            prompts = []
            for line in open(fname):
                data = json.loads(line)
                prompts.append(data["prompt"])
            print(f"Read prompts from {fname}", flush=True)
            prompts = prompts * size  # NOTE: only needed for weak scaling
            num_tot_prompts = len(prompts)
        except FileNotFoundError:
            print(f"Error: The prompt file {fname} was not found.", flush=True)
            comm.Abort(1)

        # Split into `size` chunks; distribute the remainder over the first ranks
        base, rem = divmod(num_tot_prompts, size)
        chunks = []
        offset = 0
        for r in range(size):
            n = base + (1 if r < rem else 0)
            chunks.append(prompts[offset : offset + n])
            offset += n
    else:
        chunks = None
    
    num_tot_prompts = comm.bcast(num_tot_prompts, root=0)
    prompts_chunk = comm.scatter(chunks, root=0)
    num_rank_prompts = len(prompts_chunk)
    return num_tot_prompts, num_rank_prompts, prompts_chunk


def set_gpu_affinity(
    comm: MPI.Comm, tp_size: int, log_level: Optional[str] = "info"
) -> None:
    """Detect the number of GPUs on the node and set the GPU affinity
    based on the number of inference instances and TP size
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    local_rank = int(os.getenv("PALS_LOCAL_RANKID", default=rank))
    local_size = int(os.getenv("PALS_LOCAL_SIZE", default=size))

    if torch.cuda.is_available():
        output = check_output(["nvidia-smi", "-L"], stderr=DEVNULL).decode("utf-8").splitlines()
        gpus = list(range(len(output)))
    elif torch.xpu.is_available():
        output = check_output(["xpu-smi", "discovery"], stderr=DEVNULL).decode("utf-8").splitlines()
        hierarchy_mode = os.environ.get("ZE_FLAT_DEVICE_HIERARCHY", "FLAT")
        gpus = []
        for i in range(len(output)):
            if hierarchy_mode == "FLAT":
                gpus.append(i * 2)
                gpus.append(i * 2 + 1)
            elif hierarchy_mode == "COMPOSITE":
                gpus.append(i + 0.0)
                gpus.append(i + 0.1)
    else:
        if rank == 0:
            print("No GPU devices found", flush=True)
            comm.Abort(1)
    if (len(gpus) / (local_size * tp_size)) <= 1:
        msg = (
            f"Not enough GPUs on the node to carry out {local_size} "
            f"inference instances with TP size {tp_size}"
        )
        print(msg, flush=True)
        comm.Abort(1)
    gpus = [str(i) for i in gpus]
    num_gpus_per_instance = tp_size
    start_ind = local_rank * num_gpus_per_instance
    end_ind = (local_rank + 1) * num_gpus_per_instance
    gpus_per_instance = gpus[start_ind:end_ind]
    if log_level == "debug":
        print(f"[{rank}] Setting GPUs {gpus_per_instance}", flush=True)
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus_per_instance)
    elif torch.xpu.is_available():
        os.environ["ZE_AFFINITY_MASK"] = ",".join(gpus_per_instance)


def main():
    start_time = time.time()

    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--hf_token",
        required=True,
        type=str,
        help="Hugging Face token.",
    )
    parser.add_argument(
        "--model_name",
        required=True,
        type=str,
        help="Provide the model name or path to the model you want to capture performance metrics for.",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=1,
        help="Determines the tensor parallel size to use for the run.",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="bfloat16",
        help="Determines the data type to use for the run.",
    )
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        default=128,
        help="Maximum number of tokens in the response.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for prompt batching.",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="../utils/prompts.jsonl",
        help="File containing prompts to benchmark with.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="info",
        choices=["info","debug"],
        help="Logging level for script output.",
    )
    args = parser.parse_args()

    # Initialize MPI
    if not MPI.Is_initialized():
        MPI.Init()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    local_rank = int(os.getenv("PALS_LOCAL_RANKID", default=rank))
    hostname = MPI.Get_processor_name()

    # Read prompts from file on rank 0, then scatter chunks to all ranks
    num_tot_prompts, num_rank_prompts, prompts_chunk = read_scatter_prompts(comm, args.prompt_file)
    
    # Set up the envirnoment variables
    hostname = "127.0.0.1" # localhost
    #hostname = socket.gethostbyname(socket.gethostname())
    port_number = 10000 + local_rank * 200
    port_number = find_free_port((port_number, port_number + 200), hostname)
    if args.log_level == "debug":
        print(f"[{rank}] Found port {port_number}", flush=True)
    os.environ["HF_TOKEN"] = args.hf_token
    os.environ["VLLM_HOST_IP"] = hostname
    os.environ["VLLM_PORT"] = str(port_number)
    os.environ["MASTER_ADDR"] = hostname
    os.environ["MASTER_PORT"] = str(port_number)
    os.environ["RANK"] = str(local_rank % args.tp_size)
    os.environ["WORLD_SIZE"] = str(args.tp_size)
    #os.environ["LOCAL_RANK"] = str(local_rank % args.tp_size)
    #os.environ["LOCAL_WORLD_SIZE"] = str(args.tp_size)
    my_tmp = f"/tmp/vllm_inst_{rank}"
    os.makedirs(my_tmp, exist_ok=True)
    os.environ["TMPDIR"] = my_tmp
    set_gpu_affinity(comm, args.tp_size, args.log_level)

    # Configure the inference engine and sampling parameters
    engine_args = EngineArgs(
        model=args.model_name,
        tensor_parallel_size=args.tp_size,
        enforce_eager=True,
        distributed_executor_backend="mp", # mp, uni
        disable_custom_all_reduce=True,
        dtype=args.data_type,
        gpu_memory_utilization=0.95,
        max_num_seqs=args.batch_size,
        max_model_len=8192, # max length of sequence (input+output)
    )
    sampling_params = SamplingParams(
        max_tokens=args.max_output_tokens
    )
    
    # Start the LLM Inference 
    init_start_time = time.time()
    if rank == 0:
        print(f"\nInitializing Inference Engine ...", flush=True)
    llm = LLM(**dataclasses.asdict(engine_args))
    init_time = time.time() - init_start_time
    print(f"Inference Engine initialized on {hostname}", flush=True)

    # Perform inference
    inf_start_time = time.time()
    outputs = []
    for i in range(0, num_rank_prompts, args.batch_size):
        prompt_batch = prompts_chunk[i : i + args.batch_size]
        print(
            f"Performing inference with {len(prompt_batch)} prompts ({time.time()-inf_start_time} s)",
            flush=True,
        )
        outputs.extend(llm.generate(
                prompt_batch,
                sampling_params,
                #use_tqdm=False,
            )
        )
    inference_time = time.time() - inf_start_time
    comm.Barrier()
    if rank == 0:
        print(f"Completed all inference requests! ({inference_time:.2f} s)", flush=True)

    # Extract responses
    responses = []
    num_successful = 0
    num_failed = 0
    for output in outputs:
        if output.outputs and output.outputs[0].finish_reason in ("stop", "length"):
            num_successful += 1
            responses.append(output.outputs[0].text)
        else:
            num_failed += 1
            responses.append(None)
    rps = num_successful / inference_time
    if args.log_level == "debug":
        print(f"[{rank}] Successful: {num_successful}, Failed: {num_failed}", flush=True)
        #for prompt, response in zip(prompts_chunk,responses):
        #    print(f"[{rank}] PROMPT: {prompt}\t RESPONSE: {response}\n\n", flush=True)

    comm.Barrier()
    end_time = time.time()
    total_runtime = end_time - start_time

    # Gather statistics from ranks
    init_times = comm.allgather(init_time)
    inf_times = comm.allgather(inference_time)
    tot_successful_requests = comm.allgather(num_successful)
    rpss = comm.allgather(rps)

    # Print summary of performance stats
    if rank == 0:
        print("\n\n=========================================")
        print("Performance Summary:")
        print(f"Total number of input prompts: {num_tot_prompts}")
        print(f"Total number of successful requests: {sum(tot_successful_requests)}")
        print(f"Total run time = {total_runtime:.4f} s")
        print(f"Initialization time (min, max, avg) = {min(init_times):.4f}, {max(init_times):.4f}, {sum(init_times)/len(init_times):.4f} s")
        print(f"Inference time (min, max, avg) = {min(inf_times):.4f}, {max(inf_times):.4f}, {sum(inf_times)/len(inf_times):.4f} s")
        print(f"Total successful requests per second (requests / inference time) : {sum(rpss):.4f}")

    MPI.Finalize()

if __name__ == "__main__":
    main()
