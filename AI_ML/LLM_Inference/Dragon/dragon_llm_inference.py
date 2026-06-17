import dragon
import multiprocessing as mp

from dragon.ai.inference.inference_utils import Inference
from dragon.ai.inference.reader_utils import ReadWorker
from dragon.ai.inference.config import (
    InferenceConfig, 
    ModelConfig, 
    BatchingConfig, 
    HardwareConfig,
    GuardrailsConfig,
    DynamicWorkerConfig
)
from dragon.native.machine import System

from argparse import ArgumentParser
import time
import json


def main():
    start_time = time.time()

    mp.set_start_method("dragon")

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
        required=False,
        type=int,
        default=1,
        help="Batch size for prompt batching.",
    )
    parser.add_argument(
        "--prompt_file",
        required=False,
        type=str,
        default="/tmp/hf_home/prompts.jsonl",
        help="File containing prompts to benchmark with.",
    )
    args = parser.parse_args()

    # Get system info 
    alloc = System()
    num_nodes = int(alloc.nnodes)
    node = alloc.primary_node
    num_gpus = node.num_gpus
    print(
        (f"Number of nodes: {num_nodes}\n"
         f"Number of GPUs per node: {num_gpus}"
        ), 
        flush=True
    )
    
    # Check job sizing is correct
    if args.tp_size > num_gpus:
        print(
            ("Tensor parallel size "
             "must not exceed the number of GPUs per node"
            ),
            flush=True
        )
        return 1

    # Read prompts from file
    try:
        samples = []
        for line in open(args.prompt_file):
            data = json.loads(line)
            samples.append(data["prompt"])
        print(f"Read {len(samples)} prompts from {args.prompt_file}", flush=True)
        samples = samples * (num_gpus // args.tp_size)  * num_nodes
        num_prompts = len(samples)
    except FileNotFoundError:
        print(f"Error: The prompt file {args.prompt_file} was not found.", flush=True)
        return 1

    # Set up Inference Engine config
    batch_type = "pre-batch" # dynamic or pre-batch
    config = InferenceConfig(
        model=ModelConfig(
            model_name=args.model_name,
            hf_token=args.hf_token,
            tp_size=args.tp_size,
            max_tokens=args.max_output_tokens, # max number of output tokens
            max_model_len=8192, # max length of sequence (input+output)
            dtype=args.data_type,
            top_k=50,
            top_p=0.90, # how much to allocate for model + KV cache pool
            system_prompt=["You are a helpful assistant."],
        ),
        hardware=HardwareConfig(
            num_nodes=num_nodes,
            num_gpus=num_gpus,
            num_inf_workers_per_cpu=6,
        ),
        batching=BatchingConfig(
            enabled=True, 
            batch_type=batch_type, 
            batch_wait_seconds=0.1,
            max_batch_size=args.batch_size,
        ),
        guardrails=GuardrailsConfig(enabled=False),
        dynamic_worker=DynamicWorkerConfig(enabled=False),
    )

    # Set up input and output queues
    input_queue = mp.Queue()
    response_queue = mp.Queue()
    rp_event = mp.Event()

    # Start the InferenceEngine
    print(f"\nInitializing pipeline ...", flush=True)
    inference_pipeline = None
    try:
        inference_pipeline = Inference(config, input_queue)
        inference_pipeline.initialize()
        init_time = time.time() - start_time
        print(f"Pipeline initialized successfully! ({init_time:.2f} s)\n", flush=True)
    except Exception as exc:
        import traceback
        print(f"\n[FATAL] Inference pipeline failed to initialize: {exc}", flush=True)
        traceback.print_exc()
        if inference_pipeline is not None:
            inference_pipeline.destroy()
        return 1

    # Spin up read-worker to read responses from inf-worker
    reader_end_ev = mp.Event()
    reader = ReadWorker(response_queue, reader_end_ev)
    reader_proc = mp.Process(target=reader.read, args=(num_prompts,))
    reader_proc.start()

    # Perform inference
    request_start_time = time.time()
    if batch_type == "pre-batch": # prompts are batched by user
        for i in range(0, num_prompts, args.batch_size):
            prompt_batch = samples[i : i + args.batch_size]
            print(
                f"Sending batch of size {len(prompt_batch)} to inference pipeline",
                flush=True,
            )
            inference_pipeline.query((prompt_batch, response_queue))
    elif batch_type == "dynamic": # prompts are batched by engine depending on incoming traffic
        print(f"Sending {num_prompts} prompts", flush=True)
        for prompt in samples:
            inference_pipeline.query((prompt, response_queue))
    rp_event.set()

    # Wait here until all responses are processed by the read-worker
    while not reader_end_ev.is_set():
        continue
    request_end_time = time.time() # FIX: this is not inference time, just request send time I think
    inference_time = request_end_time - request_start_time
    print(f"Completed all inference requests! ({inference_time:.2f} s)", flush=True)

    # Tear down Inference Engine
    print("\nTearing down ...", flush=True)
    teardown_start_time = time.time()
    try:
        inference_pipeline.destroy()
        reader_proc.join()
        response_queue.close()
        input_queue.close()
    except Exception:
        pass
    end_time = time.time()
    teardown_time = end_time - teardown_start_time
    print(f"Teardown completed! ({teardown_time:.2f} s)", flush=True)
    total_runtime = end_time - start_time

    # Print summary of performance stats
    print("\n\n=========================================")
    print("Performance Summary:")
    print(f"Total number of input prompts: {num_prompts}")
    print(f"Total number of successful requests: {num_prompts}")
    print(f"Total run time = {total_runtime:.4f} s")
    print(f"Initialization time (max) = {init_time:.4f} s")
    print(f"Inference time (max) = {inference_time:.4f} s")
    print(f"Total successful requests per second (requests / inference time) : {(num_prompts/inference_time):.4f}")


if __name__ == "__main__":
    main()
    
