import argparse
import logging
import os
from typing import TypedDict
import sys


def get_logger(name, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)
    return logger


def create_prompt(nprompts) -> list[str]:
    prompt = "Hi, can you introduce yourself?"
    return [prompt for i in range(nprompts)]


class Args(TypedDict):
    model: str
    host: str
    port: str
    key: str
    num_prompts: int
    cache_dir: str
    tmp_dir: str
    ngpus_per_model: int
    mode: str
    launch: str


def parse_args():
    parser = argparse.ArgumentParser(description="EL inference with vLLM")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name to load",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
        help="Directory where model weights are found",
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
        help="File containing input prompts",
    )
    #parser.add_argument(
    #    "--port",
    #    type=str,
    #    default="8000",
    #    help="Port number for the vLLM server (default: 8000)",
    #)
    #parser.add_argument(
    #    "--key",
    #    type=str,
    #    default="EMPTY",
    #    help="API key for authentication (default: EMPTY)",
    #)
    #parser.add_argument(
    #    "--num-prompts",
    #    type=int,
    #    default=10,
    #    help="Number of prompts to send (default: 1)",
    #)
    #parser.add_argument(
    #    "--tmp-dir",
    #    type=str,
    #    default="/tmp",
    #    help="tmp dir",
    #)
    #parser.add_argument(
    #    "--num-gpus-per-model",
    #    type=int,
    #    default=1,
    #    help="Number of GPUs per model, equal to the tensor parallel size (default: 1)",
    #)
    #parser.add_argument(
    #    "--mode",
    #    type=str,
    #    default="wait",
    #    choices=["wait", "submit"],
    #    help="decide the mode to launch ",
    #)
    #parser.add_argument(
    #    "--launch",
    #    type=str,
    #    default="mpi",
    #    choices=["mpi", "ssh"],
    #    help="method to launch vllm servers on multi-node (default: mpi)",
    #)
    #parser.add_argument(
    #    "--num-gpus-per-node",
    #    type=int,
    #    default=12,
    #    help="Number of GPUs per node (default: 12 as on Aurora)",
    #)
    #parser.add_argument(
    #    "--num-cpus-per-node",
    #    type=int,
    #    default=104,
    #    help="Number of CPUs per node (default: 104 as on Aurora)",
    #)
    #parser.add_argument(
    #    "--pre-build-vllm-cache",
    #    type=int,
    #    default=0,
    #)
    #parser.add_argument(
    #    "--send-timeout",
    #    type=float,
    #    default=5.0,
    #)
    #parser.add_argument(
    #    "--send-retries",
    #    type=int,
    #    default=3,
    #)
    #parser.add_argument(
    #    "--actors-per-process",
    #    type=int,
    #    default=384,
    #)

    args = parser.parse_args()
    args_dict = Args(**(vars(args)))
    return args_dict
