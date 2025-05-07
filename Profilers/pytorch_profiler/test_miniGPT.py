import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Config, GPT2LMHeadModel
import deepspeed
from torch_setup import get_device, get_device_type, init_distributed,  get_profiler_activities, PPN
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

from torch.profiler import profile
import json
import argparse
import numpy as np


from dftracer.logger import dftracer as PerfTrace, dft_fn as Profile, DFTRACER_ENABLE as DFTRACER_ENABLE

# poor man's data loader                                                                                                                                                                                                      
#data_dir = os.path.join('data', dataset)
#train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
#val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
# 2.1 Synthetic Dataset


dlp = Profile("miniGPT")

class SyntheticTextDataset(Dataset):
    def __init__(self, seq_len, vocab_size, num_samples):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples
    
    @dlp.log
    def __getitem__(self, idx):
        # Random token sequence, shifted for labels
        tokens = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)
        return {"input_ids": tokens, "labels": tokens.clone()}

class TextDataset(Dataset):
    def __init__(self, file, seq_len):
        self.seq_len = seq_len
        self.data = np.memmap(file, dtype=np.uint16, mode='r')
        self.num_samples = len(self.data)//self.seq_len
        if len(self.data) % self.seq_len == 0:
            self.num_samples = self.num_samples - 1

    def __len__(self):
        return self.num_samples

    @dlp.log
    def __getitem__(self, idx):
        # Random token sequence, shifted for labels
        tokens = torch.tensor(self.data[idx*self.seq_len:idx*self.seq_len+self.seq_len], dtype=torch.long)
        labels = torch.tensor(self.data[idx*self.seq_len+1:idx*self.seq_len+self.seq_len+1], dtype=torch.long)
        return {"input_ids": tokens, "labels": labels}

# 2.2 Model Factory
def build_model(vocab_size, seq_len, n_layers=32, hidden_size=512, n_heads=8):
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=seq_len,
        n_ctx=seq_len,
        n_embd=hidden_size,
        n_layer=n_layers,
        n_head=n_heads
    )
    return GPT2LMHeadModel(config)


def train(model, loader, epochs, steps_per_epoch, verbose=False, prof=None):
    for epoch in range(epochs):
        for step, batch in dlp.iter(enumerate(loader)):
            inputs = batch["input_ids"].to(model.local_rank)
            labels = batch["labels"].to(model.local_rank)
            with Profile(cat="compute", name="forward"): 
                outputs = model(inputs, labels=labels)
                loss = outputs.loss
            with Profile(cat="compute", name="backward"): 
                model.backward(loss)
                model.step()
            if prof is not None:
                prof.step()
            if verbose and step % 5 == 0:
                print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")
            #print(f"{step} | {steps_per_epoch}")
            if step >= steps_per_epoch:
                break

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train miniGPT with configurable parallelism")
    parser.add_argument("--tensor-parallel-size", type=int, default=None,
                        help="Tensor parallelism degree (overrides config.json if set)")
    parser.add_argument("--pipeline-parallel-size", type=int, default=None,
                        help="Pipeline parallelism degree (overrides config.json if set)")
    parser.add_argument("--num-layers", type=int, default=32,
                        help="Number of Transformer layers")
    parser.add_argument("--micro-batch-size", type=int, default=8,
                        help="Micro batch size per GPU")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Epochs")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20, help='Number of steps per epoch')
    parser.add_argument("--profile", action="store_true", help="Whether to turn on profiling")
    parser.add_argument("--no-gpu-profile", action="store_true", help="Whether to turn off gpu-profiling")
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--data", type=str, default="/flare/gpu_hack/hzheng/data/train.bin")
    parser.add_argument("--trace-dir", type=str, default="miniGPT_trace")
    parser.add_argument("--synthetic", action="store_true")

    args = parser.parse_args()

    # 1. Initialize DeepSpeed
    ds_config = "ds_config.json"
    #world_size = int(os.environ.get("WORLD_SIZE", "1"))
    dist, rank, world_size = init_distributed()
    #rank = int(os.environ.get("RANK", "0"))

    # 2. Prepare dataset & dataloader
    seq_len = args.sequence_length
    vocab_size = 50257
    if args.synthetic:
        dataset = SyntheticTextDataset(seq_len, vocab_size, num_samples=args.steps*args.micro_batch_size*world_size)
    else:
        dataset = TextDataset(args.data, seq_len)
    loader = DataLoader(dataset, batch_size=args.micro_batch_size, shuffle=True, num_workers=args.num_workers)

    # Turn on 
    PerfTrace.initialize_log(logfile=f"{args.trace_dir}/trace-{rank}-of-{world_size}.pfw", data_dir=f"{args.data}",
                                                   process_id=rank)
    # 3. Build model & wrap with DeepSpeed
    model = build_model(vocab_size, seq_len, n_layers=args.num_layers)
    # Load DeepSpeed config and ensure tensor_parallel is a dict
    with open(ds_config, 'r') as fp:
        ds_config_dict = json.load(fp)
    # Override DeepSpeed config with CLI args if provided
    if args.tensor_parallel_size is not None:
        # apply to 'tensor_parallel' shorthand
        ds_config_dict['tp'] = {'tp_size': args.tensor_parallel_size, 'enabled': True}
    if args.pipeline_parallel_size is not None:
        # ensure pipeline dict exists
        pipeline_cfg = ds_config_dict.get('pipeline', {})
        pipeline_cfg['stages'] = args.pipeline_parallel_size
        pipeline_cfg['enabled'] = True
        ds_config_dict['pipeline'] = pipeline_cfg
    # Handle tensor parallel shorthand or dict under 'tensor_parallel'
    tp_cfg = ds_config_dict.pop('tensor_parallel', None)
    if tp_cfg is not None:
        if isinstance(tp_cfg, int):
            tp_cfg = {'tp_size': tp_cfg, 'enabled': True}
        # Use alias 'tp' for DeepSpeedTPConfig
        ds_config_dict['tp'] = tp_cfg
    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config_params=ds_config_dict,
        model_parameters=model.parameters()
    )

    # 4. Training loop
    model.train()
#    prof = profile(
#        activities=get_profiler_activities(True),
#        schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
#        on_trace_ready=tensorboard_trace_handler('./log/tb_profiler'),
#        record_shapes=True,
#        with_stack=True
#    )
    prof = profile(
                activities=get_profiler_activities(no_gpu = args.no_gpu_profile),
                record_shapes=True,
                profile_memory=True, 
                with_stack=True) 
    dist.barrier()
    # We only do profiling for the first node because multi node profiling will hang
    if args.profile:
        prof.start()
        train(model, loader, epochs=args.epochs, steps_per_epoch = args.steps, verbose=(rank==0), prof=prof)
        prof.stop()
        os.makedirs(args.trace_dir, exist_ok=True)
        if rank == 0:
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))
        prof.export_chrome_trace(f"{args.trace_dir}/torch-trace-{rank}-of-{world_size}.json")
    else:
        train(model, loader, epochs=args.epochs, steps_per_epoch = args.steps, verbose=(rank==0), prof=None)
    dist.barrier()
if __name__ == "__main__":
    main()
