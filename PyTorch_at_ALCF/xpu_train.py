# Inspired by https://github.com/huggingface/accelerate

from mpi4py import MPI
import os, socket
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import intel_extension_for_pytorch as ipex
import oneccl_bindings_for_pytorch as torch_ccl
device = torch.device('xpu' if torch.xpu.is_available() else 'cpu')

# DDP: Set environmental variables used by PyTorch
SIZE = MPI.COMM_WORLD.Get_size()
RANK = MPI.COMM_WORLD.Get_rank()
LOCAL_RANK = os.environ.get('PALS_LOCAL_RANKID')
os.environ['RANK'] = str(RANK)
os.environ['WORLD_SIZE'] = str(SIZE)
MASTER_ADDR = socket.gethostname() if RANK == 0 else None
MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
os.environ['MASTER_ADDR'] = MASTER_ADDR
os.environ['MASTER_PORT'] = str(2345)
print(f"DDP: Hi from rank {RANK} of {SIZE} with local rank {LOCAL_RANK}. {MASTER_ADDR}", flush=True)

# DDP: initialize distributed communication with nccl backend
torch.distributed.init_process_group(backend='ccl', init_method='env://', rank=int(RANK), world_size=int(SIZE))

# DDP: pin GPU to local rank.
torch.xpu.set_device(int(LOCAL_RANK))

torch.manual_seed(0)
torch.xpu.manual_seed(0)

src = torch.rand((2048, 1, 512))
tgt = torch.rand((2048, 20, 512))
dataset = torch.utils.data.TensorDataset(src, tgt)

# DDP: use DistributedSampler to partition the training data
sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=SIZE, rank=RANK, seed=0, shuffle=True)
loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=32)

model = torch.nn.Transformer(batch_first=True)
model.to(device)

# DDP: wrap the model in DDP
model = DDP(model)

# DDP: scale learning rate by the number of GPUs.
optimizer = torch.optim.Adam(model.parameters(), lr=(0.001*SIZE))

model.train()
start_t = time.time()
for epoch in range(10):

    # DDP: set epoch to sampler for shuffling
    sampler.set_epoch(epoch)

    for source, targets in loader:
        source = source.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()

        output = model(source, targets)
        loss = torch.nn.functional.cross_entropy(output, targets)

        loss.backward()
        optimizer.step()

if RANK == 0:
    print(f'total train time: {time.time() - start_t:.2f}s', flush=True)

# DDP: cleanup
torch.distributed.destroy_process_group()
