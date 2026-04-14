from mpi4py import MPI
import os, socket
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, TensorDataset
import time

# MPI: Get the size and rank
SIZE = MPI.COMM_WORLD.Get_size()
RANK = MPI.COMM_WORLD.Get_rank()
LOCAL_RANK = os.environ.get('PALS_LOCAL_RANKID')

# DDP: Set environment variables used by PyTorch Distributed
os.environ['RANK'] = str(RANK)
os.environ['WORLD_SIZE'] = str(SIZE)
MASTER_ADDR = socket.gethostname() if RANK == 0 else None
MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
os.environ['MASTER_ADDR'] = MASTER_ADDR
os.environ['MASTER_PORT'] = str(2345)
print(f"Hi from rank {RANK} of {SIZE} with local rank {LOCAL_RANK} on node {MASTER_ADDR}", flush=True)

# Set the device
if torch.xpu.is_available():
    device = torch.device('xpu')
    backend = 'xccl'
    WITH_XPU = True
    if RANK == 0:
        print("Found XPU devices!", flush=True)
elif torch.cuda.is_available():
    device = torch.device('cuda')
    backend = 'nccl'
    WITH_CUDA = True
    if RANK == 0:
        print("Found CUDA devices!", flush=True)
else:
    device = torch.device('cpu')
    backend = 'gloo'
    if RANK == 0:
        print("Did not find any device! Defaulting to CPU.", flush=True)

# DDP: pin GPU to local rank.
if WITH_XPU:
    torch.xpu.set_device(int(LOCAL_RANK))
elif WITH_CUDA:
    torch.cuda.set_device(int(LOCAL_RANK))

# DDP: initialize distributed communication with vendor backend
torch.distributed.init_process_group(backend=backend, init_method='env://', rank=int(RANK), world_size=int(SIZE))

# Set the seed
seed = 42
torch.manual_seed(seed)
if WITH_XPU:
    torch.xpu.manual_seed(seed)
elif WITH_CUDA:
    torch.cuda.manual_seed(seed)

# Create a dummy dataset
src = torch.rand((2048, 1, 512), device=device)
tgt = torch.rand((2048, 20, 512), device=device)
dataset = TensorDataset(src, tgt)

# DDP: use DistributedSampler to partition the training data
sampler = DistributedSampler(dataset, num_replicas=SIZE, rank=RANK, shuffle=True)
loader = DataLoader(dataset, sampler=sampler, batch_size=32)

# Create a dummy model
model = torch.nn.Transformer(batch_first=True)
model.to(device)

# DDP: scale learning rate by the number of GPUs.
optimizer = torch.optim.Adam(model.parameters(), lr=(0.001*SIZE))

# DDP: wrap the model in DDP
model = DDP(model)

# Set the loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Start training
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
        loss = loss_fn(output, targets)

        loss.backward()
        optimizer.step()

if RANK == 0:
    print(f'Total training time on {device}: {time.time() - start_t:.2f}s', flush=True)

# DDP: cleanup
torch.distributed.destroy_process_group()
