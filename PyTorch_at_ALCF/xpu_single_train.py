# Inspired by https://github.com/huggingface/accelerate
import torch
import time

import intel_extension_for_pytorch as ipex
import oneccl_bindings_for_pytorch as torch_ccl

torch.xpu.set_device(0)
device = torch.device('xpu' if torch.xpu.is_available() else 'cpu')
torch.manual_seed(0)
torch.xpu.manual_seed(0)

src = torch.rand((2048, 1, 512), device=f"xpu:{torch.xpu.current_device()}")
tgt = torch.rand((2048, 20, 512), device=f"xpu:{torch.xpu.current_device()}")
dataset = torch.utils.data.TensorDataset(src, tgt)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True)

model = torch.nn.Transformer(batch_first=True)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model, optimizer = ipex.optimize(model, optimizer=optimizer)

model.train()
start_t = time.time()
for epoch in range(2):
    for source, targets in loader:
        source = source.to(device)
        targets = targets.to(device)
        #torch.xpu.set_device(0)
        #print(f"{source.device}")
        print(f"{targets.device}")
        optimizer.zero_grad()

        output = model(source, targets)
        loss = torch.nn.functional.cross_entropy(output, targets)

        loss.backward()
        optimizer.step()

print(f'total train time: {time.time() - start_t:.2f}s')
