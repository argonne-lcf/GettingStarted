# Inspired by https://github.com/huggingface/accelerate

import torch
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(0)
torch.cuda.manual_seed(0)

src = torch.rand((2048, 1, 512))
tgt = torch.rand((2048, 20, 512))
dataset = torch.utils.data.TensorDataset(src, tgt)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

model = torch.nn.Transformer(batch_first=True)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
start_t = time.time()
for epoch in range(10):
    for source, targets in loader:
        source = source.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()

        output = model(source, targets)
        loss = torch.nn.functional.cross_entropy(output, targets)

        loss.backward()
        optimizer.step()

print(f'total train time: {time.time() - start_t:.2f}s')
