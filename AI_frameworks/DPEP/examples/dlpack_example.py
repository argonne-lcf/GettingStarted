import dpnp as dp 
import torch 
import intel_extension_for_pytorch as ipex 

t_ary = torch.arange(4).to('xpu') # array [0, 1, 2, 3] on GPU 
dp_ary = dp.from_dlpack(t_ary) 
t_ary[0] = -2.0 # modify the PyTorch array 
print(f'Original PyTorch array: {t_ary}') 
print(f'dpnp view of PyTorch array: {dp_ary} on device {dp_ary.device}\n')
