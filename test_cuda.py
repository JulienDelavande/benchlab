import torch
print("CUDA available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0))
a = torch.rand(1024, 1024, device='cuda')
b = a @ a.T
torch.cuda.synchronize()
print("Done")
